# TorchScript C++ Integration Guide for ParticleNet

This document describes how to integrate TorchScript-exported ParticleNet models into C++ applications, specifically for SKNanoAnalyzer.

## Table of Contents

1. [Model Overview](#model-overview)
2. [Input Format Specification](#input-format-specification)
3. [C++ Loading and Inference](#c-loading-and-inference)
4. [Edge Construction](#edge-construction)
5. [Output Interpretation](#output-interpretation)
6. [Complete Example](#complete-example)
7. [Performance Considerations](#performance-considerations)

---

## Model Overview

The exported TorchScript models are multiclass ParticleNet classifiers for charged Higgs searches:

- **Architecture**: 3 DynamicEdgeConv layers + global pooling + dense layers
- **Input**: Particle-level features + graph structure + era encoding
- **Output**: 4-class probabilities (signal + 3 background categories)
- **Framework**: PyTorch C++ API (libtorch)

### Model Specifications

| Parameter | Value | Description |
|-----------|-------|-------------|
| Node features | 9 | Particle kinematics and type flags |
| Graph features | 4 | Era encoding (one-hot) |
| Output classes | 4 | Signal, nonprompt, diboson, ttZ |
| Hidden units | 128 | Internal representation size |
| k-NN neighbors | 4 | Graph connectivity |

---

## Input Format Specification

The model expects **4 input tensors** in the following order:

### 1. Node Features (`x`)
- **Type**: `torch::Tensor` (float32)
- **Shape**: `[N_particles, 9]`
- **Features** (in order):
  1. **E**: Energy (GeV)
  2. **Px**: Momentum x-component (GeV)
  3. **Py**: Momentum y-component (GeV)
  4. **Pz**: Momentum z-component (GeV)
  5. **Charge**: Electric charge (-1, 0, +1)
  6. **IsMuon**: Binary flag (0 or 1)
  7. **IsElectron**: Binary flag (0 or 1)
  8. **IsJet**: Binary flag (0 or 1)
  9. **IsBjet**: Binary flag for b-tagged jets (0 or 1)

**Important Notes**:
- All particles (leptons, jets, bjets) are included in a single tensor
- Binary flags are mutually exclusive (each particle has exactly one type flag = 1)
- Features should be normalized similarly to training data
- Typical event has 10-30 particles (leptons + jets + bjets)

### 2. Edge Index (`edge_index`)
- **Type**: `torch::Tensor` (int64)
- **Shape**: `[2, N_edges]`
- **Content**: k-NN graph structure (k=4)
- **Format**: COO (coordinate) format
  - Row 0: Source node indices
  - Row 1: Target node indices

**Construction**:
- Must compute k=4 nearest neighbors in 9D feature space
- Use Euclidean distance in (E, Px, Py, Pz, Charge, IsMuon, IsElectron, IsJet, IsBjet)
- See [Edge Construction](#edge-construction) for implementation details

### 3. Graph Features (`graph_input`)
- **Type**: `torch::Tensor` (float32)
- **Shape**: `[1, 4]`
- **Content**: Era encoding (one-hot vector)
- **Encodings**:
  - Run2: `[1.0, 0.0, 0.0, 0.0]`
  - Run3: `[0.0, 1.0, 0.0, 0.0]`
  - (Remaining dimensions reserved for future eras)

### 4. Batch Assignment (`batch`)
- **Type**: `torch::Tensor` (int64)
- **Shape**: `[N_particles]`
- **Content**: All zeros for single-event inference
- **Purpose**: Required by PyTorch Geometric layers (even for single graphs)

**Example**:
```cpp
auto batch = torch::zeros({N_particles}, torch::kInt64);
```

---

## C++ Loading and Inference

### Required Dependencies

```cmake
# CMakeLists.txt
find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS})
target_link_libraries(your_target ${TORCH_LIBRARIES})
```

### Basic Loading

```cpp
#include <torch/script.h>
#include <torch/torch.h>

// Load TorchScript model
torch::jit::script::Module model;
try {
    model = torch::jit::load("ParticleNet_scripted.pt");
    model.eval();  // Set to evaluation mode
    std::cout << "Model loaded successfully" << std::endl;
} catch (const c10::Error& e) {
    std::cerr << "Error loading model: " << e.what() << std::endl;
    return -1;
}
```

### Inference

```cpp
// Prepare inputs (see Input Format Specification)
auto x = /* construct node features */;
auto edge_index = /* construct k-NN graph */;
auto graph_input = torch::tensor({{1.0, 0.0, 0.0, 0.0}});  // Run2
auto batch = torch::zeros({x.size(0)}, torch::kInt64);

// Create input vector
std::vector<torch::jit::IValue> inputs;
inputs.push_back(x);
inputs.push_back(edge_index);
inputs.push_back(graph_input);
inputs.push_back(batch);

// Run inference
torch::Tensor output;
{
    torch::NoGradGuard no_grad;  // Disable gradient computation
    output = model.forward(inputs).toTensor();
}

// Output shape: [1, 4] (batch_size=1, num_classes=4)
```

---

## Edge Construction

The model requires k=4 nearest neighbor graph in 9D feature space.

### Option 1: Brute Force (Simple, Sufficient for Small N)

```cpp
#include <vector>
#include <algorithm>
#include <cmath>

// Compute Euclidean distance between two particles
float euclidean_distance(const std::vector<float>& a, const std::vector<float>& b) {
    float dist = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        float diff = a[i] - b[i];
        dist += diff * diff;
    }
    return std::sqrt(dist);
}

// Construct k-NN graph (k=4)
torch::Tensor construct_knn_graph(const torch::Tensor& x, int k = 4) {
    int N = x.size(0);  // Number of particles
    std::vector<std::pair<int, int>> edges;

    // Convert tensor to vector for easier manipulation
    auto x_accessor = x.accessor<float, 2>();
    std::vector<std::vector<float>> features(N);
    for (int i = 0; i < N; ++i) {
        features[i].resize(9);
        for (int j = 0; j < 9; ++j) {
            features[i][j] = x_accessor[i][j];
        }
    }

    // For each particle, find k nearest neighbors
    for (int i = 0; i < N; ++i) {
        std::vector<std::pair<float, int>> distances;

        for (int j = 0; j < N; ++j) {
            if (i != j) {  // Exclude self-loops
                float dist = euclidean_distance(features[i], features[j]);
                distances.push_back({dist, j});
            }
        }

        // Sort by distance and take k nearest
        std::sort(distances.begin(), distances.end());
        int num_neighbors = std::min(k, (int)distances.size());

        for (int n = 0; n < num_neighbors; ++n) {
            edges.push_back({i, distances[n].second});
        }
    }

    // Convert to edge_index tensor [2, N_edges]
    int num_edges = edges.size();
    auto edge_index = torch::empty({2, num_edges}, torch::kInt64);
    auto edge_accessor = edge_index.accessor<int64_t, 2>();

    for (int i = 0; i < num_edges; ++i) {
        edge_accessor[0][i] = edges[i].first;   // Source
        edge_accessor[1][i] = edges[i].second;  // Target
    }

    return edge_index;
}
```

### Option 2: ROOT TKDTree (Optimized for Large N)

```cpp
#include "TKDTree.h"

torch::Tensor construct_knn_graph_root(const torch::Tensor& x, int k = 4) {
    int N = x.size(0);
    int dim = 9;

    // Convert tensor to ROOT format
    std::vector<double> data(N * dim);
    auto x_accessor = x.accessor<float, 2>();
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < dim; ++j) {
            data[i * dim + j] = x_accessor[i][j];
        }
    }

    // Build KD-tree
    TKDTree<int, double> kdtree(N, dim, data.data());
    kdtree.Build();

    std::vector<std::pair<int, int>> edges;

    // Find k nearest neighbors for each particle
    for (int i = 0; i < N; ++i) {
        std::vector<double> point(dim);
        for (int j = 0; j < dim; ++j) {
            point[j] = x_accessor[i][j];
        }

        // Find k+1 neighbors (including self)
        std::vector<int> neighbors(k + 1);
        kdtree.FindNearestNeighbors(point.data(), k + 1, neighbors.data());

        // Add edges (excluding self)
        for (int n = 0; n < k + 1; ++n) {
            if (neighbors[n] != i) {
                edges.push_back({i, neighbors[n]});
            }
        }
    }

    // Convert to tensor
    int num_edges = edges.size();
    auto edge_index = torch::empty({2, num_edges}, torch::kInt64);
    auto edge_accessor = edge_index.accessor<int64_t, 2>();

    for (int i = 0; i < num_edges; ++i) {
        edge_accessor[0][i] = edges[i].first;
        edge_accessor[1][i] = edges[i].second;
    }

    return edge_index;
}
```

---

## Output Interpretation

### Output Format

The model outputs **logits** (unnormalized log-probabilities):

- **Shape**: `[1, 4]` (single event, 4 classes)
- **Classes**: `[signal, nonprompt, diboson, ttZ]`

### Converting to Probabilities

```cpp
// Apply softmax to get probabilities
auto probabilities = torch::softmax(output, /*dim=*/1);

// Access individual class probabilities
auto prob_accessor = probabilities.accessor<float, 2>();
float prob_signal = prob_accessor[0][0];
float prob_nonprompt = prob_accessor[0][1];
float prob_diboson = prob_accessor[0][2];
float prob_ttZ = prob_accessor[0][3];

std::cout << "Signal probability: " << prob_signal << std::endl;
```

### Extracting Predicted Class

```cpp
// Get predicted class (argmax)
auto predicted_class = std::get<1>(torch::max(probabilities, /*dim=*/1));
int class_id = predicted_class.item<int>();

std::vector<std::string> class_names = {"signal", "nonprompt", "diboson", "ttZ"};
std::cout << "Predicted class: " << class_names[class_id] << std::endl;
```

---

## Complete Example

```cpp
#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>
#include <vector>

// Helper function to construct node features from physics objects
torch::Tensor construct_node_features(
    const std::vector<TLorentzVector>& particles,
    const std::vector<int>& charges,
    const std::vector<int>& types)  // 0=muon, 1=electron, 2=jet, 3=bjet
{
    int N = particles.size();
    auto x = torch::empty({N, 9}, torch::kFloat32);
    auto x_accessor = x.accessor<float, 2>();

    for (int i = 0; i < N; ++i) {
        x_accessor[i][0] = particles[i].E();
        x_accessor[i][1] = particles[i].Px();
        x_accessor[i][2] = particles[i].Py();
        x_accessor[i][3] = particles[i].Pz();
        x_accessor[i][4] = charges[i];
        x_accessor[i][5] = (types[i] == 0) ? 1.0 : 0.0;  // IsMuon
        x_accessor[i][6] = (types[i] == 1) ? 1.0 : 0.0;  // IsElectron
        x_accessor[i][7] = (types[i] == 2) ? 1.0 : 0.0;  // IsJet
        x_accessor[i][8] = (types[i] == 3) ? 1.0 : 0.0;  // IsBjet
    }

    return x;
}

// Main inference function
void run_particlenet_inference(
    const std::vector<TLorentzVector>& particles,
    const std::vector<int>& charges,
    const std::vector<int>& types,
    const std::string& model_path,
    const std::string& era)  // "Run2" or "Run3"
{
    // Load model
    torch::jit::script::Module model = torch::jit::load(model_path);
    model.eval();

    // Construct inputs
    auto x = construct_node_features(particles, charges, types);
    auto edge_index = construct_knn_graph(x, 4);
    auto graph_input = (era == "Run2") ?
        torch::tensor({{1.0, 0.0, 0.0, 0.0}}) :
        torch::tensor({{0.0, 1.0, 0.0, 0.0}});
    auto batch = torch::zeros({x.size(0)}, torch::kInt64);

    // Prepare inputs
    std::vector<torch::jit::IValue> inputs = {x, edge_index, graph_input, batch};

    // Run inference
    torch::Tensor output;
    {
        torch::NoGradGuard no_grad;
        output = model.forward(inputs).toTensor();
    }

    // Get probabilities
    auto probabilities = torch::softmax(output, 1);
    auto prob_accessor = probabilities.accessor<float, 2>();

    // Print results
    std::cout << "ParticleNet Classification Results:" << std::endl;
    std::cout << "  Signal:    " << prob_accessor[0][0] << std::endl;
    std::cout << "  Nonprompt: " << prob_accessor[0][1] << std::endl;
    std::cout << "  Diboson:   " << prob_accessor[0][2] << std::endl;
    std::cout << "  ttZ:       " << prob_accessor[0][3] << std::endl;

    // Predicted class
    auto predicted_class = std::get<1>(torch::max(probabilities, 1));
    std::vector<std::string> class_names = {"signal", "nonprompt", "diboson", "ttZ"};
    std::cout << "  Predicted: " << class_names[predicted_class.item<int>()] << std::endl;
}
```

---

## Performance Considerations

### Optimization Tips

1. **Model Loading**: Load model once and reuse for multiple events
   ```cpp
   // Load once (expensive)
   static torch::jit::script::Module model = torch::jit::load(model_path);

   // Reuse for all events (cheap)
   auto output = model.forward(inputs).toTensor();
   ```

2. **CPU vs GPU**: Models can run on CPU or GPU
   ```cpp
   // Move model to GPU if available
   if (torch::cuda::is_available()) {
       model.to(torch::kCUDA);
       x = x.to(torch::kCUDA);
       edge_index = edge_index.to(torch::kCUDA);
       // ... move other inputs
   }
   ```

3. **Batch Processing**: For multiple events, increase batch size
   ```cpp
   // Combine multiple events in single batch
   // Requires proper batch assignment tensor
   ```

4. **Edge Caching**: If particle configuration repeats, cache edge_index
   ```cpp
   static std::map<int, torch::Tensor> edge_cache;
   int N = x.size(0);
   if (edge_cache.find(N) == edge_cache.end()) {
       edge_cache[N] = construct_knn_graph(x, 4);
   }
   auto edge_index = edge_cache[N];
   ```

### Typical Performance

- **Single event inference**: ~1-5 ms (CPU), ~0.5-2 ms (GPU)
- **Edge construction**: ~0.1-1 ms (depends on N and algorithm)
- **Model size**: ~1.2 MB per model

---

## Troubleshooting

### Common Issues

1. **Shape mismatch errors**:
   - Verify input tensor shapes match specification
   - Check edge_index is `[2, N_edges]` not `[N_edges, 2]`

2. **Type errors**:
   - Ensure `edge_index` and `batch` are `torch::kInt64`
   - Ensure `x` and `graph_input` are `torch::kFloat32`

3. **Runtime errors**:
   - Verify model was exported with same PyTorch version
   - Check libtorch ABI compatibility

4. **Numerical differences**:
   - Small differences (~1e-6) are normal due to floating point
   - Large differences indicate input format mismatch

---

## References

- [PyTorch C++ API Documentation](https://pytorch.org/cppdocs/)
- [TorchScript Documentation](https://pytorch.org/docs/stable/jit.html)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)

---

**Last Updated**: 2025-10-08
**Authors**: Claude Code + Human
**Version**: 1.0

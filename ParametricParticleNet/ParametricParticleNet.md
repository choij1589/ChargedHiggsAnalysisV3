# ParametricParticleNet Implementation Plan

## Overview
ParametricParticleNet is an enhanced version of ParticleNet that uses (mHc, mA) mass parameters as conditional inputs, enabling a single model to handle multiple signal mass points for charged Higgs boson analysis.

## Architecture Design

### Core Concept
- **Single model** for all mass points instead of individual models per mass
- **Mass-conditioned** network using attention-based mechanisms
- **Interpolation capability** for unseen mass points
- **Multi-stage training** for optimal generalization

### Mass-Aware Attention Mechanism

```python
class MassAwareAttention(nn.Module):
    """Attention mechanism conditioned on (mHc, mA) parameters"""

    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim

        # Mass embedding network
        self.mass_encoder = nn.Sequential(
            nn.Linear(2, 64),  # [mHc, mA] → 64
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, dim * 3)  # Generate Q, K, V modulations
        )

        # Standard attention components
        self.qkv = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x, mHc, mA):
        # Encode mass parameters
        mass_vec = torch.stack([mHc, mA], dim=-1)
        mass_modulation = self.mass_encoder(mass_vec)
        q_mod, k_mod, v_mod = mass_modulation.chunk(3, dim=-1)

        # Generate mass-conditioned Q, K, V
        # Apply mass-dependent modulation
        # Compute attention with mass conditioning

        return self.out_proj(x)
```

### ParametricParticleNet Model

```python
class ParametricParticleNet(nn.Module):
    """ParticleNet with mass-parametric conditioning"""

    def __init__(self, input_dim, n_classes, hidden_dims=[128, 256, 512]):
        super().__init__()

        # Mass normalization parameters
        self.register_buffer('mHc_mean', torch.tensor(130.0))
        self.register_buffer('mHc_std', torch.tensor(20.0))
        self.register_buffer('mA_mean', torch.tensor(90.0))
        self.register_buffer('mA_std', torch.tensor(10.0))

        # Edge convolution blocks (standard ParticleNet)
        self.edge_conv_blocks = nn.ModuleList()

        # Mass-aware attention blocks at multiple scales
        self.mass_attention_1 = MassAwareAttention(hidden_dims[0])
        self.mass_attention_2 = MassAwareAttention(hidden_dims[1])
        self.mass_attention_3 = MassAwareAttention(hidden_dims[2])

        # Output classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dims[-1], 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, n_classes)
        )

    def forward(self, data, mHc, mA):
        # Normalize mass values
        mHc_norm = (mHc - self.mHc_mean) / self.mHc_std
        mA_norm = (mA - self.mA_mean) / self.mA_std

        # Apply mass conditioning throughout network
        x = self.process_with_mass_conditioning(data, mHc_norm, mA_norm)

        return self.classifier(x)
```

## Dataset Strategy

### Dynamic Loading Architecture

```python
class DynamicMassDataset(Dataset):
    """Dynamically loads different mass point datasets with caching"""

    def __init__(self, mass_points, channel, fold, cache_size=3):
        self.mass_points = mass_points  # List of (mHc, mA) tuples
        self.channel = channel
        self.fold = fold
        self.cache = {}  # LRU cache for loaded datasets
        self.cache_size = cache_size

        # Create global index mapping
        self.index_map = []
        for mHc, mA in mass_points:
            dataset_path = self._get_dataset_path(mHc, mA)
            dataset_size = self._get_dataset_size(dataset_path)
            for idx in range(dataset_size):
                self.index_map.append((mHc, mA, idx))

    def _load_dataset(self, mHc, mA):
        """Load dataset with LRU caching"""
        key = (mHc, mA)
        if key not in self.cache:
            if len(self.cache) >= self.cache_size:
                # Evict least recently used
                oldest = min(self.cache, key=lambda k: self.cache[k]['access_time'])
                del self.cache[oldest]

            # Load new dataset
            path = self._get_dataset_path(mHc, mA)
            dataset = torch.load(path)
            self.cache[key] = {
                'data': dataset,
                'access_time': time.time()
            }

        self.cache[key]['access_time'] = time.time()
        return self.cache[key]['data']
```

### Mass Point Organization

| Mass Point | mHc | mA | Status | Purpose |
|------------|-----|----|--------|---------|
| MHc100_MA85 | 100 | 85 | Training | Low mass regime |
| MHc100_MA95 | 100 | 95 | Testing | Edge case (mA close to mHc) |
| MHc115_MA85 | 115 | 85 | Training | Large mass gap |
| MHc115_MA90 | 115 | 90 | Training | Standard point |
| MHc130_MA85 | 130 | 85 | Training | Very large gap |
| MHc130_MA90 | 130 | 90 | Training | Central point |
| MHc130_MA95 | 130 | 95 | Training | Small gap |
| MHc145_MA90 | 145 | 90 | Training | High mass, large gap |
| MHc145_MA95 | 145 | 95 | Training | High mass, medium gap |
| MHc160_MA85 | 160 | 85 | Training | Maximum gap |
| MHc160_MA90 | 160 | 90 | Training | High mass regime |
| MHc160_MA95 | 160 | 95 | Training | High mass, small gap |

## Training Pipeline

### Multi-Stage Training Strategy

#### Stage 1: Individual Pre-training (Epochs 1-20)
- Train separate instances on each mass point
- Freeze mass parameters to their true values
- Learn mass-specific features
- Save individual model weights

#### Stage 2: Weight Averaging (Initialization)
- Average weights from all pre-trained models
- Initialize joint model with averaged weights
- Provides better starting point than random initialization

#### Stage 3: Joint Fine-tuning (Epochs 21-70)
- Train on all mass points simultaneously
- Dynamic batch sampling across masses
- Mass parameters as conditional inputs
- Curriculum learning: easy → hard masses

### Training Script

```python
def multi_stage_training(args):
    """Complete multi-stage training pipeline"""

    # Initialize model
    model = ParametricParticleNet(
        input_dim=args.input_dim,
        n_classes=args.n_classes
    ).to(args.device)

    # Stage 1: Individual pre-training
    pretrained_weights = {}
    for mHc, mA in args.mass_points:
        print(f"Pre-training on MHc{mHc}_MA{mA}")
        weights = pretrain_single_mass(model, mHc, mA, args)
        pretrained_weights[(mHc, mA)] = weights

    # Stage 2: Weight averaging
    averaged_weights = average_model_weights(pretrained_weights)
    model.load_state_dict(averaged_weights)

    # Stage 3: Joint fine-tuning
    dataset = DynamicMassDataset(args.mass_points, args.channel, args.fold)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.finetune_lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.finetune_epochs)

    for epoch in range(args.finetune_epochs):
        train_loss = train_epoch(model, loader, optimizer)
        val_loss = validate_all_masses(model, args.mass_points)

        scheduler.step()

        if val_loss < best_val_loss:
            torch.save(model.state_dict(), 'models/parametric_best.pth')

    return model
```

### Loss Function

```python
def physics_informed_loss(predictions, targets, mHc, mA, weights):
    """Loss function with physics-based weighting"""

    # Base cross-entropy loss
    ce_loss = F.cross_entropy(predictions, targets, reduction='none')

    # Apply event weights
    weighted_loss = ce_loss * weights

    # Mass-dependent difficulty weighting
    # Harder to distinguish when mA approaches mHc
    mass_gap = (mHc - mA) / mHc  # Normalized mass gap
    difficulty = 1.0 / (mass_gap + 0.1)  # Higher weight for smaller gaps

    # Final loss
    return (weighted_loss * difficulty).mean()
```

## Evaluation Strategy

### Performance Metrics

1. **Per-Mass Performance**
   - ROC curves for each training mass point
   - Compare to dedicated single-mass models
   - Signal efficiency at fixed background rejection

2. **Interpolation Testing**
   - Test on mass points not in training set
   - Example: Train on (130, 90), test on (125, 88)
   - Measure performance degradation

3. **Extrapolation Boundaries**
   - Test outside training mass range
   - Identify model confidence limits
   - Generate uncertainty estimates

4. **Mass Plane Visualization**
   - 2D heatmap of model performance
   - Axes: mHc vs mA
   - Color: ROC AUC or signal efficiency

### Evaluation Scripts

```python
# python/evalParametric.py

def evaluate_interpolation(model, test_masses, args):
    """Test model on unseen mass points"""

    results = {}
    for mHc, mA in test_masses:
        # Load test data for this mass
        test_data = load_test_data(mHc, mA, args.channel)

        # Evaluate
        with torch.no_grad():
            predictions = model(test_data, mHc, mA)
            metrics = calculate_metrics(predictions, test_data.y)

        results[(mHc, mA)] = {
            'auc': metrics['auc'],
            'efficiency': metrics['signal_eff_at_1pct_bkg'],
            'accuracy': metrics['accuracy']
        }

    return results

def generate_mass_plane_heatmap(model, args):
    """Generate 2D performance heatmap over (mHc, mA) plane"""

    mHc_range = np.arange(100, 165, 5)
    mA_range = np.arange(80, 101, 2)

    performance_grid = np.zeros((len(mHc_range), len(mA_range)))

    for i, mHc in enumerate(mHc_range):
        for j, mA in enumerate(mA_range):
            if mA < mHc:  # Physical constraint
                auc = evaluate_single_point(model, mHc, mA, args)
                performance_grid[i, j] = auc
            else:
                performance_grid[i, j] = np.nan

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(performance_grid, aspect='auto', origin='lower',
               extent=[mA_range[0], mA_range[-1], mHc_range[0], mHc_range[-1]])
    plt.colorbar(label='ROC AUC')
    plt.xlabel('mA [GeV]')
    plt.ylabel('mHc [GeV]')
    plt.title('ParametricParticleNet Performance')

    # Mark training points
    for mHc, mA in args.training_masses:
        plt.plot(mA, mHc, 'r*', markersize=10)

    plt.savefig('plots/parametric_performance_heatmap.png')
```

## Implementation Workflow

### Phase 1: Foundation (Week 1)
- [ ] Create ParametricParticleNet model class
- [ ] Implement mass-aware attention mechanism
- [ ] Set up dynamic dataset loader
- [ ] Create basic training loop

### Phase 2: Training Infrastructure (Week 2)
- [ ] Implement multi-stage training
- [ ] Add curriculum learning
- [ ] Set up distributed training
- [ ] Implement checkpointing

### Phase 3: Evaluation & Optimization (Week 3)
- [ ] Create evaluation scripts
- [ ] Generate performance visualizations
- [ ] Compare with dedicated models
- [ ] Optimize hyperparameters

### Phase 4: Integration (Week 4)
- [ ] Integrate with GA optimization
- [ ] Export models for production
- [ ] Create deployment scripts
- [ ] Write documentation

## Directory Structure

```
ParametricParticleNet/
├── configs/
│   ├── model_config.json       # Model architecture parameters
│   ├── training_config.json    # Training hyperparameters
│   └── mass_points.json        # Mass point definitions
├── data/
│   └── cache/                  # Dataset cache directory
├── models/
│   ├── pretrained/             # Stage 1 individual models
│   ├── checkpoints/            # Training checkpoints
│   └── final/                  # Production models
├── python/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── ParametricParticleNet.py
│   │   └── MassAwareAttention.py
│   ├── datasets/
│   │   ├── DynamicMassDataset.py
│   │   └── MassAugmentation.py
│   ├── training/
│   │   ├── multi_stage.py
│   │   ├── curriculum.py
│   │   └── losses.py
│   ├── evaluation/
│   │   ├── interpolation.py
│   │   ├── visualization.py
│   │   └── metrics.py
│   └── utils/
│       ├── config.py
│       └── helpers.py
├── scripts/
│   ├── prepare_data.sh         # Dataset preparation
│   ├── train_parametric.sh     # Training pipeline
│   ├── evaluate.sh             # Evaluation pipeline
│   └── deploy.sh               # Model deployment
├── plots/
│   ├── training/               # Training curves
│   ├── evaluation/             # ROC curves, etc.
│   └── heatmaps/               # Mass plane visualizations
├── logs/
│   └── training/               # Training logs
└── README.md                   # Documentation

```

## Configuration Files

### configs/model_config.json
```json
{
  "architecture": {
    "input_dim": 14,
    "hidden_dims": [128, 256, 512],
    "n_heads": 8,
    "n_classes": 4,
    "dropout": 0.5,
    "activation": "relu"
  },
  "mass_conditioning": {
    "type": "attention",
    "embedding_dim": 128,
    "normalize": true,
    "normalization_stats": {
      "mHc": {"mean": 130, "std": 20},
      "mA": {"mean": 90, "std": 10}
    }
  }
}
```

### configs/training_config.json
```json
{
  "stage1": {
    "epochs": 20,
    "batch_size": 512,
    "lr": 0.001,
    "optimizer": "Adam"
  },
  "stage2": {
    "averaging_method": "mean"
  },
  "stage3": {
    "epochs": 50,
    "batch_size": 1024,
    "lr": 0.0001,
    "optimizer": "AdamW",
    "scheduler": "CosineAnnealing",
    "weight_decay": 0.01
  },
  "curriculum": {
    "enabled": true,
    "stages": [
      {"epochs": 10, "masses": "easy"},
      {"epochs": 20, "masses": "medium"},
      {"epochs": 20, "masses": "all"}
    ]
  }
}
```

### configs/mass_points.json
```json
{
  "training": [
    {"mHc": 100, "mA": 85},
    {"mHc": 115, "mA": 90},
    {"mHc": 130, "mA": 85},
    {"mHc": 130, "mA": 90},
    {"mHc": 130, "mA": 95},
    {"mHc": 145, "mA": 90},
    {"mHc": 160, "mA": 95}
  ],
  "validation": [
    {"mHc": 115, "mA": 85},
    {"mHc": 145, "mA": 95},
    {"mHc": 160, "mA": 90}
  ],
  "test_interpolation": [
    {"mHc": 110, "mA": 87},
    {"mHc": 125, "mA": 91},
    {"mHc": 140, "mA": 93}
  ],
  "test_extrapolation": [
    {"mHc": 95, "mA": 80},
    {"mHc": 165, "mA": 97}
  ]
}
```

## Command Line Interface

### Quick Start
```bash
# Prepare all required datasets
./scripts/prepare_data.sh --all-masses

# Run training with default configuration
./scripts/train_parametric.sh --channel Run1E2Mu

# Evaluate on test masses
./scripts/evaluate.sh --mode interpolation

# Generate performance visualizations
python python/evaluation/visualization.py --heatmap
```

### Advanced Usage
```bash
# Custom training configuration
python python/training/multi_stage.py \
  --config configs/custom_training.json \
  --mass-points "100,85;130,90;160,95" \
  --device cuda:0 \
  --num-workers 8

# Evaluate specific model
python python/evaluation/interpolation.py \
  --model models/final/parametric_v1.pth \
  --test-masses "125,88;135,92" \
  --output results/interpolation_test.json

# Compare with dedicated models
python python/evaluation/compare.py \
  --parametric models/final/parametric_v1.pth \
  --dedicated models/dedicated/ \
  --mass-points "all"
```

## Expected Benefits

| Metric | Current (Per-Mass) | ParametricParticleNet | Improvement |
|--------|-------------------|----------------------|-------------|
| **Models** | 12 separate | 1 unified | 12× reduction |
| **Storage** | ~1.2 GB | ~100 MB | 12× smaller |
| **Training Time** | 12 × 4h = 48h | 8h total | 6× faster |
| **Maintenance** | Update 12 models | Update 1 model | 12× easier |
| **New Mass Points** | Retrain from scratch | Interpolate/fine-tune | Instant/Fast |
| **Performance** | Baseline | ~95% of dedicated | Acceptable trade-off |

## Success Criteria

1. **Performance**: ≥95% AUC compared to dedicated models
2. **Interpolation**: ≥90% AUC on unseen mass points
3. **Training Time**: <10 hours for complete pipeline
4. **Stability**: Consistent results across random seeds
5. **Scalability**: Handles 15+ mass points efficiently

## References

- Original ParticleNet paper: [arXiv:1902.08570](https://arxiv.org/abs/1902.08570)
- FiLM conditioning: [arXiv:1709.07871](https://arxiv.org/abs/1709.07871)
- Attention mechanisms: [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- Curriculum learning: [ICML 2009](https://dl.acm.org/doi/10.1145/1553374.1553380)

## Next Steps

1. **Immediate**: Create project directory structure
2. **Week 1**: Implement core model and datasets
3. **Week 2**: Set up training pipeline
4. **Week 3**: Evaluation and optimization
5. **Week 4**: Integration and deployment
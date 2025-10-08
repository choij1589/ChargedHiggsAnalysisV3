# TorchScript Export Summary

**Date**: 2025-10-08
**Status**: ✅ All 6 models successfully exported and validated

## Exported Models

All models are from **fold-3** with **3 background classes** (nonprompt, diboson, ttZ).

### Run1E2Mu Channel (3 models)
1. `TTToHcToWAToMuMu-MHc160_MA85` (1.3 MB)
2. `TTToHcToWAToMuMu-MHc130_MA90` (1.3 MB)
3. `TTToHcToWAToMuMu-MHc100_MA95` (1.3 MB)

### Run3Mu Channel (3 models)
4. `TTToHcToWAToMuMu-MHc160_MA85` (1.3 MB)
5. `TTToHcToWAToMuMu-MHc130_MA90` (1.3 MB)
6. `TTToHcToWAToMuMu-MHc100_MA95` (1.3 MB)

## Model Specifications

| Parameter | Value | Description |
|-----------|-------|-------------|
| Node features | 9 | **E, Px, Py, Pz, Charge, IsMuon, IsElectron, IsJet, IsBjet** |
| Graph features | 4 | Era encoding (one-hot) |
| Output classes | 4 | Signal, nonprompt, diboson, ttZ |
| Hidden units | 128 | Internal representation size |
| Total parameters | 272,807 | Per model |
| Model size | 1.3 MB | TorchScript serialized |

## Validation Results

All models passed validation with **exact numerical equivalence**:

- **Tests per model**: 2 random inputs
- **Max absolute difference**: 0.00e+00 (exact match)
- **Max relative difference**: 0.00e+00 (exact match)
- **Tolerance used**: rtol=1e-05, atol=1e-06

## Files Created

### 1. Modified Files
- `python/MultiClassModels.py`: Added type annotations for TorchScript compatibility

### 2. New Scripts
- `python/export_to_torchscript.py`: Export script with CLI
- `python/validate_torchscript_export.py`: Validation script
- `scripts/export_all_models.sh`: Batch export shell script

### 3. Documentation
- `docs/torchscript_cpp_integration.md`: Comprehensive C++ integration guide

### 4. Exported Models
All located in: `results_bjets/{channel}/multiclass/{signal}/fold-3/models/*_scripted.pt`

## Usage Examples

### Export Models
```bash
# Export all models
python python/export_to_torchscript.py --all --fold 3

# Export single model
python python/export_to_torchscript.py --channel Run1E2Mu --signal MHc100_MA95 --fold 3
```

### Validate Models
```bash
# Validate all models
python python/validate_torchscript_export.py --all --fold 3

# Validate single model
python python/validate_torchscript_export.py --model results_bjets/Run1E2Mu/.../ParticleNet-*_scripted.pt
```

### C++ Integration
See `docs/torchscript_cpp_integration.md` for complete C++ examples including:
- Model loading
- Input construction (9 node features: E, Px, Py, Pz, Charge, IsMuon, IsElectron, IsJet, IsBjet)
- k-NN graph construction
- Inference execution
- Output interpretation

## Key Implementation Details

1. **Export Method**: `torch.jit.trace()` (not script) for PyTorch Geometric compatibility
2. **Edge Construction**: Pre-computed in C++ (k=4 nearest neighbors in 9D feature space)
3. **Input Format**:
   - Node features: (N_particles, 9) - includes leptons, jets, bjets
   - Edge index: (2, N_edges) - k-NN graph
   - Graph features: (1, 4) - era one-hot encoding
   - Batch: (N_particles,) - all zeros for single event
4. **Node Features Order**: E, Px, Py, Pz, Charge, IsMuon, IsElectron, IsJet, IsBjet

## Next Steps for C++ Integration

1. Implement k-NN graph construction in C++ (see docs for examples)
2. Load TorchScript models using libtorch
3. Prepare input tensors matching the specification
4. Run inference and extract class probabilities
5. Apply softmax for final probability outputs

## Validation Report

```
Total models validated: 6
Passed: 6
Failed: 0

All models produce numerically identical outputs to original PyTorch models.
```

---

**Authors**: Claude Code + Human
**Version**: 1.0
**Last Updated**: 2025-10-08

# SignalRegionStudy Build Notes

## Quick Start
```bash
source setup.sh
./scripts/build.sh
```

## Build Configuration

### Environment Requirements
- **CMSSW**: 14.1.0 pre4 (sourced via setup.sh)
- **ROOT**: 6.30.07 with RooFit support
- **CMake**: 3.20+
- **C++ Standard**: C++17 (matches ROOT configuration)

### CMake Configuration
The build script automatically configures:
- `ROOT_DIR=$ROOTSYS/cmake` - Enables ROOT package detection
- RooFit and RooFitCore components included
- Dictionary generation: `G__SignalRegionStudy`
- Install prefix: `$WORKDIR/SignalRegionStudy`

### Build Outputs
Located in `lib/`:
- `libSignalRegionStudy.so` - Main shared library
- `libSignalRegionStudy.rootmap` - ROOT class mapping
- `libSignalRegionStudy_rdict.pcm` - ROOT dictionary

### Common Issues

#### Issue: "Could not find a package configuration file provided by ROOT"
**Solution**: Ensure `source setup.sh` was run before building. The setup script loads CMSSW environment which provides ROOT.

#### Issue: Dictionary naming mismatch errors
**Solution**: All dictionary references should use `SignalRegionStudy` (not V1). This is now corrected in CMakeLists.txt.

#### Issue: C++ standard mismatch warnings
**Solution**: Using C++17 (not C++20) to match ROOT configuration. This is expected and non-critical.

## Library Usage
After building, the library can be used from Python/ROOT:
```python
import ROOT
ROOT.gSystem.Load("lib/libSignalRegionStudy.so")

# Use Preprocessor
prep = ROOT.Preprocessor("2022", "Skim1E2Mu", "SingleMuon")
# Use AmassFitter
fitter = ROOT.AmassFitter("input.root", "output.root")
```

## Rebuild
To force a clean rebuild:
```bash
rm -rf build lib
./scripts/build.sh
```

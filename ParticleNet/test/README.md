# ParticleNet Tests

This directory contains test programs for the ParticleNet C++ library.

## Test Programs

### test_dataformat
Tests the DataFormat classes with ROOT's RVec functionality.

**What it tests:**
- Particle class creation (Muon, Electron, Jet)
- RVec collections
- Feature extraction with `particlesToFeatures()`
- Basic particle properties (pt, charge, b-tagging)

**Build and run:**
```bash
# From ParticleNet directory
make test
./test/test_dataformat
```

## Running All Tests

Use the build_and_test.sh script:
```bash
bash scripts/build_and_test.sh
```

This will:
1. Source the ROOT environment
2. Build the C++ library
3. Build all test programs
4. Run the tests

## Expected Output

```
Created particles:
Muon: pt=50, charge=-1
Electron: pt=30, charge=1
Jet: pt=100, b-tagged=1

Feature matrix size: 4 particles
Each particle has 9 features
```

## Adding New Tests

1. Create a new `.cpp` file in this directory
2. Update the Makefile to add the new test target
3. Document the test in this README

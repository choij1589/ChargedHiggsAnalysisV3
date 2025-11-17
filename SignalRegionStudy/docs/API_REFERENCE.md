# SignalRegionStudy - API Reference

**Version**: 1.0
**Last Updated**: 2025-10-10
**Language**: C++17 with ROOT/RooFit

---

## Table of Contents

- [Preprocessor Class](#preprocessor-class)
- [AmassFitter Class](#amassfitter-class)
- [Type Definitions](#type-definitions)
- [Usage Examples](#usage-examples)
- [Shell Script API](#shell-script-api)

---

## Preprocessor Class

**Header**: `include/Preprocessor.h` | **Implementation**: `src/Preprocessor.cc`

**Purpose**: Data preprocessing and event selection for signal region template creation

### Class Declaration

```cpp
class Preprocessor {
public:
    Preprocessor(const TString &era, const TString &channel, const TString &datastream);
    ~Preprocessor() = default;

    // File Management
    void setInputFile(const TString &rtfile_path);
    void setInputTree(const TString &syst);
    void setOutputFile(const TString &outFileName);
    void closeOutputFile();
    void closeInputFile();

    // Processing
    void fillOutTree(const TString &sampleName,
                     const TString &signal,
                     const TString &syst="Central",
                     const bool applyConvSF=false,
                     const bool isTrainedSample=false);
    void saveTree();

    // Configuration
    void setConvSF(const double &convSF, const double &convSFerr);
    void setEra(const TString &era);
    void setChannel(const TString &channel);
    void setDatastream(const TString &datastream);

    // Getters
    double getConvSF(int sys=0) const;
    TString getEra() const;
    TString getChannel() const;
    TString getDatastream() const;
    TTree* getOutTree() const;  // For debugging
    TTree* getInTree() const;   // For debugging

private:
    TString era;
    TString channel;
    TString datastream;
    TFile *inFile;
    TFile *outFile;
    TTree *centralTree;
    TTree *inTree;
    TTree *outTree;

    double convSF, convSFerr;
    double mass, mass1, mass2, scoreX, scoreY, scoreZ, weight;
    int fold;
};
```

### Constructor

```cpp
Preprocessor::Preprocessor(const TString &era,
                           const TString &channel,
                           const TString &datastream)
```

**Parameters**:
- `era` (TString): Run period identifier
  - **Valid values**: `"2022"`, `"2022EE"`, `"2023"`, `"2023BPix"`, `"2016preVFP"`, `"2016postVFP"`, `"2017"`, `"2018"`
  - **Purpose**: Controls era-specific configurations (luminosity, data streams)

- `channel` (TString): Analysis channel
  - **Valid values**: `"Skim1E2Mu"`, `"Skim3Mu"`, `"SR1E2Mu"`, `"SR3Mu"`
  - **Purpose**: Determines mass extraction logic (single vs double entry per event)

- `datastream` (TString): Data stream identifier
  - **Examples**: `"SingleMuon"`, `"DoubleMuon"`, `"EGamma"`
  - **Purpose**: Tracks data source for validation

**Example**:
```cpp
Preprocessor prep("2022", "Skim1E2Mu", "SingleMuon");
```

---

### File Management Methods

#### setInputFile
```cpp
void setInputFile(const TString &rtfile_path)
```
**Purpose**: Open ROOT file containing analysis trees

**Parameters**:
- `rtfile_path` (TString): Path to input ROOT file with `Events_Central` and systematic trees

**Preconditions**: None
**Postconditions**: `inFile` points to opened ROOT file

**Example**:
```cpp
prep.setInputFile("data/2022/Skim1E2Mu/DYJets.root");
```

---

#### setInputTree
```cpp
void setInputTree(const TString &syst)
```
**Purpose**: Load central and systematic variation trees

**Parameters**:
- `syst` (TString): Systematic variation name
  - **"Central"**: Nominal values
  - **Other examples**: `"L1PrefireUp"`, `"PileupReweightDown"`, `"MuonIDSFUp"`

**Behavior**:
1. Loads `Events_Central` tree (always)
2. Loads `Events_{syst}` tree for processing

**Branch Requirements**:
- **Required**: `mass1`, `mass2`, `weight`, `fold`
- **Optional** (when `isTrainedSample=true`): `score_{signal}_vs_nonprompt`, `score_{signal}_vs_diboson`, `score_{signal}_vs_ttZ`

**Example**:
```cpp
prep.setInputTree("Central");        // Nominal
prep.setInputTree("MuonIDSFUp");     // Systematic variation
```

---

#### setOutputFile
```cpp
void setOutputFile(const TString &outFileName)
```
**Purpose**: Create output ROOT file for processed trees

**Parameters**:
- `outFileName` (TString): Output file path (will be created/overwritten)

**Postconditions**: `outFile` ready for writing trees

**Example**:
```cpp
prep.setOutputFile("samples/2022/SR1E2Mu/DYJets_processed.root");
```

---

#### saveTree
```cpp
void saveTree()
```
**Purpose**: Write output tree to file

**Preconditions**: `fillOutTree()` has been called
**Behavior**: Writes `outTree` to `outFile` and persists to disk

**Example**:
```cpp
prep.fillOutTree("DYJets", "MHc130_MA90", "Central");
prep.saveTree();
```

---

#### closeOutputFile / closeInputFile
```cpp
void closeOutputFile()
void closeInputFile()
```
**Purpose**: Properly close ROOT files and release resources

**Typical Usage Pattern**:
```cpp
prep.setInputFile("input.root");
prep.setOutputFile("output.root");
prep.setInputTree("Central");
prep.fillOutTree("DYJets", "MHc130_MA90");
prep.saveTree();
prep.closeOutputFile();
prep.closeInputFile();
```

---

### Processing Methods

#### fillOutTree (Core Method)
```cpp
void fillOutTree(const TString &sampleName,
                 const TString &signal,
                 const TString &syst="Central",
                 const bool applyConvSF=false,
                 const bool isTrainedSample=false)
```
**Purpose**: Process input tree and create signal region template

**Parameters**:
- `sampleName` (TString): Sample identifier
  - **Signal samples**: Contain `"MA"` in name → applies cross-section normalization
  - **Background samples**: Any other name

- `signal` (TString): Signal process name for score branch names
  - **Format**: `"MHc{Hc_mass}_MA{A_mass}"` (e.g., `"MHc130_MA90"`)
  - **Used for**: Score branch naming (`score_MHc130_MA90_vs_nonprompt`)

- `syst` (TString, default="Central"): Systematic variation
  - **Tree name**: Created tree named `{syst}` in output file

- `applyConvSF` (bool, default=false): Apply conversion electron scale factor
  - **When true**: Multiplies weight by `convSF ± convSFerr`
  - **When false**: No additional weighting

- `isTrainedSample` (bool, default=false): Include ML discriminator scores
  - **When true**: Reads and writes `scoreX`, `scoreY`, `scoreZ` branches
  - **When false**: Score branches omitted

**Processing Logic**:

1. **Weight Adjustment**:
   ```cpp
   if (sampleName.Contains("MA")) {
       weight /= 3.0;  // Signal: normalize to 5 fb
   } else if (applyConvSF) {
       weight *= getConvSF();  // Background: apply conversion SF
   }
   ```

2. **Channel-Specific Mass Extraction**:
   ```cpp
   // 1E2Mu: Single Z candidate per event
   if (channel.Contains("1E2Mu")) {
       mass = mass1;
       outTree->Fill();
   }

   // 3Mu: Two Z candidates per event - select based on mass point
   else if (channel.Contains("3Mu")) {
       // Extract mass point parameters from signal string
       int mHc = extractMHc(signal);
       int mA = extractMA(signal);

       // Selection algorithm based on mass point thresholds:
       // - If MHc >= 100 AND MA >= 60: mass = max(mass1, mass2)
       // - Otherwise: mass = min(mass1, mass2)
       // For background samples (no MHc/MA): defaults to min
       if (mHc >= 100 && mA >= 60) {
           mass = (mass1 > mass2) ? mass1 : mass2;  // max
       } else {
           mass = (mass1 < mass2) ? mass1 : mass2;  // min
       }
       outTree->Fill();
   }
   ```

**Output Branches**:
- `mass` (double): Invariant mass for template binning
- `mass1` (double): First lepton pair mass
- `mass2` (double): Second lepton pair mass
- `weight` (double): Event weight (includes scale factors)
- `fold` (int): K-fold validation index (0-4)
- `scoreX` (double): ML score vs nonprompt (if `isTrainedSample`)
- `scoreY` (double): ML score vs diboson (if `isTrainedSample`)
- `scoreZ` (double): ML score vs ttZ (if `isTrainedSample`)

**Example**:
```cpp
// Signal sample with ML scores
prep.fillOutTree("MHc130_MA90", "MHc130_MA90", "Central", false, true);

// Background sample with conversion SF
prep.fillOutTree("DYJets", "MHc130_MA90", "Central", true, false);

// Systematic variation
prep.fillOutTree("ttZ", "MHc130_MA90", "MuonIDSFUp", false, true);
```

---

### Configuration Methods

#### setConvSF
```cpp
void setConvSF(const double &convSF, const double &convSFerr)
```
**Purpose**: Configure conversion electron scale factor

**Parameters**:
- `convSF` (double): Nominal scale factor value
- `convSFerr` (double): Uncertainty on scale factor

**Typical Values**:
- **1E2Mu channel**: ~1.05 ± 0.10 (5% correction, 10% uncertainty)
- **3Mu channel**: Not applied (no conversion electrons)

**Example**:
```cpp
prep.setConvSF(1.05, 0.10);
prep.fillOutTree("DYJets", "MHc130_MA90", "Central", true, false);
```

---

#### setEra / setChannel / setDatastream
```cpp
void setEra(const TString &era)
void setChannel(const TString &channel)
void setDatastream(const TString &datastream)
```
**Purpose**: Update configuration after construction (rare usage)

**Use Case**: Reusing same Preprocessor instance for multiple configurations

**Example**:
```cpp
Preprocessor prep("2022", "Skim1E2Mu", "SingleMuon");
// ... process 2022 data ...
prep.setEra("2023");
prep.setDatastream("Muon");
// ... process 2023 data ...
```

---

### Getter Methods

#### getConvSF
```cpp
double getConvSF(int sys=0) const
```
**Purpose**: Retrieve conversion scale factor with systematic variation

**Parameters**:
- `sys` (int, default=0): Variation index
  - **-1**: Down variation (convSF - convSFerr)
  - **0**: Nominal (convSF)
  - **+1**: Up variation (convSF + convSFerr)

**Returns**: Scale factor value

**Example**:
```cpp
prep.setConvSF(1.05, 0.10);
double nominal = prep.getConvSF(0);   // 1.05
double down = prep.getConvSF(-1);     // 0.95
double up = prep.getConvSF(1);        // 1.15
```

---

#### getEra / getChannel / getDatastream
```cpp
TString getEra() const
TString getChannel() const
TString getDatastream() const
```
**Purpose**: Retrieve current configuration

**Returns**: Configuration string values

**Example**:
```cpp
Preprocessor prep("2022", "Skim1E2Mu", "SingleMuon");
TString era = prep.getEra();  // "2022"
```

---

#### getOutTree / getInTree
```cpp
TTree* getOutTree() const
TTree* getInTree() const
```
**Purpose**: Debugging access to internal trees

**Returns**: Pointer to tree (may be nullptr before initialization)

**Use Case**: Validate branch structure, check entry counts

**Example**:
```cpp
prep.fillOutTree("DYJets", "MHc130_MA90");
TTree *tree = prep.getOutTree();
cout << "Processed " << tree->GetEntries() << " entries" << endl;
```

---

## AmassFitter Class

**Header**: `include/AmassFitter.h` | **Implementation**: `src/AmassFitter.cc`

**Purpose**: Voigtian fitting for pseudoscalar mass (mA) peak extraction

### Class Declaration

```cpp
class AmassFitter {
public:
    AmassFitter(const TString &input_path, const TString &output_path);
    ~AmassFitter() = default;

    // Fitting
    void fitMass(const double &mA, const double &low, const double &high);

    // Output
    void saveCanvas(const TString &canvas_path);
    void Close();

    // Getters
    RooRealVar* getRooMass();
    RooRealVar* getRooWeight();
    RooRealVar* getRooMA();
    RooRealVar* getRooSigma();
    RooRealVar* getRooWidth();
    RooVoigtian* getRooModel();
    RooFitResult* getFitResult();

private:
    TFile *input_file;
    TFile *output_file;
    RooDataSet *roo_data;
    RooRealVar *roo_mass;
    RooRealVar *roo_weight;
    RooRealVar *roo_mA;
    RooRealVar *roo_sigma;
    RooRealVar *roo_width;
    RooVoigtian *roo_model;
    RooFitResult *fit_result;
    TCanvas *canvas;
};
```

### Constructor

```cpp
AmassFitter::AmassFitter(const TString &input_path, const TString &output_path)
```

**Parameters**:
- `input_path` (TString): ROOT file with `Events_Central` tree
  - **Required branches**: `mass1`, `mass2`, `weight`

- `output_path` (TString): Output ROOT file for fit results
  - **Contents**: `RooDataSet`, `RooVoigtian`, `RooFitResult`

**Example**:
```cpp
AmassFitter fitter("samples/2022/SR3Mu/MHc130_MA90.root",
                   "fit_results/mA90_fit.root");
```

---

### Fitting Methods

#### fitMass
```cpp
void fitMass(const double &mA, const double &low, const double &high)
```
**Purpose**: Perform Voigtian fit to mA peak

**Parameters**:
- `mA` (double): Nominal mA value (GeV)
  - **Initial value**: Used for Z candidate selection
  - **Fit range**: Constrained within [low, high]

- `low` (double): Fit range lower bound (GeV)
  - **Typical**: mA - 10 GeV

- `high` (double): Fit range upper bound (GeV)
  - **Typical**: mA + 10 GeV

**Algorithm**:

1. **Z Candidate Selection**:
   ```cpp
   // For each event, select Z candidate closer to nominal mA
   if (pair1_mass < 0. && pair2_mass < 0.) continue;
   Amass = fabs(pair1_mass - mA) < fabs(pair2_mass - mA)
           ? pair1_mass : pair2_mass;
   ```

2. **RooFit Model**:
   ```cpp
   // Voigtian PDF: convolution of Breit-Wigner and Gaussian
   roo_mA = new RooRealVar("mA", "mA", mA, low, high);
   roo_sigma = new RooRealVar("sigma", "sigma", 0.1, 0., 3.);
   roo_width = new RooRealVar("width", "width", 0.1, 0., 3.);
   roo_model = new RooVoigtian("model", "model",
                                *roo_mass, *roo_mA, *roo_width, *roo_sigma);
   ```

3. **Fit Execution**:
   ```cpp
   fit_result = roo_model->fitTo(*roo_data,
                                 SumW2Error(kTRUE),  // Weighted errors
                                 Save());             // Return result
   ```

**Fitted Parameters**:
- **mA**: Pseudoscalar mass peak position
- **sigma**: Gaussian resolution (detector smearing)
- **width**: Breit-Wigner natural width (decay width)

**Example**:
```cpp
fitter.fitMass(90.0, 80.0, 100.0);  // Fit mA=90 GeV in [80, 100] range

// Access results
double fitted_mA = fitter.getRooMA()->getVal();
double mA_error = fitter.getRooMA()->getError();
double chi2_ndf = fitter.getFitResult()->chiSquare();
```

---

### Output Methods

#### saveCanvas
```cpp
void saveCanvas(const TString &canvas_path)
```
**Purpose**: Generate diagnostic plot of fit

**Parameters**:
- `canvas_path` (TString): Output file path
  - **Supported formats**: `.pdf`, `.png`, `.eps`, `.root`

**Plot Contents**:
- Data histogram with error bars
- Fitted Voigtian model (red curve)
- Automatic legend and axis labels

**Example**:
```cpp
fitter.fitMass(90.0, 80.0, 100.0);
fitter.saveCanvas("plots/mA90_fit.pdf");
```

---

#### Close
```cpp
void Close()
```
**Purpose**: Write fit results to output file and close files

**Behavior**:
1. Switch to output file directory
2. Write `RooDataSet` (data)
3. Write `RooVoigtian` (model)
4. Write `RooFitResult` (fit parameters, covariance)
5. Close both input and output files

**Typical Workflow**:
```cpp
AmassFitter fitter("input.root", "output.root");
fitter.fitMass(90.0, 80.0, 100.0);
fitter.saveCanvas("plots/fit.pdf");
fitter.Close();  // Persists results
```

---

### Getter Methods

#### getRooMA / getRooSigma / getRooWidth
```cpp
RooRealVar* getRooMA()
RooRealVar* getRooSigma()
RooRealVar* getRooWidth()
```
**Purpose**: Access fitted parameters

**Returns**: Pointer to RooRealVar with value and error

**Methods**:
- `getVal()`: Central value
- `getError()`: Uncertainty
- `getBinning()`: Parameter range

**Example**:
```cpp
fitter.fitMass(90.0, 80.0, 100.0);

RooRealVar *mA_var = fitter.getRooMA();
double mA = mA_var->getVal();
double mA_err = mA_var->getError();
cout << "mA = " << mA << " ± " << mA_err << " GeV" << endl;
```

---

#### getFitResult
```cpp
RooFitResult* getFitResult()
```
**Purpose**: Access full fit result with covariance matrix

**Returns**: Pointer to RooFitResult

**Available Methods**:
- `status()`: Fit status (0 = success)
- `covQual()`: Covariance matrix quality (3 = full accurate)
- `edm()`: Estimated distance to minimum
- `minNll()`: Negative log-likelihood at minimum
- `correlationMatrix()`: Parameter correlations

**Example**:
```cpp
RooFitResult *result = fitter.getFitResult();
cout << "Fit status: " << result->status() << endl;
cout << "Covariance quality: " << result->covQual() << endl;

// Check correlations
const TMatrixDSym &corr = result->correlationMatrix();
double corr_sigma_width = corr(1, 2);  // Sigma-Width correlation
```

---

#### getRooModel
```cpp
RooVoigtian* getRooModel()
```
**Purpose**: Access fitted PDF for further calculations

**Returns**: Pointer to RooVoigtian model

**Use Cases**:
- Integrate PDF over ranges
- Calculate normalized probabilities
- Generate toy MC samples

**Example**:
```cpp
RooVoigtian *model = fitter.getRooModel();
RooRealVar *mass = fitter.getRooMass();

// Integrate over ±3σ window
mass->setRange("signal", mA - 3*sigma, mA + 3*sigma);
double integral = model->createIntegral(*mass, "signal")->getVal();
```

---

## Type Definitions

### TString (ROOT String Type)
**Include**: `#include <TString.h>`

**Common Operations**:
```cpp
TString str = "value";
str.Contains("val");      // Check substring
str.ReplaceAll("a", "b"); // Replace
str.Append("_suffix");    // Concatenate
TString formatted = TString::Format("mA_%d", 90);  // Printf-style
```

### RooRealVar (RooFit Variable)
**Include**: `#include <RooRealVar.h>`

**Common Methods**:
```cpp
RooRealVar var("name", "title", init_val, min, max);
var.setVal(100.0);         // Set value
var.setError(5.0);         // Set uncertainty
double val = var.getVal(); // Get value
double err = var.getError(); // Get uncertainty
```

### RooDataSet (RooFit Dataset)
**Include**: `#include <RooDataSet.h>`

**Creation**:
```cpp
RooDataSet data("data", "title",
                RooArgSet(mass, weight),
                WeightVar(weight),
                Import(*tree));
```

---

## Python Scripts API

### makeBinnedTemplates.py

**Purpose**: Generate binned histogram templates for HiggsCombine statistical analysis

**Location**: `python/makeBinnedTemplates.py`

**Dependencies**:
- Python 3.9+
- ROOT with RooFit
- SignalRegionStudy C++ library (`libSignalRegionStudy.so`)
- JSON configuration files (`configs/systematics.json`)

---

#### Command-Line Interface

```bash
python3 python/makeBinnedTemplates.py \
  --era ERA             # 2016preVFP, 2016postVFP, 2017, 2018, 2022, 2022EE, 2023, 2023BPix
  --channel CHANNEL     # SR1E2Mu, SR3Mu
  --masspoint MASSPOINT # MHc130_MA90, MHc100_MA60, etc.
  --method METHOD       # Baseline (ParticleNet/GBDT: future)
  [--debug]             # Enable debug logging
```

**Example**:
```bash
python3 python/makeBinnedTemplates.py \
  --era 2017 \
  --channel SR1E2Mu \
  --masspoint MHc130_MA90 \
  --method Baseline
```

---

#### Core Functions

##### getFitResult

```python
def getFitResult(input_path, output_path, mA_nominal):
    """
    Fit A mass distribution using AmassFitter C++ class.

    Args:
        input_path (str): Path to signal ROOT file with Central tree
        output_path (str): Path to save fit results
        mA_nominal (float): Nominal A mass value (GeV)

    Returns:
        tuple: (mA_fitted, width, sigma) in GeV
            - mA_fitted: Fitted peak position
            - width: Breit-Wigner decay width (Γ)
            - sigma: Gaussian detector resolution (σ)

    Side Effects:
        - Creates fit_result.png in output directory
        - Writes RooFit workspace to output_path

    Raises:
        FileNotFoundError: If input file doesn't exist
        RuntimeError: If fit fails or tree not found
    """
```

**Usage**:
```python
signal_path = f"{BASEDIR}/{args.masspoint}.root"
mA, width, sigma = getFitResult(signal_path, f"{OUTDIR}/fit_result.root", 90.0)
# mA = 89.84, width = 0.893, sigma = 0.823 (typical values)
```

---

##### ensurePositiveIntegral

```python
def ensurePositiveIntegral(hist, min_integral=1e-10):
    """
    Ensure histogram has positive integral for normalization in HiggsCombine.

    IMPORTANT: This function addresses negative/zero histogram INTEGRALS, not individual
    negative bins. Negative bins are physically meaningful (interference effects, negative
    MC weights) and are acceptable to Combine. However, negative integrals cause "Bogus norm"
    errors because they make the Poisson likelihood undefined.

    Args:
        hist (ROOT.TH1D): Histogram to check and fix
        min_integral (float, optional): Minimum positive integral value (default: 1e-10)
            - Small enough to be negligible compared to other processes
            - Large enough to avoid numerical issues in Combine

    Returns:
        bool: True if histogram was modified, False if integral was already positive

    Behavior:
        1. Calculate total integral: ∫hist = Σ(bin contents)
        2. If integral > 0: No action needed, return False
        3. If integral ≤ 0:
           - Set central bin content to min_integral
           - Set central bin error to min_integral
           - Log warning with histogram name and original integral
           - Return True

    Algorithm:
        central_bin = hist.GetNbinsX() // 2 + 1  # Middle bin
        hist.SetBinContent(central_bin, min_integral)
        hist.SetBinError(central_bin, min_integral)

    Rationale:
        - Setting only central bin minimizes shape distortion
        - Alternative approaches (scaling all bins) fail for all-negative histograms
        - min_integral = 1e-10 events is negligible in statistical analysis

    Logging:
        WARNING: Histogram {name} has non-positive integral: {integral:.4e}
        WARNING: Setting bin {central_bin} to {min_integral} to ensure positive normalization

    Raises:
        None (handles all cases gracefully)
    """
```

**Usage**:
```python
# In getHist() function after creating histogram
hist = df.Histo1D(model, "mass", "weight").GetValue()
hist.SetDirectory(0)

# Ensure positive integral for Combine
was_modified = ensurePositiveIntegral(hist)
if was_modified:
    logging.warning(f"Fixed negative integral in {hist.GetName()}")

return hist
```

**Example Output** (when fix applied):
```
WARNING: Histogram conversion_PileupReweightDown has non-positive integral: -4.4586e-03
WARNING: Setting bin 8 to 1e-10 to ensure positive normalization
```

**Technical Context**:
- **text2workspace.py error**: "Bogus norm -0.004458621144294739 for channel signal_region, process conversion, systematic PileupReweight Down"
- **Why negative integrals occur**: Statistical fluctuations in low-yield processes (especially conversion background with systematic variations)
- **Why this fix works**: Combine requires positive normalization for Poisson likelihood; minimal positive value satisfies requirement without affecting fit
- **Related**: See [NEGATIVE_BINS_HANDLING.md](../docs/NEGATIVE_BINS_HANDLING.md) for comprehensive explanation

---

##### getHist

```python
def getHist(process, mA, width, sigma, syst="Central"):
    """
    Create histogram from preprocessed tree using RDataFrame.
    Always uses 'mass' branch (set in preprocessing for both channels).

    Args:
        process (str): Process name
            - Signal: "MHc130_MA90", "MHc100_MA60", etc.
            - Background: "nonprompt", "conversion", "diboson", "ttX", "others"
        mA (float): Fitted A mass in GeV
        width (float): Breit-Wigner width Γ in GeV
        sigma (float): Gaussian resolution σ in GeV
        syst (str, optional): Systematic variation tree name
            - "Central": Nominal values
            - "{SystName}_Up": Upward variation
            - "{SystName}_Down": Downward variation

    Returns:
        ROOT.TH1D: Histogram detached from file (SetDirectory(0))

    Binning:
        - Fixed 15 bins
        - Range: [mA - 5√(Γ² + σ²), mA + 5√(Γ² + σ²)]
        - Example: mA=90, Γ=0.89, σ=0.82 → [83.76, 95.91] GeV

    Branch Usage:
        - Uses "mass" branch for both SR1E2Mu and SR3Mu
        - Uses "weight" branch for event weighting

    Raises:
        FileNotFoundError: If sample file not found
        RuntimeError: If tree doesn't exist in file
    """
```

**Usage**:
```python
# Signal central
hist_signal = getHist("MHc130_MA90", 89.84, 0.893, 0.823, "Central")

# Signal systematic
hist_muon_up = getHist("MHc130_MA90", 89.84, 0.893, 0.823, "MuonIDSF_Up")

# Background
hist_nonprompt = getHist("nonprompt", 89.84, 0.893, 0.823, "Central")
```

---

#### Processing Flow

**Main Execution**:

1. **Setup & Configuration**
   ```python
   # Load C++ library
   ROOT.gSystem.Load(f"{WORKDIR}/SignalRegionStudy/lib/libSignalRegionStudy.so")

   # Load systematics configuration
   json_systematics = json.load(open(f"{WORKDIR}/SignalRegionStudy/configs/systematics.json"))
   RUN = "Run2" if args.era in ["2016preVFP", "2017", "2018"] else "Run3"
   prompt_systematics = json_systematics[RUN][args.channel]["prompt"]
   ```

2. **Mass Fitting**
   ```python
   # Extract nominal mA from masspoint name
   mA_nominal = float(args.masspoint.split("_")[1].replace("MA", ""))  # 90.0

   # Fit signal mass distribution
   mA, width, sigma = getFitResult(signal_path, fit_output, mA_nominal)

   # Calculate binning range
   mass_range = 5 * sqrt(width**2 + sigma**2)
   ```

3. **Signal Processing**
   ```python
   # Central histogram
   hist_central = getHist(args.masspoint, mA, width, sigma, "Central")
   output_file.cd()
   hist_central.Write()

   # Systematic variations
   for syst_name, variations in prompt_systematics.items():
       for var in variations:  # ["MuonIDSF_Up", "MuonIDSF_Down"]
           hist = getHist(args.masspoint, mA, width, sigma, var)
           output_file.cd()
           hist.Write()
   ```

4. **Background Processing**
   ```python
   # Nonprompt (data-driven)
   hist_nonprompt = getHist("nonprompt", mA, width, sigma, "Central")
   data_obs.Add(hist_nonprompt)  # Add to data_obs
   hist_nonprompt.Write()

   # Nonprompt-specific systematics
   hist_nonprompt_up = getHist("nonprompt", mA, width, sigma, "Nonprompt_Up")
   hist_nonprompt_down = getHist("nonprompt", mA, width, sigma, "Nonprompt_Down")

   # Prompt MC backgrounds (conversion, diboson, ttX, others)
   for process in ["conversion", "diboson", "ttX", "others"]:
       hist = getHist(process, mA, width, sigma, "Central")
       data_obs.Add(hist)
       hist.Write()

       # Prompt systematics for each
       for syst_name, variations in prompt_systematics.items():
           for var in variations:
               hist = getHist(process, mA, width, sigma, var)
               hist.Write()
   ```

5. **Data Observable**
   ```python
   # data_obs = sum of all backgrounds
   data_obs.Write()
   ```

---

#### Configuration Files

**systematics.json**:
```json
{
  "Run2": {
    "SR1E2Mu": {
      "prompt": {
        "L1Prefire": ["L1Prefire_Up", "L1Prefire_Down"],
        "MuonIDSF": ["MuonIDSF_Up", "MuonIDSF_Down"],
        "BtagSF_HFcorr": ["BtagSF_HFcorr_Up", "BtagSF_HFcorr_Down"],
        ...
      }
    }
  },
  "Run3": { ... }
}
```

---

#### Output Structure

**Directory Layout**:
```
templates/{era}/{channel}/{masspoint}/Shape/{method}/
├── shapes.root           # All templates
├── fit_result.root       # RooFit workspace
└── fit_result.png        # Fit visualization
```

**shapes.root Contents** (~159 histograms):
- Signal: 1 central + 30 systematics = 31
- Nonprompt: 1 central + 2 variations = 3
- Conversion: 1 central + 30 systematics = 31
- Diboson: 1 central + 30 systematics = 31
- ttX: 1 central + 30 systematics = 31
- Others: 1 central + 30 systematics = 31
- data_obs: 1

**Histogram Naming**:
- Central: `{process}`
- Systematic: `{process}_{systematic}Up` / `{process}_{systematic}Down`
- Examples: `MHc130_MA90`, `nonprompt_Nonprompt_Up`, `diboson_MuonIDSF_Down`

---

#### Validation & Error Handling

**Histogram Validation**:
```python
# Check non-zero integral
if hist.Integral() <= 0:
    logging.warning(f"Empty histogram: {hist.GetName()}")

# Check systematic variation magnitude
ratio = hist_up.Integral() / hist_central.Integral()
if ratio < 0.5 or ratio > 2.0:
    logging.warning(f"Large systematic variation: {ratio:.2f}")
```

**Error Conditions**:
- Missing input files → `FileNotFoundError` with clear message
- Missing trees → `RuntimeError` with tree name
- Empty histograms → Warning logged, processing continues
- Type mismatches → Logged warnings from ROOT

---

#### Performance

**Typical Metrics**:
- **Runtime**: 2-5 minutes per masspoint
- **Memory**: ~2 GB peak usage
- **Scaling**: Linear with number of systematics
- **Parallelization**: Can process multiple masspoints in parallel

**Optimization Tips**:
```bash
# Process multiple masspoints in parallel
parallel -j 4 makeBinnedTemplates.py --era 2017 --channel SR1E2Mu \
  --masspoint {} --method Baseline ::: MHc130_MA90 MHc100_MA60 MHc160_MA120
```

---

#### Known Limitations

1. **Theory Systematics**: PDF, Scale, PS variations not yet implemented
   - Require theory systematics in preprocessed files
   - Will be added in future release

2. **ConvSF Treatment**: Conversion scale factor applied as rate uncertainty
   - Not included as shape systematic in templates
   - Specified in datacard generation step

3. **Methods**: Only Baseline implemented
   - ParticleNet method deferred (requires ML score optimization)
   - GBDT method deferred (requires model training)

4. **Channel Support**: SR1E2Mu and SR3Mu only
   - Uses unified "mass" branch approach
   - Additional channels require preprocessing updates

---

#### Example Integration

**Batch Processing Script**:
```python
#!/usr/bin/env python3
import subprocess
import sys

masspoints = ["MHc100_MA60", "MHc130_MA90", "MHc160_MA120"]
era = "2017"
channel = "SR1E2Mu"
method = "Baseline"

for mp in masspoints:
    print(f"Processing {mp}...")

    cmd = [
        "python3", "python/makeBinnedTemplates.py",
        "--era", era,
        "--channel", channel,
        "--masspoint", mp,
        "--method", method
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"ERROR processing {mp}:")
        print(result.stderr)
        sys.exit(1)

    print(result.stdout)

print("All masspoints processed successfully!")
```

---

### checkTemplates.py

**Purpose**: Validate histogram templates and generate diagnostic plots for quality assurance

**Location**: `python/checkTemplates.py`

**Dependencies**:
- Python 3.9+
- ROOT with graphics support
- Generated templates (`shapes.root` from makeBinnedTemplates.py)

---

#### Command-Line Interface

```bash
python3 python/checkTemplates.py \
  --era ERA             # 2016preVFP, 2016postVFP, 2017, 2018, 2022, 2022EE, 2023, 2023BPix
  --channel CHANNEL     # SR1E2Mu, SR3Mu
  --masspoint MASSPOINT # MHc130_MA90, MHc100_MA60, etc.
  --method METHOD       # Baseline, ParticleNet, GBDT
  [--debug]             # Enable debug logging
```

**Example**:
```bash
python3 python/checkTemplates.py \
  --era 2017 \
  --channel SR1E2Mu \
  --masspoint MHc130_MA90 \
  --method Baseline
```

---

#### Validation Functions

##### validate_histogram

```python
def validate_histogram(hist, hist_name, min_entries=0):
    """
    Validate basic histogram properties.

    Args:
        hist (ROOT.TH1D): Histogram to validate
        hist_name (str): Name for error reporting
        min_entries (int): Minimum required entries (default: 0)

    Returns:
        bool: True if valid, False otherwise

    Validation Checks:
        1. Histogram exists (not None)
        2. Integral > 0 (non-empty)
        3. No negative bin contents
        4. Minimum entry threshold met

    Side Effects:
        - Appends to global validation_issues list on failure
        - Appends to global warnings list for low statistics
    """
```

**Usage**:
```python
shapes_file = ROOT.TFile.Open("shapes.root")
signal_hist = shapes_file.Get("MHc130_MA90")
is_valid = validate_histogram(signal_hist, "MHc130_MA90", min_entries=10)
```

---

##### check_systematic_variation

```python
def check_systematic_variation(nominal_hist, syst_hist_up, syst_hist_down,
                               process, syst_name):
    """
    Check systematic variation is reasonable.

    Args:
        nominal_hist (ROOT.TH1D): Central value histogram
        syst_hist_up (ROOT.TH1D): Up variation histogram
        syst_hist_down (ROOT.TH1D): Down variation histogram
        process (str): Process name (e.g., "MHc130_MA90", "nonprompt")
        syst_name (str): Systematic name (e.g., "MuonIDSF", "BtagSF")

    Validation Checks:
        1. Variation magnitude: [0.5, 2.0] × nominal
        2. Asymmetry: |ΔUp - ΔDown| / 2 < 0.5

    Warnings Generated:
        - Large variations (>2× or <0.5× nominal)
        - Highly asymmetric variations (>50% asymmetry)
    """
```

**Usage**:
```python
central = shapes_file.Get("MHc130_MA90")
up = shapes_file.Get("MHc130_MA90_MuonIDSF_Up")
down = shapes_file.Get("MHc130_MA90_MuonIDSF_Down")
check_systematic_variation(central, up, down, "MHc130_MA90", "MuonIDSF")
```

---

##### calculate_systematic_error

```python
def calculate_systematic_error(shapes_file, bkg_name, ibin):
    """
    Calculate systematic uncertainty for a specific background and bin using envelope method.

    Args:
        shapes_file (ROOT.TFile): File containing histogram templates
        bkg_name (str): Background process name ("nonprompt", "conversion", "diboson", "ttX", "others")
        ibin (int): Histogram bin number (1-indexed)

    Returns:
        float: Total systematic uncertainty (absolute value, not percentage)

    Method:
        - Nonprompt: Uses Nonprompt_Up/Down variations only
        - Prompt backgrounds: Uses all 15 prompt systematics (L1Prefire, PileupReweight, MuonIDSF,
          ElectronIDSF, EMuTrigSF, ElectronRes, ElectronEn, JetRes, JetEn, MuonEn, UnclusteredEn,
          BtagSF_HFcorr, BtagSF_HFuncorr, BtagSF_LFcorr, BtagSF_LFuncorr)
        - Envelope: max(|up - nominal|, |down - nominal|) for each systematic
        - Combination: All systematics combined in quadrature: √(Σ δ²)

    Example:
        nonprompt_syst = calculate_systematic_error(shapes_file, "nonprompt", 5)
        diboson_syst = calculate_systematic_error(shapes_file, "diboson", 5)
        # Typical values: ~0.5-2.0 events per bin depending on statistics
    """
```

---

#### Diagnostic Plot Functions

##### make_background_stack

```python
def make_background_stack(shapes_file):
    """
    Create background stack plot with systematic uncertainties using ComparisonCanvas.

    Args:
        shapes_file (ROOT.TFile): File containing histogram templates

    Output:
        - PNG image: validation/background_stack.png
        - Two-panel plot with CMS publication styling
        - Upper panel: Stacked backgrounds + signal overlay + error bands
        - Lower panel: Data/Prediction ratio with uncertainties

    Features:
        - Systematic Error Calculation:
          * Calls calculate_systematic_error() bin-by-bin for each background
          * Nonprompt: ~30% systematic (from data-driven measurement)
          * Prompt backgrounds: ~10-15% systematic (experimental uncertainties)
          * Combined stat ⊕ syst shown as hatched error band
          * Error band labeled "Stat+Syst" in legend

        - Signal Overlay:
          * Black solid line overlaid on background stack
          * Added to legend as "Signal (X.X events)"
          * Allows direct visual comparison of S vs B

        - Two-Line Text (ROOT TLatex workaround):
          * Line 1: Channel name (e.g., "SR1E2Mu")
          * Line 2: Masspoint and method (e.g., "MHc130_MA90 (Baseline)")
          * Uses manual CMS.drawText() calls since \n not supported

        - Stack Order: others → ttX → diboson → conversion → nonprompt (bottom to top)
        - Data: Black markers with Poisson errors

    Logged Output:
        Background yield: XXX.XX ± Y.YY (stat) ± Z.ZZ (syst)
    """
```

**Publication-Ready Plot**: Suitable for analysis notes and papers with complete uncertainty representation

---

##### make_signal_vs_background

```python
def make_signal_vs_background(shapes_file):
    """
    Create signal vs background comparison using KinematicCanvas.

    Args:
        shapes_file (ROOT.TFile): File containing histogram templates

    Output:
        - PNG image: validation/signal_vs_background.png
        - Single-panel plot with line histograms
        - Signal: Distinctive color line | Background (data_obs): Contrasting color
        - Event yields shown in legend: "Signal (X.X events)", "Background (YY.Y events)"

    Features:
        - X-Axis Range: Restricted to histogram binning (mA ± 5√(Γ² + σ²))
        - Two-Line Text:
          * Line 1: Channel name (e.g., "SR1E2Mu")
          * Line 2: Masspoint with S/B ratio (e.g., "MHc130_MA90, S/B=0.082")
        - Automatic S/B calculation for quick sensitivity assessment
        - CMS publication styling with era and luminosity
    """
```

---

##### make_systematic_variations

```python
def make_systematic_variations(shapes_file):
    """
    Create individual systematic uncertainty plots using KinematicCanvas.

    Args:
        shapes_file (ROOT.TFile): File containing histogram templates

    Output:
        - Separate PNG for EACH systematic (typically 15 plots for Run2):
          validation/systematic_{name}.png (L1Prefire, PileupReweight, MuonIDSF, ElectronIDSF,
          EMuTrigSF, ElectronRes, ElectronEn, JetRes, JetEn, MuonEn, UnclusteredEn,
          BtagSF_HFcorr, BtagSF_HFuncorr, BtagSF_LFcorr, BtagSF_LFuncorr)

    Features:
        - Three Histograms Per Plot:
          * Central: Nominal values
          * Up: Upward variation with % change (e.g., "Up (+5.2%)")
          * Down: Downward variation with % change (e.g., "Down (-3.8%)")

        - X-Axis Range: Restricted to histogram binning (mA ± 5√(Γ² + σ²))

        - Two-Line Text:
          * Line 1: Channel and masspoint (e.g., "SR1E2Mu, MHc130_MA90")
          * Line 2: Systematic name (e.g., "MuonIDSF")

        - Plots ALL Systematics: Not limited to first 4, generates complete set
        - Count varies: Run2 typically 15, Run3 varies by configuration
        - Automatic percentage impact calculation for quick assessment
    """
```

---

#### Validation Workflow

**Main Execution**:

1. **Histogram Integrity Check**
   ```python
   # Check data_obs
   data_obs = shapes_file.Get("data_obs")
   validate_histogram(data_obs, "data_obs", min_entries=1)

   # Check signal
   signal_hist = shapes_file.Get(args.masspoint)
   validate_histogram(signal_hist, args.masspoint, min_entries=10)

   # Check backgrounds
   for bkg in ["nonprompt", "conversion", "diboson", "ttX", "others"]:
       hist = shapes_file.Get(bkg)
       validate_histogram(hist, bkg, min_entries=1)
   ```

2. **Systematic Coverage Check**
   ```python
   # Signal systematics
   for syst_name, variations in prompt_systematics.items():
       for var in variations:  # e.g., ["MuonIDSF_Up", "MuonIDSF_Down"]
           hist_name = f"{args.masspoint}_{var}"
           hist = shapes_file.Get(hist_name)
           if not hist:
               validation_issues.append(f"Missing: {hist_name}")

   # Background systematics
   for bkg in ["conversion", "diboson", "ttX", "others"]:
       for syst_name, variations in prompt_systematics.items():
           # Check all variations exist
   ```

3. **Diagnostic Plot Generation**
   ```python
   make_background_stack(shapes_file)
   make_signal_vs_background(shapes_file)
   make_systematic_variations(shapes_file)
   ```

4. **Validation Report**
   ```python
   with open(f"{VALIDATION_DIR}/validation_report.txt", "w") as report:
       report.write(f"Era: {args.era}\n")
       report.write(f"Process yields:\n")
       for process in PROCESSES:
           report.write(f"  {process}: {hist.Integral():.4f}\n")
       report.write(f"\nValidation Issues:\n")
       for issue in validation_issues:
           report.write(f"  ✗ {issue}\n")
       report.write(f"\nWarnings:\n")
       for warning in warnings:
           report.write(f"  ⚠ {warning}\n")
   ```

---

#### Output Structure

**Directory Layout**:
```
templates/{era}/{channel}/{masspoint}/Shape/{method}/validation/
├── background_stack.png              # Stacked background + signal overlay + stat⊕syst errors
├── signal_vs_background.png          # Signal vs background comparison (restricted x-axis)
├── systematic_L1Prefire.png          # L1Prefire variations (Central/Up/Down with %)
├── systematic_PileupReweight.png     # PileupReweight variations
├── systematic_MuonIDSF.png           # MuonIDSF variations
├── systematic_ElectronIDSF.png       # ElectronIDSF variations
├── systematic_EMuTrigSF.png          # EMuTrigSF variations
├── systematic_ElectronRes.png        # ElectronRes variations
├── systematic_ElectronEn.png         # ElectronEn variations
├── systematic_JetRes.png             # JetRes variations
├── systematic_JetEn.png              # JetEn variations
├── systematic_MuonEn.png             # MuonEn variations
├── systematic_UnclusteredEn.png      # UnclusteredEn variations
├── systematic_BtagSF_HFcorr.png      # BtagSF_HFcorr variations
├── systematic_BtagSF_HFuncorr.png    # BtagSF_HFuncorr variations
├── systematic_BtagSF_LFcorr.png      # BtagSF_LFcorr variations
├── systematic_BtagSF_LFuncorr.png    # BtagSF_LFuncorr variations
└── validation_report.txt             # Text summary with yields and uncertainties
```

**Features**:
- **background_stack.png**: Includes signal overlay (black line), stat⊕syst error bands, two-line text
- **signal_vs_background.png**: X-axis restricted to binning, two-line text with S/B ratio
- **systematic_*.png**: All systematics plotted (15 for Run2 2017 SR1E2Mu), two-line text, restricted x-axis

**validation_report.txt**: Contains era/channel/masspoint info, process yields, S/B ratio, validation issues (negative bins, missing systematics), warnings (large variations), histogram count (~159), and output file list. See example in `templates/2017/SR1E2Mu/MHc130_MA90/Shape/Baseline/validation/`.

---

#### Common Validation Issues

**1. Negative Bin Contents** (Weighted MC with negative weights)
- **Impact**: HiggsCombine may reject templates, common in low-stats samples (conversion)
- **Solutions**: Increase MC statistics, merge into "others", or use Barlow-Beeston lite method

**2. Missing Systematics** (Not available in preprocessed files)
- **Solutions**: Check systematics.json, verify preprocessing, ensure theory systematics in SKNano input

**3. Large Variations** (Extreme weight fluctuations, ratio >2× or <0.5×)
- **Investigation**: Compare `up.Integral() / central.Integral()` and inspect bin-by-bin ratios

**4. Empty Histograms** (Tight cuts or low MC statistics)
- **Solutions**: Check input file entries, verify selection cuts, merge low-yield processes

---

#### Performance & Best Practices

**Performance**: Runtime ~30-60s, Memory ~500 MB per validation

**Workflow**:
1. Always run validation between template generation and datacard creation
2. Review diagnostic plots (background stack, signal vs background, systematic variations)
3. Address critical issues first (negative bins, large variations >2×, missing systematics)
4. Archive validation results: `tar -czf validation_$(date +%Y%m%d).tar.gz templates/*/validation/`

**Batch Processing**:
```bash
for MP in MHc100_MA60 MHc130_MA90 MHc160_MA120; do
  checkTemplates.py --era 2017 --channel SR1E2Mu --masspoint $MP --method Baseline
done
```

---

#### Known Limitations

1. **Negative Bins**: Detects but doesn't auto-fix (manual intervention required)
2. **Plots**: One plot per systematic (15 for Run2 2017 SR1E2Mu, varies by era/channel)
3. **Thresholds**: Hardcoded (0.5×-2.0× variations, 50% asymmetry) - adjust in script if needed
4. **Graphics**: Batch mode only (`ROOT.gROOT.SetBatch(True)`), PNG output

---

### printDatacard.py

**Purpose**: Generate HiggsCombine datacards with automatic handling of problematic systematics

**Location**: `python/printDatacard.py`

**Dependencies**:
- Python 3.9+
- ROOT for histogram manipulation
- Generated templates (`shapes.root` from makeBinnedTemplates.py)
- Systematic configuration (`configs/systematics.json`)

---

#### Command-Line Interface

```bash
python3 python/printDatacard.py \
  --era ERA             # 2016preVFP, 2016postVFP, 2017, 2018, 2022, 2022EE, 2023, 2023BPix
  --channel CHANNEL     # SR1E2Mu, SR3Mu
  --masspoint MASSPOINT # MHc130_MA90, MHc100_MA60, etc.
  --method METHOD       # Baseline, ParticleNet, GBDT
  [--output PATH]       # Optional output path (default: auto-determined)
```

**Example**:
```bash
python3 python/printDatacard.py \
  --era 2017 \
  --channel SR1E2Mu \
  --masspoint MHc130_MA90 \
  --method Baseline
```

---

#### Core Methods

##### DatacardManager.check_systematic_validity

```python
def check_systematic_validity(self, syst_name, applies_to):
    """
    Check if systematic variations produce negative normalizations.

    IMPORTANT: This method detects negative histogram INTEGRALS (normalizations),
    not individual negative bins. Negative bins are acceptable, but negative
    normalizations cause "Bogus norm" errors in text2workspace.py.

    Args:
        syst_name (str): Base systematic name without Up/Down suffix
            - Examples: "MuonIDSF", "BtagSF_HFcorr", "PileupReweight"
        applies_to (list[str]): List of processes to check
            - Valid processes: "signal", "nonprompt", "conversion", "diboson", "ttX", "others"

    Returns:
        tuple: (has_negative, details)
            - has_negative (bool): True if any process has negative normalization
            - details (list[str]): Human-readable description of problematic variations
              Format: "{process} {Up|Down}: {normalization:.4e}"

    Algorithm:
        For each process in applies_to:
            1. Get central normalization: hist.Integral()
            2. Get Up variation normalization
            3. Get Down variation normalization
            4. If central > 0 but Up/Down ≤ 0: Flag as problematic

    Use Case:
        Called by syststring() to determine if shape systematic should be
        converted to lnN (log-normal) due to negative normalization issues.

    Logged Output:
        WARNING: Negative normalization detected for {syst}, switching to lnN
          {process} Up: {norm:.4e}
          {process} Down: {norm:.4e}
    """
```

**Usage**:
```python
# In syststring() method
if sysType == "shape":
    applies_to = [p for p in ["signal"] + self.backgrounds if p not in skip]
    has_negative, details = self.check_systematic_validity(syst, applies_to)

    if has_negative:
        print(f"WARNING: Negative normalization detected for {syst}, switching to lnN", file=sys.stderr)
        for detail in details:
            print(f"  {detail}", file=sys.stderr)
        sysType = "lnN"
        # Calculate appropriate lnN value based on variation size
```

**Example Output**:
```
WARNING: Negative normalization detected for PileupReweight, switching to lnN
  conversion Up: 5.2341e-03
  conversion Down: -4.4586e-03
```

**Technical Context**:
- **When it triggers**: Low-yield processes (conversion ~0.4 events) with systematic variations
- **Why conversions fail**: Statistical fluctuations + negative MC weights → negative integral
- **Automatic fix**: Switches shape → lnN to avoid text2workspace.py "Bogus norm" error
- **Impact on physics**: Minor - lnN provides flat rate uncertainty instead of shape variation
- **Related**: See [NEGATIVE_BINS_HANDLING.md](../docs/NEGATIVE_BINS_HANDLING.md) for comprehensive explanation

---

##### DatacardManager.syststring

```python
def syststring(self, syst, sysType, value=None, skip=None, denoteEra=False):
    """
    Generate systematic line with automatic shape→lnN conversion for negative rates.

    This method is the primary interface for adding systematics to datacards. It
    automatically handles problematic shape systematics by converting them to lnN.

    Args:
        syst (str): Systematic name (e.g., "MuonIDSF", "Nonprompt")
        sysType (str): Type of systematic
            - "shape": Histogram-based uncertainty (can vary bin-by-bin)
            - "lnN": Log-normal rate uncertainty (flat across bins)
        value (float, optional): For lnN systematics, the uncertainty value (e.g., 1.10 for ±10%)
        skip (list[str], optional): Processes to skip (use "-" in datacard)
        denoteEra (bool, optional): Add era suffix for uncorrelated uncertainties

    Returns:
        str: Formatted datacard line
            Format: "{name}\t{type}\t{values}\t"
            Example: "MuonIDSF_17\tshape\t1\t\t1\t\t1\t\t1\t\t-\t\t"

    Behavior (Shape Systematics):
        1. Check if systematic applies to any present process
        2. Call check_systematic_validity()
        3. If negative normalizations detected:
           - Switch sysType from "shape" to "lnN"
           - Calculate appropriate lnN value from variation magnitude
           - Log warning to stderr
        4. Build datacard line with proper formatting

    Behavior (lnN Systematics):
        - Use provided value
        - Apply to specified processes only

    Automatic Handling Features:
        - Skips systematics that don't apply (e.g., Nonprompt if nonprompt background absent)
        - Adds era suffix for uncorrelated systematics
        - Tracks era-suffixed systematics for ROOT file renaming
        - Calculates appropriate lnN values when converting from shape
    """
```

**Usage**:
```python
# Example 1: Shape systematic (may auto-convert to lnN)
line = manager.syststring(
    syst="PileupReweight",
    sysType="shape",
    skip=["nonprompt"],  # Nonprompt uses data-driven uncertainty
    denoteEra=True
)
# Output: "PileupReweight_17\tshape\t1\t\t-\t\t1\t\t1\t\t1\t\t1\t\t"
#         or "PileupReweight_17\tlnN\t1.050\t\t-\t\t1.020\t\t..." (if negative norm)

# Example 2: lnN systematic
line = manager.syststring(
    syst="Luminosity",
    sysType="lnN",
    value=1.016,  # 1.6% uncertainty
    denoteEra=False
)
# Output: "Luminosity\tlnN\t1.016\t\t1.016\t\t1.016\t\t..."
```

---

#### DatacardManager Workflow

**Full Datacard Generation**:

1. **Initialization**
   ```python
   manager = DatacardManager(era, channel, masspoint, method, backgrounds)
   # Opens shapes.root
   # Checks which backgrounds have positive yields
   ```

2. **Load Configuration**
   ```python
   syst_config = load_datacard_systematics(era, channel)
   # Loads systematics from configs/systematics.json
   # Organized by category: experimental, datadriven, normalization
   ```

3. **Generate Datacard**
   ```python
   datacard = manager.generate_datacard(syst_config)
   # Calls syststring() for each systematic
   # Automatically converts shape→lnN when needed
   # Returns complete datacard string
   ```

4. **Update ROOT File**
   ```python
   manager.update_root_file_era_suffix()
   # Renames histograms for uncorrelated systematics
   # Example: MHc130_MA90_PileupReweightUp → MHc130_MA90_PileupReweight_17Up
   ```

5. **Save Datacard**
   ```python
   with open(output_path, 'w') as f:
       f.write(datacard)
   ```

---

#### Output Structure

**Datacard Format** (example excerpt):
```
# Datacard for charged Higgs search
# Era: 2017, Channel: SR1E2Mu, Masspoint: MHc130_MA90, Method: Baseline
# Signal cross-section scaled to 5 fb
--------------------------------------------------------------------------------
imax		1 number of bins
jmax		4 number of backgrounds
kmax		* number of nuisance parameters
--------------------------------------------------------------------------------
shapes	*	*	shapes.root	$PROCESS	$PROCESS_$SYSTEMATIC
shapes	signal	*	shapes.root	MHc130_MA90	MHc130_MA90_$SYSTEMATIC
--------------------------------------------------------------------------------
bin			signal_region
observation		105.6792
--------------------------------------------------------------------------------
bin			signal_region	signal_region	signal_region	signal_region	signal_region	signal_region
process			signal		nonprompt	conversion	diboson		ttX		others
process			0		1		2		3		4		5
rate			-1		-1		-1		-1		-1		-1
--------------------------------------------------------------------------------
signal_region	autoMCStats	10
L1Prefire_17	shape		1		-		1		1		1		1
MuonIDSF_17	shape		1		-		1		1		1		1
PileupReweight_17	lnN	1.050		-		1.020		1.015		1.030		1.012
Nonprompt	lnN		-		1.300		-		-		-		-
ConvSF		lnN		-		-		1.100		-		-		-
```

**Key Features**:
- Era suffixes (_17) for uncorrelated systematics
- Shape systematics (1 = present, - = not applicable)
- lnN systematics with explicit values
- Automatic conversion applied (PileupReweight switched from shape to lnN)
- autoMCStats for statistical uncertainties

---

#### Known Behavior

**Automatic Conversions**:
- Most common: PileupReweight systematic in conversion background (low yield ~0.4 events)
- Typical scenarios: Any shape systematic on processes with <1 expected event
- Conservative fallback: Uses maximum variation magnitude for lnN value

**Performance**:
- Runtime: <1 minute per datacard
- I/O: Reads shapes.root multiple times (once per check)
- Memory: ~100 MB peak usage

**Validation**:
- Checks process existence before adding systematics
- Verifies histogram availability for shape systematics
- Logs all automatic conversions to stderr

---

## Usage Examples

### Complete Preprocessing Workflow

```cpp
#include "Preprocessor.h"

void preprocess_example() {
    // Initialize preprocessor
    Preprocessor prep("2022", "Skim1E2Mu", "SingleMuon");
    prep.setConvSF(1.05, 0.10);

    // Open files
    prep.setInputFile("data/2022/Skim1E2Mu/DYJets.root");
    prep.setOutputFile("samples/2022/SR1E2Mu/DYJets.root");

    // Process central values
    prep.setInputTree("Central");
    prep.fillOutTree("DYJets", "MHc130_MA90", "Central", true, true);
    prep.saveTree();

    // Process systematic variations
    vector<TString> systs = {"MuonIDSFUp", "MuonIDSFDown", "PileupReweightUp"};
    for (const auto &syst : systs) {
        prep.setInputTree(syst);
        prep.fillOutTree("DYJets", "MHc130_MA90", syst, true, true);
        prep.saveTree();
    }

    // Cleanup
    prep.closeOutputFile();
    prep.closeInputFile();

    cout << "Preprocessing complete!" << endl;
}
```

---

### Complete Fitting Workflow

```cpp
#include "AmassFitter.h"

void fit_example() {
    // Initialize fitter
    AmassFitter fitter("samples/2022/SR3Mu/MHc130_MA90.root",
                       "fit_results/mA90_fit.root");

    // Perform fit
    fitter.fitMass(90.0, 80.0, 100.0);

    // Extract results
    double mA = fitter.getRooMA()->getVal();
    double mA_err = fitter.getRooMA()->getError();
    double sigma = fitter.getRooSigma()->getVal();
    double width = fitter.getRooWidth()->getVal();

    // Print results
    cout << "Fitted mA: " << mA << " ± " << mA_err << " GeV" << endl;
    cout << "Resolution: " << sigma << " GeV" << endl;
    cout << "Natural width: " << width << " GeV" << endl;

    // Save diagnostic plot
    fitter.saveCanvas("plots/mA90_fit.pdf");

    // Persist results
    fitter.Close();

    cout << "Fitting complete!" << endl;
}
```

---

### Python Integration (via ROOT)

```python
import ROOT

# Load library
ROOT.gSystem.Load("lib/libSignalRegionStudy.so")

# Use Preprocessor from Python
prep = ROOT.Preprocessor("2022", "Skim1E2Mu", "SingleMuon")
prep.setConvSF(1.05, 0.10)
prep.setInputFile("data/2022/Skim1E2Mu/DYJets.root")
prep.setOutputFile("samples/2022/SR1E2Mu/DYJets.root")
prep.setInputTree("Central")
prep.fillOutTree("DYJets", "MHc130_MA90", "Central", True, True)
prep.saveTree()
prep.closeOutputFile()
prep.closeInputFile()

# Use AmassFitter from Python
fitter = ROOT.AmassFitter("samples/2022/SR3Mu/MHc130_MA90.root",
                          "fit_results/mA90_fit.root")
fitter.fitMass(90.0, 80.0, 100.0)
fitter.saveCanvas("plots/mA90_fit.pdf")

# Access results
mA = fitter.getRooMA().getVal()
mA_err = fitter.getRooMA().getError()
print(f"Fitted mA: {mA:.2f} ± {mA_err:.2f} GeV")

fitter.Close()
```

---

## Shell Script API

This section documents the shell scripts used for automated template preparation and statistical analysis.

### `prepareCombine.sh`

**Location**: `scripts/prepareCombine.sh`

**Purpose**: Automated template creation pipeline (runs 4 sub-steps)

**Syntax**:
```bash
./scripts/prepareCombine.sh <ERA> <CHANNEL> <MASSPOINT> <METHOD>
```

**Parameters**:

| Parameter | Type | Description | Valid Values |
|-----------|------|-------------|--------------|
| `ERA` | String | Data-taking period | `2016preVFP`, `2016postVFP`, `2017`, `2018`, `2022`, `2022EE`, `2023`, `2023BPix` |
| `CHANNEL` | String | Analysis channel | `SR1E2Mu`, `SR3Mu` |
| `MASSPOINT` | String | Signal hypothesis | `MHc100_MA95`, `MHc115_MA87`, `MHc130_MA90`, `MHc145_MA92`, `MHc160_MA85`, `MHc160_MA98` |
| `METHOD` | String | Discrimination method | `Baseline`, `ParticleNet` |

**Execution Steps**:

1. **Environment Setup**:
   ```bash
   export PATH="${PWD}/python:${PATH}"
   export LD_LIBRARY_PATH="${PWD}/lib:${LD_LIBRARY_PATH}"
   ```

2. **Directory Cleanup**:
   ```bash
   rm -rf samples/$ERA/$CHANNEL/$MASSPOINT
   rm -rf templates/$ERA/$CHANNEL/$MASSPOINT
   ```

3. **Preprocessing** (sub-step 1):
   ```bash
   preprocess.py --era $ERA --channel $CHANNEL --signal $MASSPOINT --method Baseline
   ```
   - Loads Preprocessor C++ class
   - Processes all samples (nonprompt, diboson, ttX, conversion, signal)
   - Outputs: `samples/$ERA/$CHANNEL/$MASSPOINT/Baseline/*.root`

4. **Template Creation** (sub-step 2):
   ```bash
   makeBinnedTemplates.py --era $ERA --channel $CHANNEL \
       --masspoint $MASSPOINT --method $METHOD
   ```
   - Fits signal mass distribution
   - Creates binned templates (~159 histograms)
   - Outputs: `templates/$ERA/$CHANNEL/$MASSPOINT/Shape/$METHOD/shapes.root`

5. **Validation** (sub-step 3):
   ```bash
   checkTemplates.py --era $ERA --channel $CHANNEL \
       --masspoint $MASSPOINT --method $METHOD
   ```
   - Validates histogram integrity
   - Checks for negative bins/integrals
   - Outputs: validation plots in `validation/` subdirectory

6. **Datacard Generation** (sub-step 4):
   ```bash
   printDatacard.py --era $ERA --channel $CHANNEL \
       --masspoint $MASSPOINT --method $METHOD
   ```
   - Generates HiggsCombine datacard
   - Creates RooFit workspace for signal
   - Outputs: `datacard.txt`, `fit_result.root`, `signal_fit.png`

7. **Optional ParticleNet Plotting** (if METHOD=ParticleNet):
   ```bash
   plotScores.py --era $ERA --channel $CHANNEL \
       --masspoint $MASSPOINT --method $METHOD
   ```

**Outputs**:

```
samples/$ERA/$CHANNEL/$MASSPOINT/Baseline/
├── nonprompt.root
├── diboson.root
├── ttX.root
├── conversion.root
├── others.root
└── $MASSPOINT.root

templates/$ERA/$CHANNEL/$MASSPOINT/Shape/$METHOD/
├── shapes.root              # All histogram templates
├── datacard.txt             # HiggsCombine input
├── fit_result.root          # Signal fit workspace
├── signal_fit.png           # Diagnostic plot
└── validation/              # QA plots
```

**Example Usage**:

```bash
# Process 2022 data for SR1E2Mu channel
./scripts/prepareCombine.sh 2022 SR1E2Mu MHc130_MA90 ParticleNet

# Process all Run2 eras
for era in 2016preVFP 2016postVFP 2017 2018; do
    ./scripts/prepareCombine.sh $era SR1E2Mu MHc130_MA90 ParticleNet
done
```

**Exit Codes**:
- `0`: Success
- Non-zero: Error in one of the sub-steps

**Dependencies**:
- Python scripts: `preprocess.py`, `makeBinnedTemplates.py`, `checkTemplates.py`, `printDatacard.py`
- C++ library: `lib/libSignalRegionStudy.so`
- Input data: `$WORKDIR/SKNanoOutput/SignalRegion/$ERA/`

---

### `runCombine.sh`

**Location**: `scripts/runCombine.sh`

**Purpose**: Execute HiggsCombine statistical framework

**Syntax**:
```bash
./scripts/runCombine.sh <ERA> <CHANNEL> <MASSPOINT> <METHOD>
```

**Parameters**:

| Parameter | Type | Description | Valid Values |
|-----------|------|-------------|--------------|
| `ERA` | String | Data-taking period or combination | `2016preVFP`, `2016postVFP`, `2017`, `2018`, `2022`, `2022EE`, `2023`, `2023BPix`, **`FullRun2`** |
| `CHANNEL` | String | Analysis channel or combination | `SR1E2Mu`, `SR3Mu`, **`Combined`** |
| `MASSPOINT` | String | Signal hypothesis | Same as `prepareCombine.sh` |
| `METHOD` | String | Discrimination method | `Baseline`, `ParticleNet` |

**Special Parameter Values**:
- `ERA=FullRun2`: Combines all Run2 eras (2016preVFP + 2016postVFP + 2017 + 2018)
- `CHANNEL=Combined`: Merges SR1E2Mu + SR3Mu channels using `combineCards.py`

**Execution Steps**:

1. **Environment Check**:
   ```bash
   if [ -z "$WORKDIR" ]; then
       echo "Error: WORKDIR is not set. Please run 'source setup.sh' first."
       exit 1
   fi
   ```

2. **Navigate to Template Directory**:
   ```bash
   BASEDIR="templates/$ERA/$CHANNEL/$MASSPOINT/Shape/$METHOD"
   cd $BASEDIR
   ```

3. **Card Combination** (if needed):

   For **Combined** channel:
   ```bash
   combineCards.py \
       ch1=../../../SR1E2Mu/$MASSPOINT/Shape/$METHOD/datacard.txt \
       ch2=../../../SR3Mu/$MASSPOINT/Shape/$METHOD/datacard.txt \
       > datacard.txt
   ```

   For **FullRun2** era:
   ```bash
   combineCards.py \
       era1=../../../../../2016preVFP/$CHANNEL/$MASSPOINT/Shape/$METHOD/datacard.txt \
       era2=../../../../../2016postVFP/$CHANNEL/$MASSPOINT/Shape/$METHOD/datacard.txt \
       era3=../../../../../2017/$CHANNEL/$MASSPOINT/Shape/$METHOD/datacard.txt \
       era4=../../../../../2018/$CHANNEL/$MASSPOINT/Shape/$METHOD/datacard.txt \
       > datacard.txt
   ```

4. **Create RooWorkspace**:
   ```bash
   text2workspace.py datacard.txt -o workspace.root
   ```
   - Converts datacard to RooFit workspace
   - Builds full probability model

5. **Fit Diagnostics**:
   ```bash
   combine -M FitDiagnostics workspace.root
   ```
   - Performs maximum likelihood fit
   - Generates pre-fit and post-fit shapes
   - Output: `higgsCombineTest.FitDiagnostics.mH120.root`

6. **Asymptotic Limits**:
   ```bash
   combine -M AsymptoticLimits workspace.root -t -1
   ```
   - Calculates expected limits (Asimov dataset)
   - Uses CLs method
   - Output: `higgsCombineTest.AsymptoticLimits.mH120.root`

7. **Return to Working Directory**:
   ```bash
   cd $WORKDIR
   ```

**Commented-Out Advanced Methods** (lines 46-54):

```bash
# HybridNew method (toy-based limits)
# combine -M HybridNew workspace.root --LHCmode LHC-limits -T 500 -m 120

# Impact analysis
# combineTool.py -M Impacts -d workspace.root -m 120 --doInitialFit --robustFit 1
# combineTool.py -M Impacts -d workspace.root -m 120 --doFits --robustFit 1
# plotImpacts.py -i impacts.json -o impacts
```

**Outputs**:

```
templates/$ERA/$CHANNEL/$MASSPOINT/Shape/$METHOD/
├── workspace.root                                    # RooFit workspace
├── higgsCombineTest.FitDiagnostics.mH120.root       # Fit results
└── higgsCombineTest.AsymptoticLimits.mH120.root     # Limit values
```

**Example Usage**:

```bash
# Single era, single channel
./scripts/runCombine.sh 2022 SR1E2Mu MHc130_MA90 ParticleNet

# Combined channel (requires both channels prepared)
./scripts/prepareCombine.sh 2022 SR1E2Mu MHc130_MA90 ParticleNet
./scripts/prepareCombine.sh 2022 SR3Mu MHc130_MA90 ParticleNet
./scripts/runCombine.sh 2022 Combined MHc130_MA90 ParticleNet

# FullRun2 (requires all eras prepared)
for era in 2016preVFP 2016postVFP 2017 2018; do
    ./scripts/prepareCombine.sh $era SR1E2Mu MHc130_MA90 ParticleNet
    ./scripts/runCombine.sh $era SR1E2Mu MHc130_MA90 ParticleNet
done
./scripts/runCombine.sh FullRun2 SR1E2Mu MHc130_MA90 ParticleNet
```

**Exit Codes**:
- `0`: Success
- `1`: WORKDIR not set
- Non-zero: Combine command failed

**Dependencies**:
- Must run `prepareCombine.sh` first
- HiggsCombine tools: `text2workspace.py`, `combine`
- For combinations: `combineCards.py`
- CMSSW environment (`cmsenv` in setup.sh)

---

### `runCombineWrapper.sh`

**Location**: `scripts/runCombineWrapper.sh`

**Purpose**: Orchestrate complete workflow for single masspoint across all eras and channels

**Syntax**:
```bash
./scripts/runCombineWrapper.sh <MASSPOINT> <METHOD>
```

**Parameters**:

| Parameter | Type | Description | Valid Values |
|-----------|------|-------------|--------------|
| `MASSPOINT` | String | Signal hypothesis | Same as above |
| `METHOD` | String | Discrimination method | `Baseline`, `ParticleNet` |

**Execution Flow**:

1. **Environment Setup**:
   ```bash
   cd /data9/Users/choij/workspace/ChargedHiggsAnalysisV3/SignalRegionStudy
   source setup.sh
   ```
   **Note**: Line 26 contains hardcoded path - should use `$WORKDIR` instead

2. **Define Eras**:
   ```bash
   ERAs=("2016preVFP" "2016postVFP" "2017" "2018")
   ```

3. **Process SR1E2Mu Channel** (all eras):
   ```bash
   for era in ${ERAs[@]}; do
       ./scripts/prepareCombine.sh $era SR1E2Mu $MASSPOINT $METHOD
       ./scripts/runCombine.sh $era SR1E2Mu $MASSPOINT $METHOD
   done
   ```

4. **Process SR3Mu Channel** (all eras):
   ```bash
   for era in ${ERAs[@]}; do
       ./scripts/prepareCombine.sh $era SR3Mu $MASSPOINT $METHOD
       ./scripts/runCombine.sh $era SR3Mu $MASSPOINT $METHOD
   done
   ```

5. **Process Combined Channels** (all eras):
   ```bash
   for era in ${ERAs[@]}; do
       ./scripts/runCombine.sh $era Combined $MASSPOINT $METHOD
   done
   ```

6. **Process FullRun2 Combinations**:
   ```bash
   ./scripts/runCombine.sh FullRun2 SR1E2Mu $MASSPOINT $METHOD
   ./scripts/runCombine.sh FullRun2 SR3Mu $MASSPOINT $METHOD
   ./scripts/runCombine.sh FullRun2 Combined $MASSPOINT $METHOD
   ```

**Total Executions**: 15 combinations per masspoint
- 8 individual era×channel combinations
- 4 combined channel (per era)
- 3 FullRun2 combinations

**Example Usage**:

```bash
# Process single masspoint through all combinations
./scripts/runCombineWrapper.sh MHc130_MA90 ParticleNet

# Process multiple masspoints in parallel (via doThis.sh)
parallel -j 18 "./scripts/runCombineWrapper.sh" {1} {2} \
    ::: MHc100_MA95 MHc130_MA90 MHc160_MA85 ::: "ParticleNet"
```

**Runtime**: ~30-60 minutes per masspoint

**Resource Requirements**:
- **CPU**: 1 core per wrapper instance
- **RAM**: 2-4 GB per instance
- **Disk**: 5-10 GB per masspoint (all eras/channels)

**Exit Codes**:
- `0`: Success
- Non-zero: Error in one of the sub-scripts

**Known Issues**:
- **Line 26**: Hardcoded absolute path - breaks for other users
  ```bash
  # Current (problematic):
  cd /data9/Users/choij/workspace/ChargedHiggsAnalysisV3/SignalRegionStudy

  # Suggested fix:
  cd $WORKDIR/SignalRegionStudy
  ```

**Dependencies**:
- `prepareCombine.sh`
- `runCombine.sh`
- All dependencies of above scripts

---

### Shell Script Best Practices

#### 1. Error Handling

**Check exit codes**:
```bash
./scripts/prepareCombine.sh 2022 SR1E2Mu MHc130_MA90 ParticleNet
if [ $? -ne 0 ]; then
    echo "Error: prepareCombine.sh failed"
    exit 1
fi
```

**Validate inputs**:
```bash
# Check era is valid
if [[ ! " 2016preVFP 2016postVFP 2017 2018 2022 2022EE 2023 2023BPix FullRun2 " =~ " $ERA " ]]; then
    echo "Error: Invalid ERA: $ERA"
    exit 1
fi
```

#### 2. Logging

**Redirect output**:
```bash
# Save logs
./scripts/prepareCombine.sh 2022 SR1E2Mu MHc130_MA90 ParticleNet \
    > logs/prepare_2022_SR1E2Mu_MHc130_MA90.log 2>&1
```

**Track progress**:
```bash
# Monitor in real-time
tail -f logs/prepare_2022_SR1E2Mu_MHc130_MA90.log
```

#### 3. Parallel Execution

**GNU parallel**:
```bash
# Process multiple eras in parallel
parallel -j 4 "./scripts/prepareCombine.sh" {1} SR1E2Mu MHc130_MA90 ParticleNet \
    ::: 2016preVFP 2016postVFP 2017 2018
```

**Resource management**:
```bash
# Limit concurrent jobs based on available RAM
NCORES=$(nproc)
MEM_PER_JOB=4  # GB
AVAILABLE_MEM=$(free -g | awk '/^Mem:/{print $7}')
MAX_JOBS=$((AVAILABLE_MEM / MEM_PER_JOB))
JOBS=$(($NCORES < $MAX_JOBS ? $NCORES : $MAX_JOBS))

parallel -j $JOBS "./scripts/runCombineWrapper.sh" {1} ParticleNet \
    ::: MHc100_MA95 MHc130_MA90 MHc160_MA85
```

#### 4. Debugging

**Dry run mode**:
```bash
# Add to scripts for debugging
DRY_RUN=${DRY_RUN:-false}

if [ "$DRY_RUN" = "true" ]; then
    echo "Would execute: preprocess.py --era $ERA ..."
else
    preprocess.py --era $ERA ...
fi

# Usage:
DRY_RUN=true ./scripts/prepareCombine.sh 2022 SR1E2Mu MHc130_MA90 ParticleNet
```

**Verbose output**:
```bash
# Add to scripts
set -x  # Print each command before execution
set -e  # Exit on first error

# Or run with bash -x
bash -x ./scripts/prepareCombine.sh 2022 SR1E2Mu MHc130_MA90 ParticleNet
```

#### 5. Configuration Management

**Use environment variables**:
```bash
# config.sh
export DEFAULT_METHOD="ParticleNet"
export MASSPOINTS=("MHc100_MA95" "MHc130_MA90" "MHc160_MA85")
export RUN2_ERAS=("2016preVFP" "2016postVFP" "2017" "2018")

# In scripts:
source config.sh
for mp in "${MASSPOINTS[@]}"; do
    ./scripts/runCombineWrapper.sh $mp $DEFAULT_METHOD
done
```

---

## Notes and Best Practices

### Memory Management
- ROOT owns created objects (trees, files) - manual `delete` not required
- Call `Close()` methods to ensure proper file writing
- RooFit objects cleaned up automatically when fitter goes out of scope

### Performance Considerations
- Process systematic variations sequentially (not parallel) to avoid memory issues
- For large datasets, consider chunked processing with multiple output files
- RooFit fitting benefits from good initial parameter guesses

### Error Handling
- Check tree pointers before use: `if (!tree) { cerr << "Tree not found!"; }`
- Verify fit status: `if (result->status() != 0) { cerr << "Fit failed!"; }`
- Handle missing branches gracefully in production code

### Debugging Tips
- Use `getOutTree()->Print()` to inspect branch structure
- Check entry counts: `cout << tree->GetEntries() << " entries processed"`
- For RooFit issues, enable verbose output: `RooMsgService::instance().setGlobalKillBelow(RooFit::DEBUG)`

---

**Last Updated**: 2025-10-10
**Maintainer**: ChargedHiggsAnalysisV3 Development Team

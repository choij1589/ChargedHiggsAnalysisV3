#include "Preprocessor.h"

Preprocessor::Preprocessor(const TString &era,
                           const TString &channel,
                           const TString &datastream)
    : era(era), channel(channel), datastream(datastream) {}

void Preprocessor::setInputTree(const TString &syst) {
    if (!inFile || inFile->IsZombie()) {
        cerr << "ERROR: Input file is invalid or not opened" << endl;
        exit(EXIT_FAILURE);
    }

    // Try PromptTreeProducer structure first (Events_Central)
    centralTree = static_cast<TTree*>(inFile->Get("Events_Central"));

    // If not found, try MatrixTreeProducer structure (Events)
    if (!centralTree) {
        centralTree = static_cast<TTree*>(inFile->Get("Events"));
        if (!centralTree) {
            cerr << "ERROR: Could not find Events_Central or Events in file" << endl;
            exit(EXIT_FAILURE);
        }
        // For MatrixTreeProducer files, use tree/Events directly (no systematics)
        inTree = centralTree;
        return;
    }

    // For PromptTreeProducer files, get the systematic variation tree
    inTree = static_cast<TTree*>(inFile->Get(("Events_" + syst).Data()));
    if (!inTree) {
        cerr << "ERROR: Could not find Events_" << syst << " in file" << endl;
        exit(EXIT_FAILURE);
    }
}

void Preprocessor::fillOutTree(const TString &sampleName, const TString &signal, const TString &syst, const bool applyConvSF, const bool isTrainedSample, const double weightScale) {
    outTree = new TTree(syst, "");
    outTree->Branch("mass", &mass);
    outTree->Branch("mass1", &mass1);
    outTree->Branch("mass2", &mass2);
    outTree->Branch("MT1", &MT1);
    outTree->Branch("MT2", &MT2);
    if (isTrainedSample) {
        outTree->Branch(("score_" + signal + "_signal").Data(), &score_0);
        outTree->Branch(("score_" + signal + "_nonprompt").Data(), &score_1);
        outTree->Branch(("score_" + signal + "_diboson").Data(), &score_2);
        outTree->Branch(("score_" + signal + "_ttZ").Data(), &score_3);
    }
    outTree->Branch("weight", &weight);
    outTree->Branch("fold", &fold);

    // Set branch addresses
    inTree->SetBranchAddress("mass1", &mass1);
    inTree->SetBranchAddress("mass2", &mass2);
    inTree->SetBranchAddress("MT1", &MT1);
    inTree->SetBranchAddress("MT2", &MT2);
    if (isTrainedSample) {
        inTree->SetBranchAddress(("score_" + signal + "_signal").Data(), &score_0);
        inTree->SetBranchAddress(("score_" + signal + "_nonprompt").Data(), &score_1);
        inTree->SetBranchAddress(("score_" + signal + "_diboson").Data(), &score_2);
        inTree->SetBranchAddress(("score_" + signal + "_ttZ").Data(), &score_3);
    }

    inTree->SetBranchAddress("weight", &weight);
    inTree->SetBranchAddress("fold", &fold);

    // Loop over tree entries
    for (unsigned int entry = 0; entry < inTree->GetEntries(); ++entry) {
        inTree->GetEntry(entry);

        // Signal cross-section scaling to 5 fb
        if (sampleName.Contains("MA")) {
            weight /= 3.0;
        } else if (applyConvSF) {
            weight *= getConvSF();
        }

        // Apply weight scale for systematic variations (e.g., nonprompt up/down)
        weight *= weightScale;

        // Process based on the channel
        if (channel.Contains("1E2Mu")) {
            mass = mass1;
            outTree->Fill();
        } else if (channel.Contains("3Mu")) {
            // Extract mass point parameters
            int mHc = extractMHc(signal);
            int mA = extractMA(signal);

            // New selection algorithm based on mass point thresholds
            // For MHc >= 100 AND MA >= 60: use max(mass1, mass2)
            // Otherwise: use min(mass1, mass2)
            if (mHc >= 100 && mA >= 60) {
                mass = (mass1 > mass2) ? mass1 : mass2;  // max
            } else {
                mass = (mass1 < mass2) ? mass1 : mass2;  // min
            }
            outTree->Fill();
        } else {
            cerr << "Wrong channel: " << channel << endl;
            exit(EXIT_FAILURE);
        }
    }
}

int Preprocessor::extractMHc(const TString &signal) const {
    // Extract MHc value from string like "MHc130_MA90"
    if (!signal.Contains("MHc")) return -1;
    TString temp = signal;
    temp.Remove(0, temp.Index("MHc") + 3);  // Remove up to and including "MHc"
    if (temp.Contains("_")) temp.Remove(temp.Index("_"));  // Remove everything after "_"
    return temp.Atoi();
}

int Preprocessor::extractMA(const TString &signal) const {
    // Extract MA value from string like "MHc130_MA90"
    if (!signal.Contains("MA")) return -1;
    TString temp = signal;
    temp.Remove(0, temp.Index("MA") + 2);  // Remove up to and including "MA"
    return temp.Atoi();
}

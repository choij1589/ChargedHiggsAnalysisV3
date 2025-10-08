#ifndef DATAFORMAT_H
#define DATAFORMAT_H

#include <TTree.h>
#include <TLorentzVector.h>
#include <TMath.h>
#include <ROOT/RVec.hxx>
#include <memory>

using namespace ROOT::VecOps;

// Base class for all physics objects
class Particle : public TLorentzVector {
protected:
    Float_t charge;
    Float_t btagScore;
    Bool_t isMuon;
    Bool_t isElectron;
    Bool_t isJet;

public:
    Particle();
    Particle(Double_t pt, Double_t eta, Double_t phi, Double_t mass);
    
    Float_t Charge() const;
    Float_t BtagScore() const;
    Bool_t IsMuon() const;
    Bool_t IsElectron() const;
    Bool_t IsJet() const;
    
    Double_t MT(const Particle& part) const;
    
    void SetCharge(Float_t c);
    void SetBtagScore(Float_t score);
};

// Lepton base class
class Lepton : public Particle {
public:
    Lepton(Double_t pt, Double_t eta, Double_t phi, Double_t mass);
};

// Muon class
class Muon : public Lepton {
public:
    Muon(Double_t pt, Double_t eta, Double_t phi, Double_t mass);
};

// Electron class
class Electron : public Lepton {
public:
    Electron(Double_t pt, Double_t eta, Double_t phi, Double_t mass);
};

// Jet class
class Jet : public Particle {
protected:
    Bool_t isBtagged;
    
public:
    Jet(Double_t pt, Double_t eta, Double_t phi, Double_t mass);
    
    void SetBtagging(Bool_t tagged);
    Bool_t IsBtagged() const;
};

// Event data structure
struct EventData {
    // Muon collections
    RVec<float>* MuonPtColl = nullptr;
    RVec<float>* MuonEtaColl = nullptr;
    RVec<float>* MuonPhiColl = nullptr;
    RVec<float>* MuonMassColl = nullptr;
    RVec<float>* MuonChargeColl = nullptr;
    
    // Electron collections
    RVec<float>* ElectronPtColl = nullptr;
    RVec<float>* ElectronEtaColl = nullptr;
    RVec<float>* ElectronPhiColl = nullptr;
    RVec<float>* ElectronMassColl = nullptr;
    RVec<float>* ElectronChargeColl = nullptr;
    
    // Jet collections
    RVec<float>* JetPtColl = nullptr;
    RVec<float>* JetEtaColl = nullptr;
    RVec<float>* JetPhiColl = nullptr;
    RVec<float>* JetMassColl = nullptr;
    RVec<float>* JetChargeColl = nullptr;
    RVec<float>* JetBtagScoreColl = nullptr;
    RVec<bool>* JetIsBtaggedColl = nullptr;
    
    // MET and other event variables
    Float_t METvPt;
    Float_t METvPhi;
    Int_t nJets;
    
    void SetBranches(TTree* tree);
};

// Multi-class labels
enum ClassLabel {
    SIGNAL = 0,
    NONPROMPT = 1,
    DIBOSON = 2,
    TTZ = 3
};

// Helper functions
RVec<Muon> getMuons(const EventData& evt);
RVec<Electron> getElectrons(const EventData& evt);
std::pair<RVec<Jet>, RVec<Jet>> getJets(const EventData& evt);
RVec<RVec<float>> particlesToFeatures(const RVec<Muon>& muons,
                                      const RVec<Electron>& electrons,
                                      const RVec<Jet>& jets,
                                      const Particle& met);

#endif // DATAFORMAT_H

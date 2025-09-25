#include "DataFormat.h"

// Particle class implementations
Particle::Particle() : TLorentzVector(), charge(0), btagScore(0), isMuon(false), isElectron(false), isJet(false) {}

Particle::Particle(Double_t pt, Double_t eta, Double_t phi, Double_t mass) : TLorentzVector() {
    SetPtEtaPhiM(pt, eta, phi, mass);
    charge = 0;
    btagScore = 0;
    isMuon = false;
    isElectron = false;
    isJet = false;
}

Float_t Particle::Charge() const { return charge; }
Float_t Particle::BtagScore() const { return btagScore; }
Bool_t Particle::IsMuon() const { return isMuon; }
Bool_t Particle::IsElectron() const { return isElectron; }
Bool_t Particle::IsJet() const { return isJet; }

Double_t Particle::MT(const Particle& part) const {
    Double_t dPhi = DeltaPhi(part);
    return TMath::Sqrt(2 * Pt() * part.Pt() * (1. - TMath::Cos(dPhi)));
}

void Particle::SetCharge(Float_t c) { charge = c; }
void Particle::SetBtagScore(Float_t score) { btagScore = score; }

// Lepton class implementations
Lepton::Lepton(Double_t pt, Double_t eta, Double_t phi, Double_t mass) : Particle(pt, eta, phi, mass) {}

// Muon class implementations
Muon::Muon(Double_t pt, Double_t eta, Double_t phi, Double_t mass) : Lepton(pt, eta, phi, mass) {
    isMuon = true;
}

// Electron class implementations
Electron::Electron(Double_t pt, Double_t eta, Double_t phi, Double_t mass) : Lepton(pt, eta, phi, mass) {
    isElectron = true;
}

// Jet class implementations
Jet::Jet(Double_t pt, Double_t eta, Double_t phi, Double_t mass) : Particle(pt, eta, phi, mass) {
    isJet = true;
    isBtagged = false;
}

void Jet::SetBtagging(Bool_t tagged) { isBtagged = tagged; }
Bool_t Jet::IsBtagged() const { return isBtagged; }

// EventData implementations
void EventData::SetBranches(TTree* tree) {
    // Muon branches
    tree->SetBranchAddress("MuonPtColl", &MuonPtColl);
    tree->SetBranchAddress("MuonEtaColl", &MuonEtaColl);
    tree->SetBranchAddress("MuonPhiColl", &MuonPhiColl);
    tree->SetBranchAddress("MuonMassColl", &MuonMassColl);
    tree->SetBranchAddress("MuonChargeColl", &MuonChargeColl);
    
    // Electron branches
    tree->SetBranchAddress("ElectronPtColl", &ElectronPtColl);
    tree->SetBranchAddress("ElectronEtaColl", &ElectronEtaColl);
    tree->SetBranchAddress("ElectronPhiColl", &ElectronPhiColl);
    tree->SetBranchAddress("ElectronMassColl", &ElectronMassColl);
    tree->SetBranchAddress("ElectronChargeColl", &ElectronChargeColl);
    
    // Jet branches
    tree->SetBranchAddress("JetPtColl", &JetPtColl);
    tree->SetBranchAddress("JetEtaColl", &JetEtaColl);
    tree->SetBranchAddress("JetPhiColl", &JetPhiColl);
    tree->SetBranchAddress("JetMassColl", &JetMassColl);
    tree->SetBranchAddress("JetChargeColl", &JetChargeColl);
    tree->SetBranchAddress("JetBtagScoreColl", &JetBtagScoreColl);
    tree->SetBranchAddress("JetIsBtaggedColl", &JetIsBtaggedColl);
    
    // Event variables
    tree->SetBranchAddress("METvPt", &METvPt);
    tree->SetBranchAddress("METvPhi", &METvPhi);
    tree->SetBranchAddress("nJets", &nJets);
}

// Helper functions to extract particles from event data
RVec<Muon> getMuons(const EventData& evt) {
    RVec<Muon> muons;
    if (!evt.MuonPtColl) return muons;
    
    size_t nMuons = evt.MuonPtColl->size();
    muons.reserve(nMuons);
    for (size_t i = 0; i < nMuons; ++i) {
        Muon mu((*evt.MuonPtColl)[i], 
                (*evt.MuonEtaColl)[i], 
                (*evt.MuonPhiColl)[i], 
                (*evt.MuonMassColl)[i]);
        mu.SetCharge((*evt.MuonChargeColl)[i]);
        muons.push_back(mu);
    }
    return muons;
}

RVec<Electron> getElectrons(const EventData& evt) {
    RVec<Electron> electrons;
    if (!evt.ElectronPtColl) return electrons;
    
    size_t nElectrons = evt.ElectronPtColl->size();
    electrons.reserve(nElectrons);
    for (size_t i = 0; i < nElectrons; ++i) {
        Electron el((*evt.ElectronPtColl)[i], 
                    (*evt.ElectronEtaColl)[i], 
                    (*evt.ElectronPhiColl)[i], 
                    (*evt.ElectronMassColl)[i]);
        el.SetCharge((*evt.ElectronChargeColl)[i]);
        electrons.push_back(el);
    }
    return electrons;
}

std::pair<RVec<Jet>, RVec<Jet>> getJets(const EventData& evt) {
    RVec<Jet> jets;
    RVec<Jet> bjets;
    if (!evt.JetPtColl) return std::make_pair(jets, bjets);
    
    size_t nJets = evt.JetPtColl->size();
    jets.reserve(nJets);
    for (size_t i = 0; i < nJets; ++i) {
        Jet jet((*evt.JetPtColl)[i], 
                (*evt.JetEtaColl)[i], 
                (*evt.JetPhiColl)[i], 
                (*evt.JetMassColl)[i]);
        jet.SetCharge((*evt.JetChargeColl)[i]);
        jet.SetBtagScore((*evt.JetBtagScoreColl)[i]);
        jet.SetBtagging((*evt.JetIsBtaggedColl)[i]);
        
        jets.push_back(jet);
        if (jet.IsBtagged()) {
            bjets.push_back(jet);
        }
    }
    return std::make_pair(jets, bjets);
}

// Convert particles to feature array for ML
RVec<RVec<float>> particlesToFeatures(const RVec<Muon>& muons,
                                      const RVec<Electron>& electrons,
                                      const RVec<Jet>& jets,
                                      const Particle& met) {
    RVec<RVec<float>> features;
    features.reserve(muons.size() + electrons.size() + jets.size() + 1); // +1 for MET
    
    // Add muons
    for (const auto& mu : muons) {
        features.push_back({
            static_cast<float>(mu.E()), 
            static_cast<float>(mu.Px()), 
            static_cast<float>(mu.Py()), 
            static_cast<float>(mu.Pz()),
            mu.Charge(), 
            mu.BtagScore(),
            1.0f, // IsMuon
            0.0f, // IsElectron
            0.0f  // IsJet
        });
    }
    
    // Add electrons
    for (const auto& el : electrons) {
        features.push_back({
            static_cast<float>(el.E()), 
            static_cast<float>(el.Px()), 
            static_cast<float>(el.Py()), 
            static_cast<float>(el.Pz()),
            el.Charge(), 
            el.BtagScore(),
            0.0f, // IsMuon
            1.0f, // IsElectron
            0.0f  // IsJet
        });
    }
    
    // Add jets
    for (const auto& jet : jets) {
        features.push_back({
            static_cast<float>(jet.E()), 
            static_cast<float>(jet.Px()), 
            static_cast<float>(jet.Py()), 
            static_cast<float>(jet.Pz()),
            jet.Charge(), 
            jet.BtagScore(),
            0.0f, // IsMuon
            0.0f, // IsElectron
            1.0f  // IsJet
        });
    }
    
    // Add MET
    features.push_back({
        static_cast<float>(met.E()), 
        static_cast<float>(met.Px()), 
        static_cast<float>(met.Py()), 
        static_cast<float>(met.Pz()),
        0.0f, // Charge
        0.0f, // BtagScore
        0.0f, // IsMuon
        0.0f, // IsElectron
        0.0f  // IsJet
    });
    
    return features;
}
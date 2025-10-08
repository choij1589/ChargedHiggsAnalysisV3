// Test program to verify DataFormat with RVec
#include "DataFormat.h"
#include <iostream>

int main() {
    // Test creating particles
    Muon mu(50.0, 1.2, 0.5, 0.106);
    mu.SetCharge(-1);
    
    Electron el(30.0, -0.8, 2.1, 0.000511);
    el.SetCharge(1);
    
    Jet jet(100.0, 0.3, -1.5, 15.0);
    jet.SetBtagScore(0.8);
    jet.SetBtagging(true);
    
    // Test RVec functionality
    RVec<Muon> muons = {mu};
    RVec<Electron> electrons = {el};
    RVec<Jet> jets = {jet};
    
    std::cout << "Created particles:" << std::endl;
    std::cout << "Muon: pt=" << muons[0].Pt() << ", charge=" << muons[0].Charge() << std::endl;
    std::cout << "Electron: pt=" << electrons[0].Pt() << ", charge=" << electrons[0].Charge() << std::endl;
    std::cout << "Jet: pt=" << jets[0].Pt() << ", b-tagged=" << jets[0].IsBtagged() << std::endl;
    
    // Test particlesToFeatures
    Particle met(50.0, 0.0, 1.0, 0.0);
    RVec<RVec<float>> features = particlesToFeatures(muons, electrons, jets, met);
    
    std::cout << "\nFeature matrix size: " << features.size() << " particles" << std::endl;
    std::cout << "Each particle has " << features[0].size() << " features" << std::endl;
    
    return 0;
}

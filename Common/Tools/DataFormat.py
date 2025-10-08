import ROOT

class Particle(ROOT.TLorentzVector):
    def __init__(self, pt, eta, phi, mass):
        ROOT.TLorentzVector.__init__(self)
        self.SetPtEtaPhiM(pt, eta, phi, mass)
        self.charge = 0
        self.btagScore = 0.
        self.isMuon = False
        self.isElectron = False
        self.isJet = False

    def setCharge(self, charge):
        self.charge = charge

    def charge(self):
        return self.charge

    def btagScore(self):
        return self.btagScore

    def mT(self, part):
        dPhi = self.DeltaPhi(part)
        return ROOT.TMath.Sqrt(2*self.Pt()*part.Pt()*(1.-ROOT.TMath.Cos(dPhi)))

    def isMuon(self):
        return self.isMuon

    def isElectron(self):
        return self.isElectron

    def isJet(self):
        return self.isJet
    
class Lepton(Particle):
    def __init__(self, pt, eta, phi, mass):
        Particle.__init__(self, pt, eta, phi, mass)
        self.nearestJetFlavour = -999.
        self.lepType = ""
    
    def setLepType(self, lepType, nearestJetFlavour):
        if lepType in [1, 2, 6]:
            self.lepType = "prompt"
        elif abs(lepType) == 3:
            self.lepType = "fromTau"
        elif abs(lepType) in [4, 5] or lepType == -6:
            self.lepType = "conv"
        elif lepType < 0:
            if nearestJetFlavour == 4:
                self.lepType = "fromC"
            elif nearestJetFlavour == 5:
                self.lepType = "fromB"
            else:
                self.lepType = "fromL"
        else:
            self.lepType = "unknown"

    def lepType(self):
        return self.lepType
        

# Base class for electron / muon
class Electron(Lepton):
    def __init__(self, pt, eta, phi, mass):
        Lepton.__init__(self, pt, eta, phi, mass)
        self.isElectron = True
        self.region = ""
        if abs(eta) < 0.8:
            self.region = "InnerBarrel"
        elif abs(eta) < 1.479:
            self.region = "OuterBarrel"
        else:
            self.region = "Endcap"

        # Trigger emulation cuts
        self.sieie = -999.
        self.deltaEtaInSC = -999.
        self.deltaPhiInSeed = -999.
        self.hoe = -999.
        self.ecalPFClusterIso = -999.
        self.hcalPFClusterIso = -999.
        self.trackRelIso = -999.
        
        # ID cuts
        self.isMVANoIsoWP90 = False
        self.convVeto = False
        self.lostHits = -999.
        self.dz = -999.
        self.miniPFRelIso = -999.
        self.sip3d = -999.
        self.mvaNoIso = -999.

    def setTriggerVariables(self, sieie, deltaEtaInSC, deltaPhiInSeed, hoe, ecalPFClusterIso, hcalPFClusterIso, rho, trackRelIso):
        self.sieie = sieie
        self.deltaEtaInSC = deltaEtaInSC
        self.deltaPhiInSeed = deltaPhiInSeed
        self.hoe = hoe
        self.ecalPFClusterIso = ecalPFClusterIso
        self.hcalPFClusterIso = hcalPFClusterIso
        self.rho = rho
        self.trackRelIso = trackRelIso
        
    def setIDVariables(self, isMVANoIsoWP90, convVeto, lostHits, dz, miniPFRelIso, sip3d, mvaNoIso):
        self.isMVANoIsoWP90 = isMVANoIsoWP90
        self.convVeto = convVeto
        self.lostHits = lostHits
        self.dz = dz
        self.miniPFRelIso = miniPFRelIso
        self.sip3d = sip3d
        self.mvaNoIso = mvaNoIso

    def passTriggerCuts(self):
        if "Barrel" in self.region:
            if self.sieie() < 0.013: return False
            if self.deltaEtaInSC() < 0.01: return False
            if self.deltaPhiInSeed() < 0.07: return False
            if self.hoe() < 0.13: return False
            if self.ecalPFClusterIso() < 0.5: return False
            if self.hcalPFClusterIso() < 0.3: return False
            if self.trackRelIso() < 0.2: return False
        else:
            if self.sieie() < 0.035: return False
            if self.deltaEtaInSC() < 0.015: return False
            if self.deltaPhiInSeed() < 0.1: return False
            if self.hoe() < 0.13: return False
            if self.ecalPFClusterIso() < 0.5: return False
            if self.hcalPFClusterIso() < 0.3: return False
            if self.trackRelIso() < 0.2: return False
        
    def sieie(self):
        return self.sieie
    
    def deltaEtaInSC(self):
        return self.deltaEtaInSC
    
    def deltaPhiInSeed(self):
        return self.deltaPhiInSeed
    
    def hoe(self):
        return self.hoe
    
    def ecalPFClusterIso(self):
        effectiveArea = 0.16544 if "Barrel" in self.region else 0.13212
        return max(0., self.ecalPFClusterIso - self.rho*effectiveArea)/self.pt
    
    def hcalPFClusterIso(self):
        effectiveArea = 0.05956 if "Barrel" in self.region else 0.13052
        return max(0., self.hcalPFClusterIso - self.rho*effectiveArea)/self.pt
    
    def trackRelIso(self):
        return self.trackRelIso
    
    def isMVANoIsoWP90(self):
        return self.isMVANoIsoWP90
    
    def convVeto(self):
        return self.convVeto
    
    def lostHits(self):
        return self.lostHits
    
    def dz(self):
        return self.dz
    
    def miniPFRelIso(self):
        return self.miniPFRelIso

    def sip3d(self):
        return self.sip3d
    
    def mvaNoIso(self):
        return self.mvaNoIso
    
class Muon(Lepton):
    def __init__(self, pt, eta, phi, mass):
        Lepton.__init__(self, pt, eta, phi, mass)
        self.isMuon = True

        # Trigger emulation cuts
        self.isPOGMediumId = False
        self.miniPFRelIso = -999.
        self.trackRelIso = -999.
        self.sip3d = -999.
        self.dz = -999.

    def setVariables(self, isPOGMediumId, miniPFRelIso, trackRelIso, sip3d, dz):
        self.isPOGMediumId = isPOGMediumId
        self.miniPFRelIso = miniPFRelIso
        self.trackRelIso = trackRelIso
        self.sip3d = sip3d
        self.dz = dz

    def PassTriggerCuts(self):
        if self.trackRelIso() < 0.4: return False
        return True
        
    def IsPOGMediumId(self):
        return self.isPOGMediumId

    def MiniPFRelIso(self):
        return self.miniPFRelIso
    
    def TrackRelIso(self):
        return self.trackRelIso
    
    def Sip3d(self):
        return self.sip3d
    
    def DZ(self):
        return self.dz
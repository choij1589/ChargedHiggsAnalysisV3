from ROOT import TLorentzVector
from ROOT import TMath

# Base class for all objects
class Particle(TLorentzVector):
    def __init__(self, pt, eta, phi, mass):
        TLorentzVector.__init__(self)
        self.SetPtEtaPhiM(pt, eta, phi, mass)
        self.charge = 0
        self.btagScore = 0.
        self.isMuon = False
        self.isElectron = False
        self.isJet = False
        self.isBjet = False

    def Charge(self):
        return self.charge

    def BtagScore(self):
        return self.btagScore

    def MT(self, part):
        dPhi = self.DeltaPhi(part)
        return TMath.Sqrt(2*self.Pt()*part.Pt()*(1.-TMath.Cos(dPhi)))

    def IsMuon(self):
        return self.isMuon

    def IsElectron(self):
        return self.isElectron

    def IsJet(self):
        return self.isJet

    def IsBjet(self):
        return self.isBjet

# Base class for electron / muon
class Lepton(Particle):
    def __init__(self, pt, eta, phi, mass):
        Particle.__init__(self, pt, eta, phi, mass)

    def SetCharge(self, charge):
        self.charge = charge


class Muon(Lepton):
    def __init__(self, pt, eta, phi, mass):
        Lepton.__init__(self, pt, eta, phi, mass)
        self.isMuon = True


class Electron(Lepton):
    def __init__(self, pt, eta, phi, mass):
        Lepton.__init__(self, pt, eta, phi, mass)
        self.isElectron = True


class Jet(Particle):
    def __init__(self, pt, eta, phi, mass):
        Particle.__init__(self, pt, eta, phi, mass)
        self.isJet = True
        self.isBtagged = False

    def SetCharge(self, charge):
        self.charge = charge

    def SetBtagScore(self, btagScore):
        self.btagScore = btagScore

    def SetBtagging(self, isBtagged):
        self.isBtagged = isBtagged

    def IsBtagged(self):
        return self.isBtagged


class Bjet(Particle):
    def __init__(self, pt, eta, phi, mass):
        Particle.__init__(self, pt, eta, phi, mass)
        self.isJet = True
        self.isBjet = True
        self.isBtagged = True

    def SetCharge(self, charge):
        self.charge = charge

    def SetBtagScore(self, btagScore):
        self.btagScore = btagScore

    def SetBtagging(self, isBtagged):
        self.isBtagged = isBtagged

    def IsBtagged(self):
        return self.isBtagged

def getAllMuons(evt):
    """Get all muons with ptCorr applied (no cap).
    ptCorr = pT * (1 + max(0, miniIso - 0.1))
    Tight: ptCorr ≈ pT (miniIso < 0.1 → correction ≈ 0)
    Non-tight: ptCorr > pT (larger miniIso)
    Note: the 50 GeV Run3 muon cap is ONLY for FR table lookup,
    NOT for node features. Node features use uncapped ptCorr.
    Returns: (muons_list, is_tight_list)
    """
    muons = []
    is_tight = []
    muons_zip = zip(evt.MuonPtColl,
                    evt.MuonEtaColl,
                    evt.MuonPhiColl,
                    evt.MuonMassColl,
                    evt.MuonChargeColl,
                    evt.MuonIsTightColl,
                    evt.MuonMiniIsoColl)
    for pt, eta, phi, mass, charge, tight, miniIso in muons_zip:
        ptCorr = pt * (1.0 + max(0.0, miniIso - 0.1))
        thisMuon = Muon(ptCorr, eta, phi, mass)
        thisMuon.SetCharge(charge)
        muons.append(thisMuon)
        is_tight.append(bool(tight))
    return muons, is_tight


def getAllElectrons(evt):
    """Get all electrons with ptCorr applied (no cap).
    Same formula as muons.
    Returns: (electrons_list, is_tight_list)
    """
    electrons = []
    is_tight = []
    electrons_zip = zip(evt.ElectronPtColl,
                        evt.ElectronEtaColl,
                        evt.ElectronPhiColl,
                        evt.ElectronMassColl,
                        evt.ElectronChargeColl,
                        evt.ElectronIsTightColl,
                        evt.ElectronMiniIsoColl)
    for pt, eta, phi, mass, charge, tight, miniIso in electrons_zip:
        ptCorr = pt * (1.0 + max(0.0, miniIso - 0.1))
        thisElectron = Electron(ptCorr, eta, phi, mass)
        thisElectron.SetCharge(charge)
        electrons.append(thisElectron)
        is_tight.append(bool(tight))
    return electrons, is_tight


def getMuons(evt):
    muons = []
    muons_zip = zip(evt.MuonPtColl,
                    evt.MuonEtaColl,
                    evt.MuonPhiColl,
                    evt.MuonMassColl,
                    evt.MuonChargeColl,
                    evt.MuonIsTightColl)
    for pt, eta, phi, mass, charge, isTight in muons_zip:
        if not isTight:
            continue
        thisMuon = Muon(pt, eta, phi, mass)
        thisMuon.SetCharge(charge)
        muons.append(thisMuon)
    return muons

def getElectrons(evt):
    electrons = []
    electrons_zip = zip(evt.ElectronPtColl,
                        evt.ElectronEtaColl,
                        evt.ElectronPhiColl,
                        evt.ElectronMassColl,
                        evt.ElectronChargeColl,
                        evt.ElectronIsTightColl)
    for pt, eta, phi, mass, charge, isTight in electrons_zip:
        if not isTight:
            continue
        thisElectron = Electron(pt, eta, phi, mass)
        thisElectron.SetCharge(charge)
        electrons.append(thisElectron)
    return electrons

def getJets(evt):
    """
    Get jets and b-jets from event.

    ParticleNetMD always uses separate b-jets (no btagScore in node features).
    B-tagged jets are returned as separate Bjet objects.

    Returns:
        jets: List of non-b-tagged Jet objects
        bjets: List of b-tagged Bjet objects
    """
    jets = []
    bjets = []
    jets_zip = zip(evt.JetPtColl,
                   evt.JetEtaColl,
                   evt.JetPhiColl,
                   evt.JetMassColl,
                   evt.JetChargeColl,
                   evt.JetBtagScoreColl,
                   evt.JetIsBtaggedColl)

    for pt, eta, phi, mass, charge, btagScore, isBtagged in jets_zip:
        if isBtagged:
            # Create Bjet object: isJet=True, isBjet=True
            thisBjet = Bjet(pt, eta, phi, mass)
            thisBjet.SetCharge(charge)
            thisBjet.SetBtagScore(btagScore)
            thisBjet.SetBtagging(isBtagged)
            bjets.append(thisBjet)
        else:
            # Create regular Jet object: isJet=True, isBjet=False
            thisJet = Jet(pt, eta, phi, mass)
            thisJet.SetCharge(charge)
            thisJet.SetBtagScore(btagScore)
            thisJet.SetBtagging(isBtagged)
            jets.append(thisJet)

    return jets, bjets

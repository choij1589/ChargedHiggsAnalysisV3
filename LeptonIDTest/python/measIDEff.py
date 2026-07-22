#!/usr/bin/env python
import os
import argparse
import array
import ROOT

ROOT.gROOT.SetBatch(True)

parser = argparse.ArgumentParser()
parser.add_argument("--era", type=str, required=True,
                    choices=["2016preVFP", "2016postVFP", "2017", "2018",
                             "2022", "2022EE", "2023", "2023BPix"],
                    help="Era")
parser.add_argument("--object", type=str, required=True,
                    choices=["muon", "electron"], help="Object type")
parser.add_argument("--reduction", default=1, type=int,
                    help="Reduce the number of events in the TChain (for quick tests)")
args = parser.parse_args()
WORKDIR = os.getenv("WORKDIR")
if WORKDIR is None:
    raise RuntimeError("WORKDIR is not set. Did you `source setup.sh`?")

if args.era in ["2016preVFP", "2016postVFP", "2017", "2018"]:
    RUN = 2
else:
    RUN = 3

# eta x pT binning (matches MeasTrigEff ID efficiency histograms)
if args.object == "muon":
    parse_dir = "ParseMuIDVariables"
    eta_bins = [0., 0.9, 1.2, 2.1, 2.4]            # |eta|
    pt_bins = [10., 15., 20., 25., 30., 40., 50., 60., 120., 200.]
else:  # electron
    parse_dir = "ParseEleIDVariables"
    eta_bins = [-2.5, -2.0, -1.566, -1.444, -0.8, 0., 0.8, 1.444, 1.566, 2.0, 2.5]  # signed scEta
    pt_bins = [15., 20., 35., 50., 100., 200., 500.]

eta_edges = array.array('d', eta_bins)
pt_edges = array.array('d', pt_bins)

# MC truth sources (granular; 'unknown' stored for closure but not plotted)
sources = ["prompt", "conv", "fromTau", "fromB", "fromC", "fromL", "fromPU", "unknown"]
working_points = ["loose", "tight"]


def classify_lepton(lepType, jetFlavour):
    if lepType in [1, 2, 6]:
        return "prompt"
    elif lepType == 3:
        return "fromTau"
    elif lepType in [4, 5, -5, -6]:
        return "conv"
    elif lepType < 0:
        if jetFlavour == -1:
            return "fromPU"
        elif jetFlavour == 0:
            return "fromL"
        elif jetFlavour == 4:
            return "fromC"
        elif jetFlavour == 5:
            return "fromB"
        else:
            return "unknown"
    else:
        return "unknown"


def check_region_electron(scEta):
    if abs(scEta) < 0.8:
        return "InnerBarrel"
    elif abs(scEta) < 1.479:
        return "OuterBarrel"
    else:
        return "Endcap"


def get_electron_cuts(region):
    """Trigger-emulation + loose thresholds, transcribed from fill_electrons.py:get_cuts."""
    if "Barrel" in region:
        c_sieie = 0.013
        c_dEta, c_dPhi = 0.01, 0.07
        c_hoe = 0.13
        ecalEA, hcalEA = 0.16544, 0.05956
    else:  # Endcap
        c_sieie = 0.035
        c_dEta, c_dPhi = 0.015, 0.1
        c_hoe = 0.13
        ecalEA, hcalEA = 0.13212, 0.13052

    if RUN == 2:
        c_sip3d = 8.
        c_miniIso = 0.4
        c_mva = {"InnerBarrel": 0.985, "OuterBarrel": 0.96, "Endcap": 0.85}[region]
    else:
        c_sip3d = 6.
        c_miniIso = 0.4
        c_mva = {"InnerBarrel": 0.8, "OuterBarrel": 0.5, "Endcap": -0.8}[region]

    return c_sieie, c_dEta, c_dPhi, c_hoe, ecalEA, hcalEA, c_miniIso, c_sip3d, c_mva


def pass_muon_wp(evt, i):
    """Return (pass_loose, pass_tight) for muon probe i. Denominator = all reco probes."""
    if not evt.isPOGMediumId[i]:
        return False, False
    if not evt.tkRelIso[i] < 0.4:
        return False, False
    if not abs(evt.dZ[i]) < 0.1:
        return False, False

    if RUN == 2:
        c_sip3d, c_miniIso = 5., 0.6
    else:
        c_sip3d, c_miniIso = 8., 0.4

    if not (evt.sip3d[i] < c_sip3d and evt.miniPFRelIso[i] < c_miniIso):
        return False, False
    pass_loose = True
    pass_tight = (evt.sip3d[i] < 3. and evt.miniPFRelIso[i] < 0.1)
    return pass_loose, pass_tight


def pass_electron_wp(evt, i):
    """Return (pass_loose, pass_tight) for electron probe i. Denominator = all reco probes."""
    region = check_region_electron(evt.scEta[i])
    c_sieie, c_dEta, c_dPhi, c_hoe, ecalEA, hcalEA, c_miniIso, c_sip3d, c_mva = get_electron_cuts(region)

    # Trigger-emulation cuts
    if not evt.sieie[i] < c_sieie:
        return False, False
    if not abs(evt.deltaEtaInSC[i]) < c_dEta:
        return False, False
    if not abs(evt.deltaPhiInSeed[i]) < c_dPhi:
        return False, False
    if not evt.hoe[i] < c_hoe:
        return False, False
    ecalPFClusterIso = max(0., evt.ecalPFClusterIso[i] - evt.rho[i] * ecalEA) / evt.pt[i]
    hcalPFClusterIso = max(0., evt.hcalPFClusterIso[i] - evt.rho[i] * hcalEA) / evt.pt[i]
    trackIso = evt.dr03TkSumPt[i] / evt.pt[i]
    if not ecalPFClusterIso < 0.5:
        return False, False
    if not hcalPFClusterIso < 0.3:
        return False, False
    if not trackIso < 0.2:
        return False, False

    # Baseline
    if not evt.convVeto[i]:
        return False, False
    if not evt.lostHits[i] < 2:
        return False, False
    if not abs(evt.dZ[i]) < 0.1:
        return False, False

    # Loose ID
    if not (evt.isMVANoIsoWP90[i] or (evt.mvaNoIso[i] > c_mva)):
        return False, False
    if not evt.sip3d[i] < c_sip3d:
        return False, False
    if not evt.miniPFRelIso[i] < c_miniIso:
        return False, False
    pass_loose = True

    # Tight ID
    pass_tight = (evt.isMVANoIsoWP90[i] and evt.sip3d[i] < 4. and evt.miniPFRelIso[i] < 0.1)
    return pass_loose, pass_tight


# Input chain
tree = ROOT.TChain("Events")
for sample in ["TTLJ_powheg", "TTLL_powheg"]:
    tree.Add(f"{WORKDIR}/SKNanoOutput/{parse_dir}/{args.era}/{sample}.root")
n_total = tree.GetEntries()
if n_total == 0:
    raise RuntimeError(f"No entries found for {parse_dir}/{args.era}. Check inputs.")
maxevt = int(n_total / args.reduction)

# Book histograms
denom = {}
num = {}
for src in sources:
    denom[src] = ROOT.TH2D(f"denom_{src}", "", len(eta_bins) - 1, eta_edges,
                           len(pt_bins) - 1, pt_edges)
    denom[src].Sumw2()
    for wp in working_points:
        h = ROOT.TH2D(f"num_{src}_{wp}", "", len(eta_bins) - 1, eta_edges,
                      len(pt_bins) - 1, pt_edges)
        h.Sumw2()
        num[(src, wp)] = h

print(f"Measuring {args.object} ID efficiency for {args.era}: {maxevt}/{n_total} events")

for ievt, evt in enumerate(tree):
    if ievt > maxevt:
        break
    if ievt % 100000 == 0:
        print(f"Processing event {ievt}/{maxevt}")

    genWeight = evt.genWeight
    if args.object == "muon":
        nLeptons = evt.nMuons
    else:
        nLeptons = evt.nElectrons

    for i in range(nLeptons):
        src = classify_lepton(evt.lepType[i], evt.nearestJetFlavour[i])
        if args.object == "muon":
            eta_val = abs(evt.eta[i])
            pass_loose, pass_tight = pass_muon_wp(evt, i)
        else:
            eta_val = evt.scEta[i]
            pass_loose, pass_tight = pass_electron_wp(evt, i)
        pt_val = evt.pt[i]

        denom[src].Fill(eta_val, pt_val, genWeight)
        if pass_loose:
            num[(src, "loose")].Fill(eta_val, pt_val, genWeight)
        if pass_tight:
            num[(src, "tight")].Fill(eta_val, pt_val, genWeight)

# Efficiencies and output
outpath = f"{WORKDIR}/LeptonIDTest/results/{args.era}/idEff_{args.object}.root"
os.makedirs(os.path.dirname(outpath), exist_ok=True)
out = ROOT.TFile(outpath, "RECREATE")
for src in sources:
    denom[src].Write()
    for wp in working_points:
        num[(src, wp)].Write()
        eff = num[(src, wp)].Clone(f"eff_{src}_{wp}")
        eff.Divide(num[(src, wp)], denom[src], 1., 1., "B")
        eff.Write()
out.Close()
print(f"Wrote {outpath}")

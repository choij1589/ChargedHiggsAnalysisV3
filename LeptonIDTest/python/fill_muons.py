#!/usr/bin/env python
import os
import argparse
import ROOT
from itertools import product

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(1111)
ROOT.gStyle.SetOptFit(1111)
ROOT.gStyle.SetPadGridX(True)
ROOT.gStyle.SetPadGridY(True)

parser = argparse.ArgumentParser()
parser.add_argument("--era", type=str, required=True, help="Era")
parser.add_argument("--reduction", default=1, type=int, help="reduce the number of events in TChain")
args = parser.parse_args()
WORKDIR = os.getenv("WORKDIR")

if args.era in ["2016preVFP", "2016postVFP", "2017", "2018"]:
    RUN = 2
elif args.era in ["2022", "2022EE", "2023", "2023BPix"]:
    RUN = 3
else:
    raise ValueError(f"Invalid era: {args.era}")

# Create a TChain instead of opening a single file
tree = ROOT.TChain("Events")
#tree.Add(f"{WORKDIR}/SKNanoOutput/ParseMuIDVariables/{args.era}/WJets_MG.root")
tree.Add(f"{WORKDIR}/SKNanoOutput/ParseMuIDVariables/{args.era}/TTLJ_powheg.root")
tree.Add(f"{WORKDIR}/SKNanoOutput/ParseMuIDVariables/{args.era}/TTLL_powheg.root")
maxevt = int(tree.GetEntries() / args.reduction)

out = ROOT.TFile(f"{WORKDIR}/LeptonIDTest/histograms/muon.{args.era}.root", "RECREATE")

regions = ["InnerBarrel", "OuterBarrel", "Endcap"]
leptonTypes = ["prompt", "conv", "fromTau", "fromL", "fromC", "fromB", "fromPU", "unknown"]
ptcorr_bins = [10., 15., 20., 30., 50., 70.]

def check_region(eta):
    if abs(eta) < 0.9:
        return "InnerBarrel"
    elif abs(eta) < 1.6:
        return "OuterBarrel"
    else:
        return "Endcap"

def classify_lepton(lepType, jetFlavour):
    # prompt
    if lepType in [1, 2, 6]:
        return "prompt"
    # from tau
    elif lepType == 3:
        return "fromTau"
    # conv
    elif lepType in [4, 5, -5, -6]:
        return "conv"
    # nonprompt
    elif lepType < 0:
        if jetFlavour == -1:
            return "fromPU"
        elif jetFlavour == 0:
            return "fromL"
        if jetFlavour == 4:
            return "fromC"
        elif jetFlavour == 5:
            return "fromB"
        else:
            raise ValueError(f"Invalid jetFlavour: {jetFlavour}")
    else:
        return "unknown"

def get_pt_bin_name(pt_corr):
    for i in range(len(ptcorr_bins)-1):
        if ptcorr_bins[i] <= pt_corr < ptcorr_bins[i+1]:
            return f"pt{int(ptcorr_bins[i])}to{int(ptcorr_bins[i+1])}"
    if pt_corr >= ptcorr_bins[-1]:
        return f"pt{int(ptcorr_bins[-1])}toInf"
    return None


# Register histograms and directories
histograms = {}
directories = {}

## Create main directories for trigger emulation cuts
out.cd()  # Make sure we're in the main directory
# Trigger emulation cuts
for region in regions:
    directories[region] = out.mkdir(region)
    directories[region].cd()
    histograms[f"{region}/trackIso"] = ROOT.TH1F("trackIso", "", 100, 0., 1.)

# Create directories for each region and lepton type
out.cd()
for region, lType in product(regions, leptonTypes):
    directories[f"{region}/{lType}"] = directories[region].mkdir(lType)
    directories[f"{region}/{lType}"].cd()
    
    histograms[f"{region}/{lType}/isPOGMediumId"] = ROOT.TH1F("isPOGM", "", 2, 0., 2.)
    histograms[f"{region}/{lType}/sip3d"] = ROOT.TH1F("sip3d", "", 100, 0., 20.)
    histograms[f"{region}/{lType}/dz"] = ROOT.TH1F("dz", "", 100, 0., 1.)
    histograms[f"{region}/{lType}/miniIso"] = ROOT.TH1F("miniIso", "", 100, 0., 1.)
    histograms[f"{region}/{lType}/miniIso_sip3d"] = ROOT.TH2F("miniIso_sip3d", "", 100, 0., 1., 100, 0., 10.)
    histograms[f"{region}/{lType}/trackRelIso"] = ROOT.TH1F("trackRelIso", "", 100, 0., 1.)

    histograms[f"{region}/{lType}/passTrigCuts"] = ROOT.TH1F(f"passTrigCuts", "", 100, 0., 100.)
    histograms[f"{region}/{lType}/passTightID"] = ROOT.TH1F(f"passTightID", "", 100, 0., 100.)
    histograms[f"{region}/{lType}/passLooseID"] = ROOT.TH1F(f"passLooseID", "", 100, 0., 100.)
    
    ## Pt-binned histograms for SIP3D and miniIso
    for i in range(len(ptcorr_bins)-1):
        pt_bin = f"pt{int(ptcorr_bins[i])}to{int(ptcorr_bins[i+1])}"
        histograms[f"{region}/{lType}/sip3d_{pt_bin}"] = ROOT.TH1F(f"sip3d_{pt_bin}", "", 100, 0., 20.)
        histograms[f"{region}/{lType}/miniIso_{pt_bin}"] = ROOT.TH1F(f"miniIso_{pt_bin}", "", 100, 0., 1.)
    
    # Additional bin for high pt
    pt_bin = f"pt{int(ptcorr_bins[-1])}toInf"
    histograms[f"{region}/{lType}/sip3d_{pt_bin}"] = ROOT.TH1F(f"sip3d_{pt_bin}", "", 100, 0., 20.)
    histograms[f"{region}/{lType}/miniIso_{pt_bin}"] = ROOT.TH1F(f"miniIso_{pt_bin}", "", 100, 0., 1.)
    

## Loop over events
print("Starting event loop...")
for i, evt in enumerate(tree):
    if i > maxevt: break
    if i % 100000 == 0:
        print(f"Processing event {i}/{maxevt}")

    genWeight = evt.genWeight
    nMuons = evt.nMuons

    for i in range(nMuons):
        if not evt.isIsoMuTrigMatched[i]: continue
        region = check_region(evt.eta[i])

        histograms[f"{region}/trackIso"].Fill(evt.tkRelIso[i], genWeight)
        if not evt.tkRelIso[i] < 0.4: continue

        lType = classify_lepton(evt.lepType[i], evt.nearestJetFlavour[i])
        histograms[f"{region}/{lType}/isPOGMediumId"].Fill(evt.isPOGMediumId[i], genWeight)
        histograms[f"{region}/{lType}/sip3d"].Fill(evt.sip3d[i], genWeight)
        histograms[f"{region}/{lType}/dz"].Fill(abs(evt.dZ[i]), genWeight)
        histograms[f"{region}/{lType}/miniIso"].Fill(evt.miniPFRelIso[i], genWeight)

        # For efficiency study
        histograms[f"{region}/{lType}/miniIso_sip3d"].Fill(evt.miniPFRelIso[i], evt.sip3d[i], genWeight)

        # For fake rate study
        pt_corr = evt.pt[i] * (1.+max(0., evt.miniPFRelIso[i] - 0.1))
        histograms[f"{region}/{lType}/passTrigCuts"].Fill(pt_corr, genWeight)
        
        ## Fill pt-binned histograms for SIP3D and miniIso
        pt_bin = get_pt_bin_name(pt_corr)
        if pt_bin:
            histograms[f"{region}/{lType}/sip3d_{pt_bin}"].Fill(evt.sip3d[i], genWeight)
            histograms[f"{region}/{lType}/miniIso_{pt_bin}"].Fill(evt.miniPFRelIso[i], genWeight)

        # Baseline ID
        if not evt.isPOGMediumId[i]: continue
        if not evt.tkRelIso[i] < 0.4: continue
        if not abs(evt.dZ[i]) < 0.1: continue

        if RUN == 2:
            c_miniIso = 0.6
            c_sip3d = 5.
        else:
            c_miniIso = 0.4
            c_sip3d = 8.

        # Check current ID
        if not (evt.sip3d[i] < c_sip3d): continue
        if not (evt.miniPFRelIso[i] < c_miniIso): continue
        histograms[f"{region}/{lType}/passLooseID"].Fill(pt_corr, genWeight)

        if not (evt.sip3d[i] < 3.): continue
        if not (evt.miniPFRelIso[i] < 0.1): continue
        histograms[f"{region}/{lType}/passTightID"].Fill(pt_corr, genWeight)

out.Write()
out.Close()
print(f"Finished filling histograms for {args.era}")

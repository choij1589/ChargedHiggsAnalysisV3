#!/usr/bin/env python
import os
import argparse
import ROOT
from itertools import product

# Enable ROOT batch mode to avoid GUI windows
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
#tree.Add(f"{WORKDIR}/SKNanoOutput/ParseEleIDVariables/{args.era}/WJets_MG.root")
tree.Add(f"{WORKDIR}/SKNanoOutput/ParseEleIDVariables/{args.era}/TTLJ_powheg.root")
tree.Add(f"{WORKDIR}/SKNanoOutput/ParseEleIDVariables/{args.era}/TTLL_powheg.root")
maxevt = int(tree.GetEntries() / args.reduction)

out = ROOT.TFile(f"{WORKDIR}/LeptonIDTest/histograms/electron.{args.era}.root", "RECREATE")

regions = ["InnerBarrel", "OuterBarrel", "Endcap"]
leptonTypes = ["prompt", "conv", "fromTau", "fromL", "fromC", "fromB", "fromPU", "unknown"]
ptcorr_bins = [10., 15., 20., 25., 35., 50., 70.]

def check_region(scEta):
    if abs(scEta) < 0.8:
        return "InnerBarrel"
    elif abs(scEta) < 1.479:
        return "OuterBarrel"
    else:
        return "Endcap"

def get_cuts(region):
    if "Barrel" in region:
        c_sieie = 0.013
        c_dEta, c_dPhi = 0.01, 0.07
        c_hoe = 0.13
        ecalEA, hcalEA = 0.16544, 0.05956
    else:
        c_sieie = 0.035
        c_dEta, c_dPhi = 0.015, 0.1
        c_hoe = 0.13
        ecalEA, hcalEA = 0.13212, 0.13052
    
    if RUN == 2:
        c_sip3d = 8.
        c_miniIso = 0.4
        if region == "InnerBarrel":
            c_mva = 0.985
        elif region == "OuterBarrel":
            c_mva = 0.96
        else:
            c_mva = 0.85
    else:
        c_sip3d = 6.
        c_miniIso = 0.4
        if region == "InnerBarrel":
            c_mva = 0.8
        elif region == "OuterBarrel":
            c_mva = 0.5
        else:
            c_mva = -0.8
    
    return c_sieie, c_dEta, c_dPhi, c_hoe, ecalEA, hcalEA, c_miniIso, c_sip3d, c_mva

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
        elif jetFlavour == 4:
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
    histograms[f"{region}/sieie"] = ROOT.TH1F(f"sieie", "", 200, 0., 0.2)
    histograms[f"{region}/deltaEtaInSC"] = ROOT.TH1F(f"deltaEtaInSC", "", 100, 0., 1.0)
    histograms[f"{region}/deltaPhiInSeed"] = ROOT.TH1F(f"deltaPhiInSeed", "", 100, 0., 1.0)
    histograms[f"{region}/hoe"] = ROOT.TH1F(f"hoe", "", 100, 0., 1.)
    histograms[f"{region}/ecalPFClusterIso"] = ROOT.TH1F(f"ecalPFClusterIso", "", 100, 0., 1.)
    histograms[f"{region}/hcalPFClusterIso"] = ROOT.TH1F(f"hcalPFClusterIso", "", 100, 0., 1.)
    histograms[f"{region}/trackIso"] = ROOT.TH1F(f"trackIso", "", 100, 0., 1.)

## Create directories for each region and lepton type
out.cd()  # Go back to main directory
for region, lType in product(regions, leptonTypes):
        directories[f"{region}/{lType}"] = directories[region].mkdir(lType)
        directories[f"{region}/{lType}"].cd()
        
        histograms[f"{region}/{lType}/isMVANoIsoWP90"] = ROOT.TH1F("isMVANoIsoWP90", "", 2, 0., 2.)
        histograms[f"{region}/{lType}/convVeto"] = ROOT.TH1F("convVeto", "", 2, 0., 2.)
        histograms[f"{region}/{lType}/lostHits"] = ROOT.TH1F("lostHits", "", 10, 0., 10.)
        histograms[f"{region}/{lType}/sip3d"] = ROOT.TH1F("sip3d", "", 100, 0., 20.)
        histograms[f"{region}/{lType}/mvaNoIso"] = ROOT.TH1F("mvaNoIso", "", 200, -1., 1.)
        histograms[f"{region}/{lType}/dz"] = ROOT.TH1F("dz", "", 100, 0., 1.)
        histograms[f"{region}/{lType}/miniIso"] = ROOT.TH1F("miniIso", "", 100, 0., 1.)
        
        ## For efficiency study
        histograms[f"{region}/{lType}/miniIso_sip3d_mvaNoIso"] = ROOT.TH3F("miniIso_sip3d_mvaNoIso", "", 100, 0., 1., 100, 0., 10., 2000, -1., 1.)
        

        ## Check current ID
        histograms[f"{region}/{lType}/passTrigCuts"] = ROOT.TH1F(f"passTrigCuts", "", 100, 0., 100.)
        histograms[f"{region}/{lType}/passTightID"] = ROOT.TH1F(f"passTightID", "", 100, 0., 100.)
        histograms[f"{region}/{lType}/passLooseID"] = ROOT.TH1F(f"passLooseID", "", 100, 0., 100.)
        
        ## Pt-binned histograms for mvaNoIso, SIP3D, and miniIso
        for i in range(len(ptcorr_bins)-1):
            pt_bin = f"pt{int(ptcorr_bins[i])}to{int(ptcorr_bins[i+1])}"
            histograms[f"{region}/{lType}/mvaNoIso_{pt_bin}"] = ROOT.TH1F(f"mvaNoIso_{pt_bin}", "", 200, -1., 1.)
            histograms[f"{region}/{lType}/sip3d_{pt_bin}"] = ROOT.TH1F(f"sip3d_{pt_bin}", "", 100, 0., 20.)
            histograms[f"{region}/{lType}/miniIso_{pt_bin}"] = ROOT.TH1F(f"miniIso_{pt_bin}", "", 100, 0., 1.)
        
        # Additional bin for high pt
        pt_bin = f"pt{int(ptcorr_bins[-1])}toInf"
        histograms[f"{region}/{lType}/mvaNoIso_{pt_bin}"] = ROOT.TH1F(f"mvaNoIso_{pt_bin}", "", 200, -1., 1.)
        histograms[f"{region}/{lType}/sip3d_{pt_bin}"] = ROOT.TH1F(f"sip3d_{pt_bin}", "", 100, 0., 20.)
        histograms[f"{region}/{lType}/miniIso_{pt_bin}"] = ROOT.TH1F(f"miniIso_{pt_bin}", "", 100, 0., 1.)

## Loop over events
print("Starting event loop...")
for i, evt in enumerate(tree):
    if i > maxevt: break
    if i % 100000 == 0:
        print(f"Processing event {i}/{maxevt}")

    genWeight = evt.genWeight
    nElectrons = evt.nElectrons

    for i in range(nElectrons):
        if not evt.isIsoElTrigMatched[i]: continue
        region = check_region(evt.scEta[i])
        c_sieie, c_dEta, c_dPhi, c_hoe, ecalEA, hcalEA, c_miniIso, c_sip3d, c_mva = get_cuts(region)
        
        histograms[f"{region}/sieie"].Fill(evt.sieie[i], genWeight)   # 0.013 / 0.035
        histograms[f"{region}/deltaEtaInSC"].Fill(abs(evt.deltaEtaInSC[i]), genWeight)   # 0.01 / 0.015
        histograms[f"{region}/deltaPhiInSeed"].Fill(abs(evt.deltaPhiInSeed[i]), genWeight)   # 0.07 / 0.1
        histograms[f"{region}/hoe"].Fill(evt.hoe[i], genWeight)   # 0.13
        ecalPFClusterIso = max(0., evt.ecalPFClusterIso[i] - evt.rho[i]*ecalEA)/evt.pt[i] # 0.5
        hcalPFClusterIso = max(0., evt.hcalPFClusterIso[i] - evt.rho[i]*hcalEA)/evt.pt[i] # 0.3
        trackIso = evt.dr03TkSumPt[i]/evt.pt[i] # 0.2
        histograms[f"{region}/ecalPFClusterIso"].Fill(ecalPFClusterIso, genWeight)
        histograms[f"{region}/hcalPFClusterIso"].Fill(hcalPFClusterIso, genWeight)
        histograms[f"{region}/trackIso"].Fill(trackIso, genWeight)
        
        # apply trigger emulation cuts
        if not evt.sieie[i] < c_sieie: continue
        if not abs(evt.deltaEtaInSC[i]) < c_dEta: continue
        if not abs(evt.deltaPhiInSeed[i]) < c_dPhi: continue
        if not evt.hoe[i] < c_hoe: continue
        if not ecalPFClusterIso < 0.5: continue
        if not hcalPFClusterIso < 0.3: continue
        if not trackIso < 0.2: continue

        lType = classify_lepton(evt.lepType[i], evt.nearestJetFlavour[i])
        histograms[f"{region}/{lType}/isMVANoIsoWP90"].Fill(int(evt.isMVANoIsoWP90[i]), genWeight)
        histograms[f"{region}/{lType}/convVeto"].Fill(int(evt.convVeto[i]), genWeight)
        histograms[f"{region}/{lType}/lostHits"].Fill(evt.lostHits[i], genWeight)
        histograms[f"{region}/{lType}/sip3d"].Fill(evt.sip3d[i], genWeight)
        histograms[f"{region}/{lType}/mvaNoIso"].Fill(evt.mvaNoIso[i], genWeight)
        histograms[f"{region}/{lType}/dz"].Fill(abs(evt.dZ[i]), genWeight)
        histograms[f"{region}/{lType}/miniIso"].Fill(evt.miniPFRelIso[i], genWeight)

        ## For efficiency study
        histograms[f"{region}/{lType}/miniIso_sip3d_mvaNoIso"].Fill(evt.miniPFRelIso[i], evt.sip3d[i], evt.mvaNoIso[i], genWeight)
        
        ## For fake rate study
        pt_corr = evt.pt[i] * (1.+max(0., evt.miniPFRelIso[i] - 0.1))
        histograms[f"{region}/{lType}/passTrigCuts"].Fill(pt_corr, genWeight)
        
        ## Fill pt-binned histograms for mvaNoIso, SIP3D, and miniIso
        pt_bin = get_pt_bin_name(pt_corr)
        if pt_bin:
            histograms[f"{region}/{lType}/mvaNoIso_{pt_bin}"].Fill(evt.mvaNoIso[i], genWeight)
            histograms[f"{region}/{lType}/sip3d_{pt_bin}"].Fill(evt.sip3d[i], genWeight)
            histograms[f"{region}/{lType}/miniIso_{pt_bin}"].Fill(evt.miniPFRelIso[i], genWeight)

        # Baseline ID
        if not evt.convVeto[i]: continue
        if not evt.lostHits[i] < 2: continue
        if not abs(evt.dZ[i]) < 0.1: continue

        # Loose ID
        if not (evt.isMVANoIsoWP90[i] or (evt.mvaNoIso[i] > c_mva)): continue
        if not (evt.sip3d[i] < c_sip3d): continue
        if not (evt.miniPFRelIso[i] < c_miniIso): continue
        
        histograms[f"{region}/{lType}/passLooseID"].Fill(pt_corr, genWeight)
    
        if not (evt.isMVANoIsoWP90[i]): continue
        if not (evt.sip3d[i] < 4.): continue
        if not (evt.miniPFRelIso[i] < 0.1): continue
        histograms[f"{region}/{lType}/passTightID"].Fill(pt_corr, genWeight)
        
# Save the histograms - they will be automatically saved in their respective directories
out.Write()
out.Close()
print(f"Finished filling histograms for {args.era}!")

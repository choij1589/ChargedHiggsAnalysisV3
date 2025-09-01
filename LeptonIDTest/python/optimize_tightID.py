#!/usr/bin/env python
import os
import argparse
import json
import csv
import ROOT
import numpy as np
from itertools import product

ROOT.gROOT.SetBatch(True)

def check_region_muon(eta):
    if abs(eta) < 0.9:
        return "InnerBarrel"
    elif abs(eta) < 1.6:
        return "OuterBarrel"
    else:
        return "Endcap"

def check_region_electron(scEta):
    if abs(scEta) < 0.8:
        return "InnerBarrel"
    elif abs(scEta) < 1.479:
        return "OuterBarrel"
    else:
        return "Endcap"

def apply_electron_trigger_cuts(evt, idx, region):
    """Apply electron trigger emulation cuts"""
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
    
    # Apply cuts
    if not evt.sieie[idx] < c_sieie: return False
    if not abs(evt.deltaEtaInSC[idx]) < c_dEta: return False
    if not abs(evt.deltaPhiInSeed[idx]) < c_dPhi: return False
    if not evt.hoe[idx] < c_hoe: return False
    
    ecalPFClusterIso = max(0., evt.ecalPFClusterIso[idx] - evt.rho[idx]*ecalEA)/evt.pt[idx]
    hcalPFClusterIso = max(0., evt.hcalPFClusterIso[idx] - evt.rho[idx]*hcalEA)/evt.pt[idx]
    trackIso = evt.dr03TkSumPt[idx]/evt.pt[idx]
    
    if not ecalPFClusterIso < 0.5: return False
    if not hcalPFClusterIso < 0.3: return False
    if not trackIso < 0.2: return False
    
    return True

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

def calculate_efficiency(numerator, denominator):
    if denominator == 0:
        return 0.0, 0.0
    eff = numerator / denominator
    eff_err = np.sqrt(eff * (1 - eff) / denominator) if denominator > 0 else 0.0
    return eff, eff_err

def optimize_tight_id(era, object_type, reduction=1):
    WORKDIR = os.getenv("WORKDIR")
    
    # Load events
    tree = ROOT.TChain("Events")
    if object_type == "muon":
        tree.Add(f"{WORKDIR}/SKNanoOutput/ParseMuIDVariables/{era}/TTLJ_powheg.root")
        tree.Add(f"{WORKDIR}/SKNanoOutput/ParseMuIDVariables/{era}/TTLL_powheg.root")
    else:  # electron
        tree.Add(f"{WORKDIR}/SKNanoOutput/ParseEleIDVariables/{era}/TTLJ_powheg.root")
        tree.Add(f"{WORKDIR}/SKNanoOutput/ParseEleIDVariables/{era}/TTLL_powheg.root")
    maxevt = int(tree.GetEntries() / reduction)
    
    # Define cut ranges for optimization
    miniiso_cuts = [0.05, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13]
    sip3d_cuts = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    
    regions = ["InnerBarrel", "OuterBarrel", "Endcap"]
    leptonTypes = ["prompt", "conv", "fromTau", "fromL", "fromC", "fromB", "fromPU", "unknown"]
    
    # Initialize counters
    # Structure: [region][leptonType][miniiso_cut][sip3d_cut] = {'pass': count, 'total': count}
    counters = {}
    for region in regions:
        counters[region] = {}
        for lType in leptonTypes:
            counters[region][lType] = {}
            for miniiso_cut in miniiso_cuts:
                counters[region][lType][miniiso_cut] = {}
                for sip3d_cut in sip3d_cuts:
                    counters[region][lType][miniiso_cut][sip3d_cut] = {'pass': 0.0, 'total': 0.0}
    
    print(f"Starting {object_type} optimization for {era} with {maxevt} events...")
    
    # Event loop
    for i, evt in enumerate(tree):
        if i > maxevt: 
            break
        if i % 50000 == 0:
            print(f"Processing event {i}/{maxevt}")
        
        genWeight = evt.genWeight
        nLeptons = evt.nMuons if object_type == "muon" else evt.nElectrons
        
        for lepton_idx in range(nLeptons):
            if object_type == "muon":
                if not evt.isIsoMuTrigMatched[lepton_idx]: continue
                region = check_region_muon(evt.eta[lepton_idx])
                if not evt.tkRelIso[lepton_idx] < 0.4: continue
            else:  # electron
                if not evt.isIsoElTrigMatched[lepton_idx]: continue
                region = check_region_electron(evt.scEta[lepton_idx])
                if not apply_electron_trigger_cuts(evt, lepton_idx, region): continue
            
            lType = classify_lepton(evt.lepType[lepton_idx], evt.nearestJetFlavour[lepton_idx])
            
            # Apply baseline cuts (before optimization)
            if object_type == "muon":
                if not evt.isPOGMediumId[lepton_idx]: 
                    continue
                if not abs(evt.dZ[lepton_idx]) < 0.1:  # Fixed dZ cut
                    continue
            else:  # electron
                if not evt.isMVANoIsoWP90[lepton_idx]:
                    continue
                if not evt.convVeto[lepton_idx]: 
                    continue
                if not evt.lostHits[lepton_idx] < 2: 
                    continue
                if not abs(evt.dZ[lepton_idx]) < 0.1:  # Fixed dZ cut
                    continue
            
            # Test all combinations of miniIso and sip3d cuts
            for miniiso_cut in miniiso_cuts:
                for sip3d_cut in sip3d_cuts:
                    # Count total events passing baseline
                    counters[region][lType][miniiso_cut][sip3d_cut]['total'] += genWeight
                    
                    # Check if passes tight cuts
                    passes_tight = False
                    if object_type == "muon":
                        passes_tight = (evt.miniPFRelIso[lepton_idx] < miniiso_cut and
                                      evt.sip3d[lepton_idx] < sip3d_cut)
                    else:  # electron
                        passes_tight = (evt.miniPFRelIso[lepton_idx] < miniiso_cut and
                                      evt.sip3d[lepton_idx] < sip3d_cut)
                    
                    if passes_tight:
                        counters[region][lType][miniiso_cut][sip3d_cut]['pass'] += genWeight
    
    return counters

def save_results(results, era, object_type, format_type="json"):
    WORKDIR = os.getenv("WORKDIR")
    output_dir = f"{WORKDIR}/LeptonIDTest/optimization"
    os.makedirs(output_dir, exist_ok=True)
    
    if format_type == "json":
        # Convert to JSON-serializable format with efficiency calculations
        json_results = {}
        for region in results:
            json_results[region] = {}
            for lType in results[region]:
                json_results[region][lType] = {}
                for miniiso_cut in results[region][lType]:
                    json_results[region][lType][str(miniiso_cut)] = {}
                    for sip3d_cut in results[region][lType][miniiso_cut]:
                        pass_count = results[region][lType][miniiso_cut][sip3d_cut]['pass']
                        total_count = results[region][lType][miniiso_cut][sip3d_cut]['total']
                        eff, eff_err = calculate_efficiency(pass_count, total_count)
                        
                        json_results[region][lType][str(miniiso_cut)][str(sip3d_cut)] = {
                            'efficiency': eff,
                            'efficiency_error': eff_err,
                            'pass_events': pass_count,
                            'total_events': total_count
                        }
        
        with open(f"{output_dir}/tightID_optimization_{object_type}_{era}.json", 'w') as f:
            json.dump(json_results, f, indent=2)
    
    elif format_type == "csv":
        csv_file = f"{output_dir}/tightID_optimization_{object_type}_{era}.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Era', 'Region', 'LeptonType', 'miniIso_cut', 'sip3d_cut', 
                           'Efficiency', 'Efficiency_Error', 'Pass_Events', 'Total_Events'])
            
            for region in results:
                for lType in results[region]:
                    for miniiso_cut in results[region][lType]:
                        for sip3d_cut in results[region][lType][miniiso_cut]:
                            pass_count = results[region][lType][miniiso_cut][sip3d_cut]['pass']
                            total_count = results[region][lType][miniiso_cut][sip3d_cut]['total']
                            eff, eff_err = calculate_efficiency(pass_count, total_count)
                            
                            writer.writerow([era, region, lType, miniiso_cut, sip3d_cut,
                                           eff, eff_err, pass_count, total_count])

def main():
    parser = argparse.ArgumentParser(description="Optimize tight lepton ID for Run3")
    parser.add_argument("--era", type=str, required=True, 
                       choices=["2016preVFP", "2016postVFP", "2017", "2018", "2022", "2022EE", "2023", "2023BPix"],
                       help="era to optimize")
    parser.add_argument("--object", type=str, required=True,
                       choices=["muon", "electron"],
                       help="Object type to optimize (muon or electron)")
    parser.add_argument("--reduction", default=1, type=int, 
                       help="Reduce number of events for testing")
    parser.add_argument("--format", default="json", choices=["json", "csv"],
                       help="Output format for results")
    
    args = parser.parse_args()
    
    print(f"Starting {args.object} tight ID optimization for {args.era}")
    results = optimize_tight_id(args.era, args.object, args.reduction)
    
    print(f"Saving results in {args.format} format...")
    save_results(results, args.era, args.object, args.format)
    
    print(f"Optimization complete for {args.era}")
    print(f"Results saved to LeptonIDTest/optimization/tightID_optimization_{args.object}_{args.era}.{args.format}")

if __name__ == "__main__":
    main()

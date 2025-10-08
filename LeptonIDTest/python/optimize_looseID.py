#!/usr/bin/env python
import os
import argparse
import csv
import ROOT
import numpy as np

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

def get_electron_cuts(region):
    """Get electron trigger cuts for each region"""
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
    
    return c_sieie, c_dEta, c_dPhi, c_hoe, ecalEA, hcalEA

def apply_electron_trigger_cuts(evt, idx, region):
    """Apply electron trigger emulation cuts"""
    c_sieie, c_dEta, c_dPhi, c_hoe, ecalEA, hcalEA = get_electron_cuts(region)
    
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

def calculate_muon_bc_metric(results_b, results_c, pt_bins):
    """Calculate metric for muon b/c jet fake rate differences"""
    difference_metrics = {}
    
    for region in results_b:
        difference_metrics[region] = {}
        for miniiso_cut in results_b[region]['fromB']:
            difference_metrics[region][miniiso_cut] = {}
            for sip3d_cut in results_b[region]['fromB'][miniiso_cut]:
                difference_metrics[region][miniiso_cut][sip3d_cut] = {}
                
                total_significance = 0.0
                n_bins_used = 0
                
                for pt_bin in pt_bins:
                    if (pt_bin in results_b[region]['fromB'][miniiso_cut][sip3d_cut] and 
                        pt_bin in results_c[region]['fromC'][miniiso_cut][sip3d_cut]):
                        
                        b_data = results_b[region]['fromB'][miniiso_cut][sip3d_cut][pt_bin]
                        c_data = results_c[region]['fromC'][miniiso_cut][sip3d_cut][pt_bin]
                        
                        if b_data['loose_events'] < 5 or c_data['loose_events'] < 5:
                            continue
                        
                        fakerate_diff = abs(b_data['fake_rate'] - c_data['fake_rate'])
                        combined_err = np.sqrt(b_data['fake_rate_error']**2 + c_data['fake_rate_error']**2)
                        
                        if combined_err > 0:
                            significance = fakerate_diff / combined_err
                            total_significance += significance
                            n_bins_used += 1
                
                difference_metrics[region][miniiso_cut][sip3d_cut] = {
                    'optimization_metric': total_significance,
                    'n_bins_used': n_bins_used
                }
    
    return difference_metrics

def calculate_electron_lbc_metric(results_l, results_b, results_c, pt_bins):
    """Calculate metric for electron L/B/C jet fake rate differences"""
    difference_metrics = {}
    
    for region in results_l:
        difference_metrics[region] = {}
        for miniiso_cut in results_l[region]['fromL']:
            difference_metrics[region][miniiso_cut] = {}
            for sip3d_cut in results_l[region]['fromL'][miniiso_cut]:
                difference_metrics[region][miniiso_cut][sip3d_cut] = {}
                for mva_cut in results_l[region]['fromL'][miniiso_cut][sip3d_cut]:
                    difference_metrics[region][miniiso_cut][sip3d_cut][mva_cut] = {}
                    
                    total_significance = 0.0
                    n_bins_used = 0
                    
                    for pt_bin in pt_bins:
                        l_data = results_l[region]['fromL'][miniiso_cut][sip3d_cut][mva_cut].get(pt_bin)
                        b_data = results_b[region]['fromB'][miniiso_cut][sip3d_cut][mva_cut].get(pt_bin)
                        c_data = results_c[region]['fromC'][miniiso_cut][sip3d_cut][mva_cut].get(pt_bin)
                        
                        if not (l_data and b_data and c_data):
                            continue
                        
                        if (l_data['loose_events'] < 5 or b_data['loose_events'] < 5 or c_data['loose_events'] < 5):
                            continue
                        
                        # Calculate pairwise significances
                        lb_diff = abs(l_data['fake_rate'] - b_data['fake_rate'])
                        lb_err = np.sqrt(l_data['fake_rate_error']**2 + b_data['fake_rate_error']**2)
                        lb_sig = lb_diff / lb_err if lb_err > 0 else 0.0
                        
                        lc_diff = abs(l_data['fake_rate'] - c_data['fake_rate'])
                        lc_err = np.sqrt(l_data['fake_rate_error']**2 + c_data['fake_rate_error']**2)
                        lc_sig = lc_diff / lc_err if lc_err > 0 else 0.0
                        
                        bc_diff = abs(b_data['fake_rate'] - c_data['fake_rate'])
                        bc_err = np.sqrt(b_data['fake_rate_error']**2 + c_data['fake_rate_error']**2)
                        bc_sig = bc_diff / bc_err if bc_err > 0 else 0.0
                        
                        total_significance += lb_sig + lc_sig + bc_sig
                        n_bins_used += 1
                    
                    difference_metrics[region][miniiso_cut][sip3d_cut][mva_cut] = {
                        'optimization_metric': total_significance,
                        'n_bins_used': n_bins_used
                    }
    
    return difference_metrics

def optimize_muon(era, reduction=1):
    WORKDIR = os.getenv("WORKDIR")
    
    tree = ROOT.TChain("Events")
    tree.Add(f"{WORKDIR}/SKNanoOutput/ParseMuIDVariables/{era}/TTLJ_powheg.root")
    tree.Add(f"{WORKDIR}/SKNanoOutput/ParseMuIDVariables/{era}/TTLL_powheg.root")
    maxevt = int(tree.GetEntries() / reduction)
    
    miniiso_cuts = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
    sip3d_cuts = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    regions = ["InnerBarrel", "OuterBarrel", "Endcap"]
    target_types = ["fromB", "fromC"]
    pt_bins = [10, 20, 30, 50, 70]
    
    counters = {}
    for region in regions:
        counters[region] = {}
        for lType in target_types:
            counters[region][lType] = {}
            for miniiso_cut in miniiso_cuts:
                counters[region][lType][miniiso_cut] = {}
                for sip3d_cut in sip3d_cuts:
                    counters[region][lType][miniiso_cut][sip3d_cut] = {}
                    for pt_bin in pt_bins:
                        counters[region][lType][miniiso_cut][sip3d_cut][pt_bin] = {'loose': 0.0, 'tight': 0.0}
    
    print(f"Starting muon loose ID optimization for {era} with {maxevt} events...")
    print(f"Target: minimize b/c jet fake rate difference")
    
    for i, evt in enumerate(tree):
        if i > maxevt: break
        if i % 50000 == 0:
            print(f"Processing event {i}/{maxevt}")
        
        genWeight = evt.genWeight
        nMuons = evt.nMuons
        
        for muon_idx in range(nMuons):
            if not evt.isIsoMuTrigMatched[muon_idx]: continue
            region = check_region_muon(evt.eta[muon_idx])
            if not evt.tkRelIso[muon_idx] < 0.4: continue
            
            lType = classify_lepton(evt.lepType[muon_idx], evt.nearestJetFlavour[muon_idx])
            if lType not in target_types: continue
            
            if not evt.isPOGMediumId[muon_idx]: continue
            if not abs(evt.dZ[muon_idx]) < 0.1: continue
            
            pt_corr = evt.pt[muon_idx] * (1.0 + max(0., evt.miniPFRelIso[muon_idx] - 0.1))
            
            pt_bin = None
            if 10 <= pt_corr < 20: pt_bin = 10
            elif 20 <= pt_corr < 30: pt_bin = 20
            elif 30 <= pt_corr < 50: pt_bin = 30
            elif 50 <= pt_corr < 70: pt_bin = 50
            elif pt_corr >= 70: pt_bin = 70
            
            if pt_bin is None: continue
            
            for miniiso_cut in miniiso_cuts:
                for sip3d_cut in sip3d_cuts:
                    passes_loose = (evt.miniPFRelIso[muon_idx] < miniiso_cut and evt.sip3d[muon_idx] < sip3d_cut)
                    
                    if passes_loose:
                        counters[region][lType][miniiso_cut][sip3d_cut][pt_bin]['loose'] += genWeight
                        
                        passes_tight = (evt.miniPFRelIso[muon_idx] < 0.1 and evt.sip3d[muon_idx] < 3.0)
                        if passes_tight:
                            counters[region][lType][miniiso_cut][sip3d_cut][pt_bin]['tight'] += genWeight
    
    return counters, pt_bins, target_types

def optimize_electron(era, reduction=1):
    WORKDIR = os.getenv("WORKDIR")
    
    tree = ROOT.TChain("Events")
    tree.Add(f"{WORKDIR}/SKNanoOutput/ParseEleIDVariables/{era}/TTLJ_powheg.root")
    tree.Add(f"{WORKDIR}/SKNanoOutput/ParseEleIDVariables/{era}/TTLL_powheg.root")
    maxevt = int(tree.GetEntries() / reduction)
    
    miniiso_cuts = [0.4]
    sip3d_cuts = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    mva_cuts = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95]
    regions = ["InnerBarrel", "OuterBarrel", "Endcap"]
    target_types = ["fromL", "fromB", "fromC"]
    pt_bins = [15, 25, 35, 50, 70]
    
    counters = {}
    for region in regions:
        counters[region] = {}
        for lType in target_types:
            counters[region][lType] = {}
            for miniiso_cut in miniiso_cuts:
                counters[region][lType][miniiso_cut] = {}
                for sip3d_cut in sip3d_cuts:
                    counters[region][lType][miniiso_cut][sip3d_cut] = {}
                    for mva_cut in mva_cuts:
                        counters[region][lType][miniiso_cut][sip3d_cut][mva_cut] = {}
                        for pt_bin in pt_bins:
                            counters[region][lType][miniiso_cut][sip3d_cut][mva_cut][pt_bin] = {'loose': 0.0, 'tight': 0.0}
    
    print(f"Starting electron loose ID optimization for {era} with {maxevt} events...")
    print(f"Target: minimize light/b/c jet fake rate differences")
    
    for i, evt in enumerate(tree):
        if i > maxevt: break
        if i % 50000 == 0:
            print(f"Processing event {i}/{maxevt}")
        
        genWeight = evt.genWeight
        nElectrons = evt.nElectrons
        
        for electron_idx in range(nElectrons):
            if not evt.isIsoElTrigMatched[electron_idx]: continue
            region = check_region_electron(evt.scEta[electron_idx])
            if not apply_electron_trigger_cuts(evt, electron_idx, region): continue
            
            lType = classify_lepton(evt.lepType[electron_idx], evt.nearestJetFlavour[electron_idx])
            if lType not in target_types: continue
            
            if not evt.convVeto[electron_idx]: continue
            if not evt.lostHits[electron_idx] < 2: continue
            if not abs(evt.dZ[electron_idx]) < 0.1: continue
            
            pt_corr = evt.pt[electron_idx] * (1.0 + max(0., evt.miniPFRelIso[electron_idx] - 0.1))
            
            pt_bin = None
            if 15 <= pt_corr < 25: pt_bin = 15
            elif 25 <= pt_corr < 35: pt_bin = 25
            elif 35 <= pt_corr < 50: pt_bin = 35
            elif 50 <= pt_corr < 70: pt_bin = 50
            elif pt_corr >= 70: pt_bin = 70
            
            if pt_bin is None: continue
            
            for miniiso_cut in miniiso_cuts:
                for sip3d_cut in sip3d_cuts:
                    for mva_cut in mva_cuts:
                        passes_loose = (evt.miniPFRelIso[electron_idx] < miniiso_cut and
                                       evt.sip3d[electron_idx] < sip3d_cut and
                                       (evt.isMVANoIsoWP90[electron_idx] or evt.mvaNoIso[electron_idx] > mva_cut))
                        
                        if passes_loose:
                            counters[region][lType][miniiso_cut][sip3d_cut][mva_cut][pt_bin]['loose'] += genWeight
                            
                            passes_tight = (evt.isMVANoIsoWP90[electron_idx] and
                                           evt.sip3d[electron_idx] < 3.0 and
                                           evt.miniPFRelIso[electron_idx] < 0.1)
                            
                            if passes_tight:
                                counters[region][lType][miniiso_cut][sip3d_cut][mva_cut][pt_bin]['tight'] += genWeight
    
    return counters, pt_bins, target_types

def process_muon_results(counters, pt_bins):
    results = {}
    for region in counters:
        results[region] = {}
        for lType in counters[region]:
            results[region][lType] = {}
            for miniiso_cut in counters[region][lType]:
                results[region][lType][miniiso_cut] = {}
                for sip3d_cut in counters[region][lType][miniiso_cut]:
                    results[region][lType][miniiso_cut][sip3d_cut] = {}
                    for pt_bin in counters[region][lType][miniiso_cut][sip3d_cut]:
                        tight_count = counters[region][lType][miniiso_cut][sip3d_cut][pt_bin]['tight']
                        loose_count = counters[region][lType][miniiso_cut][sip3d_cut][pt_bin]['loose']
                        fake_rate, fake_rate_err = calculate_efficiency(tight_count, loose_count)
                        results[region][lType][miniiso_cut][sip3d_cut][pt_bin] = {
                            'fake_rate': fake_rate,
                            'fake_rate_error': fake_rate_err,
                            'tight_events': tight_count,
                            'loose_events': loose_count
                        }
    return results

def process_electron_results(counters, pt_bins):
    results = {}
    for region in counters:
        results[region] = {}
        for lType in counters[region]:
            results[region][lType] = {}
            for miniiso_cut in counters[region][lType]:
                results[region][lType][miniiso_cut] = {}
                for sip3d_cut in counters[region][lType][miniiso_cut]:
                    results[region][lType][miniiso_cut][sip3d_cut] = {}
                    for mva_cut in counters[region][lType][miniiso_cut][sip3d_cut]:
                        results[region][lType][miniiso_cut][sip3d_cut][mva_cut] = {}
                        for pt_bin in counters[region][lType][miniiso_cut][sip3d_cut][mva_cut]:
                            tight_count = counters[region][lType][miniiso_cut][sip3d_cut][mva_cut][pt_bin]['tight']
                            loose_count = counters[region][lType][miniiso_cut][sip3d_cut][mva_cut][pt_bin]['loose']
                            fake_rate, fake_rate_err = calculate_efficiency(tight_count, loose_count)
                            results[region][lType][miniiso_cut][sip3d_cut][mva_cut][pt_bin] = {
                                'fake_rate': fake_rate,
                                'fake_rate_error': fake_rate_err,
                                'tight_events': tight_count,
                                'loose_events': loose_count
                            }
    return results

def save_muon_results(difference_metrics, era):
    WORKDIR = os.getenv("WORKDIR")
    output_dir = f"{WORKDIR}/LeptonIDTest/optimization"
    os.makedirs(output_dir, exist_ok=True)
    
    csv_file = f"{output_dir}/looseID_optimization_muon_{era}.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Era', 'Region', 'miniIso_cut', 'sip3d_cut', 
                        'optimization_metric', 'n_bins_used', 'recommendation'])
        
        best_cuts = {}
        for region in difference_metrics:
            best_cuts[region] = {'cut': None, 'metric': float('inf')}
            for miniiso_cut in difference_metrics[region]:
                for sip3d_cut in difference_metrics[region][miniiso_cut]:
                    metric = difference_metrics[region][miniiso_cut][sip3d_cut]['optimization_metric']
                    n_bins = difference_metrics[region][miniiso_cut][sip3d_cut]['n_bins_used']
                    
                    recommendation = ""
                    if metric < best_cuts[region]['metric'] and n_bins >= 3:
                        best_cuts[region]['metric'] = metric
                        best_cuts[region]['cut'] = f"miniIso<{miniiso_cut}, SIP3D<{sip3d_cut}"
                        recommendation = "BEST"
                    
                    writer.writerow([era, region, miniiso_cut, sip3d_cut, 
                                   f"{metric:.3f}", f"{n_bins}", recommendation])
    
    print(f"Results saved to: {csv_file}")
    print(f"\nRecommended muon loose ID WP for {era}:")
    for region in best_cuts:
        if best_cuts[region]['cut'] and best_cuts[region]['metric'] != float('inf'):
            print(f"  {region}: {best_cuts[region]['cut']} (sum significance: {best_cuts[region]['metric']:.3f})")
        else:
            print(f"  {region}: Insufficient statistics for recommendation")

def save_electron_results(difference_metrics, era):
    WORKDIR = os.getenv("WORKDIR")
    output_dir = f"{WORKDIR}/LeptonIDTest/optimization"
    os.makedirs(output_dir, exist_ok=True)
    
    csv_file = f"{output_dir}/looseID_optimization_electron_{era}.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Era', 'Region', 'miniIso_cut', 'sip3d_cut', 'mva_cut',
                        'optimization_metric', 'n_bins_used', 'recommendation'])
        
        best_cuts = {}
        for region in difference_metrics:
            best_cuts[region] = {'cut': None, 'metric': float('inf')}
            for miniiso_cut in difference_metrics[region]:
                for sip3d_cut in difference_metrics[region][miniiso_cut]:
                    for mva_cut in difference_metrics[region][miniiso_cut][sip3d_cut]:
                        metric = difference_metrics[region][miniiso_cut][sip3d_cut][mva_cut]['optimization_metric']
                        n_bins = difference_metrics[region][miniiso_cut][sip3d_cut][mva_cut]['n_bins_used']
                        
                        recommendation = ""
                        if metric < best_cuts[region]['metric'] and n_bins >= 3:
                            best_cuts[region]['metric'] = metric
                            best_cuts[region]['cut'] = f"miniIso<{miniiso_cut}, SIP3D<{sip3d_cut}, MVA>{mva_cut}"
                            recommendation = "BEST"
                        
                        writer.writerow([era, region, miniiso_cut, sip3d_cut, mva_cut,
                                       f"{metric:.3f}", f"{n_bins}", recommendation])
    
    print(f"Results saved to: {csv_file}")
    print(f"\nRecommended electron loose ID WP for {era}:")
    for region in best_cuts:
        if best_cuts[region]['cut'] and best_cuts[region]['metric'] != float('inf'):
            print(f"  {region}: {best_cuts[region]['cut']} (sum significance: {best_cuts[region]['metric']:.3f})")
        else:
            print(f"  {region}: Insufficient statistics for recommendation")

def main():
    parser = argparse.ArgumentParser(description="Optimize loose lepton ID for Run3")
    parser.add_argument("--era", type=str, required=True, 
                       choices=["2022", "2022EE", "2023", "2023BPix"],
                       help="Run3 era to optimize")
    parser.add_argument("--object", type=str, required=True,
                       choices=["muon", "electron"],
                       help="Object type to optimize")
    parser.add_argument("--reduction", default=1, type=int, 
                       help="Reduce number of events for testing (default: 1, no reduction)")
    
    args = parser.parse_args()
    
    print(f"Starting {args.object} loose ID optimization for {args.era}")
    
    if args.object == "muon":
        print("Objective: Minimize fake rate difference between muons from b-jets and c-jets")
        print("Reference Run2 loose WP: SIP3D < 6, miniIso < 0.6")
        
        counters, pt_bins, target_types = optimize_muon(args.era, args.reduction)
        results = process_muon_results(counters, pt_bins)
        
        results_b = {region: {'fromB': results[region]['fromB']} for region in results}
        results_c = {region: {'fromC': results[region]['fromC']} for region in results}
        difference_metrics = calculate_muon_bc_metric(results_b, results_c, pt_bins)
        
        save_muon_results(difference_metrics, args.era)
        
    else:  # electron
        print("Objective: Minimize fake rate differences among light, b, and c jets")
        print("Reference Run2 loose WP: isMVANoIsoWP90 OR mvaNoIso > 0.5, SIP3D < 8, miniIso < 0.4")
        
        counters, pt_bins, target_types = optimize_electron(args.era, args.reduction)
        results = process_electron_results(counters, pt_bins)
        
        results_l = {region: {'fromL': results[region]['fromL']} for region in results}
        results_b = {region: {'fromB': results[region]['fromB']} for region in results}
        results_c = {region: {'fromC': results[region]['fromC']} for region in results}
        difference_metrics = calculate_electron_lbc_metric(results_l, results_b, results_c, pt_bins)
        
        save_electron_results(difference_metrics, args.era)
    
    print(f"{args.object.capitalize()} loose ID optimization complete for {args.era}")

if __name__ == "__main__":
    main()

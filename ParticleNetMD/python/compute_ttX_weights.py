#!/usr/bin/env python3
"""Compute sum of event weights for ttX samples after Tight+Bjet selection.

Selection:
  - All muons pass MuonIsTightColl
  - All electrons pass ElectronIsTightColl
  - At least one jet passes JetIsBtaggedColl

Weight: genWeight * puWeight * prefireWeight
"""

import os
import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.EnableImplicitMT(8)

BASE = "/pscratch/sd/c/choij/workspace/ChargedHiggsAnalysisV3/SKNanoOutput/EvtTreeProducer"

CHANNELS = ["Run1E2Mu", "Run3Mu"]
RUN2_ERAS = ["2016preVFP", "2016postVFP", "2017", "2018"]
RUN3_ERAS = ["2022", "2022EE", "2023", "2023BPix"]

# Sample definitions: (display_name, file_sample_name, eras)
SAMPLES = [
    ("TTZToLLNuNu", "TTZToLLNuNu", RUN2_ERAS),
    ("TTZ_M50",     "TTZ_M50",     RUN3_ERAS),
    ("TTWToLNu",    "TTWToLNu",    RUN2_ERAS + RUN3_ERAS),
    ("TTHToNonbb",  "TTHToNonbb",  RUN2_ERAS + RUN3_ERAS),
    ("tZq",         "tZq",         RUN2_ERAS + RUN3_ERAS),
]


def process_file(filepath):
    """Process a single ROOT file. Returns (sumW, nEvents) or None if file missing."""
    if not os.path.exists(filepath):
        return None

    rdf = ROOT.RDataFrame("Events", filepath)

    # Define Tight+Bjet selection using C++ lambdas for array branches
    # All muons tight
    rdf_sel = rdf.Define("allMuonsTight",
        "bool pass = true; for (unsigned int i = 0; i < nMuons; i++) { if (!MuonIsTightColl[i]) { pass = false; break; } } return pass;")

    # All electrons tight
    rdf_sel = rdf_sel.Define("allElectronsTight",
        "bool pass = true; for (unsigned int i = 0; i < nElectrons; i++) { if (!ElectronIsTightColl[i]) { pass = false; break; } } return pass;")

    # At least one b-tagged jet
    rdf_sel = rdf_sel.Define("hasBjet",
        "bool pass = false; for (unsigned int i = 0; i < nJets; i++) { if (JetIsBtaggedColl[i]) { pass = true; break; } } return pass;")

    # Apply selection
    rdf_sel = rdf_sel.Filter("allMuonsTight && allElectronsTight && hasBjet")

    # Compute weight
    rdf_sel = rdf_sel.Define("evtWeight", "genWeight * puWeight * prefireWeight")

    # Get results
    sum_w = rdf_sel.Sum("evtWeight")
    n_evt = rdf_sel.Count()

    return (sum_w.GetValue(), n_evt.GetValue())


def main():
    # Results: results[sample_name][channel] = {"run2_sumW", "run3_sumW", "run2_nEvt", "run3_nEvt"}
    results = {}

    for display_name, file_sample, eras in SAMPLES:
        results[display_name] = {}
        for channel in CHANNELS:
            run2_sumW = 0.0
            run3_sumW = 0.0
            run2_nEvt = 0
            run3_nEvt = 0

            for era in eras:
                filepath = os.path.join(BASE, channel, era, f"Skim_TriLep_{file_sample}.root")
                result = process_file(filepath)

                if result is None:
                    print(f"  WARNING: Missing {filepath}")
                    continue

                sumW, nEvt = result
                print(f"  {channel}/{era}/{file_sample}: sumW={sumW:.4f}, nEvents={nEvt}")

                if era in RUN2_ERAS:
                    run2_sumW += sumW
                    run3_nEvt += 0  # no-op, just for clarity
                    run2_nEvt += nEvt
                else:
                    run3_sumW += sumW
                    run3_nEvt += nEvt

            results[display_name][channel] = {
                "run2_sumW": run2_sumW,
                "run3_sumW": run3_sumW,
                "run2_nEvt": run2_nEvt,
                "run3_nEvt": run3_nEvt,
            }

    # Print summary table
    print("\n" + "=" * 130)
    print(f"{'Sample':<16} | {'Channel':<10} | {'Run2 sumW':>14} | {'Run3 sumW':>14} | {'Total sumW':>14} | {'Run2 nEvt':>10} | {'Run3 nEvt':>10} | {'Total nEvt':>10}")
    print("-" * 130)

    for display_name in [s[0] for s in SAMPLES]:
        for channel in CHANNELS:
            r = results[display_name][channel]
            total_sumW = r["run2_sumW"] + r["run3_sumW"]
            total_nEvt = r["run2_nEvt"] + r["run3_nEvt"]
            print(f"{display_name:<16} | {channel:<10} | {r['run2_sumW']:>14.4f} | {r['run3_sumW']:>14.4f} | {total_sumW:>14.4f} | {r['run2_nEvt']:>10d} | {r['run3_nEvt']:>10d} | {total_nEvt:>10d}")
        print("-" * 130)

    # Print combined (both channels) summary
    print("\n" + "=" * 110)
    print("Combined (Run1E2Mu + Run3Mu)")
    print("=" * 110)
    print(f"{'Sample':<16} | {'Run2 sumW':>14} | {'Run3 sumW':>14} | {'Total sumW':>14} | {'Run2 nEvt':>10} | {'Run3 nEvt':>10} | {'Total nEvt':>10}")
    print("-" * 110)

    grand_run2_sumW = 0.0
    grand_run3_sumW = 0.0
    grand_run2_nEvt = 0
    grand_run3_nEvt = 0

    for display_name in [s[0] for s in SAMPLES]:
        comb_run2_sumW = sum(results[display_name][ch]["run2_sumW"] for ch in CHANNELS)
        comb_run3_sumW = sum(results[display_name][ch]["run3_sumW"] for ch in CHANNELS)
        comb_run2_nEvt = sum(results[display_name][ch]["run2_nEvt"] for ch in CHANNELS)
        comb_run3_nEvt = sum(results[display_name][ch]["run3_nEvt"] for ch in CHANNELS)
        comb_total_sumW = comb_run2_sumW + comb_run3_sumW
        comb_total_nEvt = comb_run2_nEvt + comb_run3_nEvt

        grand_run2_sumW += comb_run2_sumW
        grand_run3_sumW += comb_run3_sumW
        grand_run2_nEvt += comb_run2_nEvt
        grand_run3_nEvt += comb_run3_nEvt

        print(f"{display_name:<16} | {comb_run2_sumW:>14.4f} | {comb_run3_sumW:>14.4f} | {comb_total_sumW:>14.4f} | {comb_run2_nEvt:>10d} | {comb_run3_nEvt:>10d} | {comb_total_nEvt:>10d}")

    print("-" * 110)
    grand_total_sumW = grand_run2_sumW + grand_run3_sumW
    grand_total_nEvt = grand_run2_nEvt + grand_run3_nEvt
    print(f"{'ALL ttX':<16} | {grand_run2_sumW:>14.4f} | {grand_run3_sumW:>14.4f} | {grand_total_sumW:>14.4f} | {grand_run2_nEvt:>10d} | {grand_run3_nEvt:>10d} | {grand_total_nEvt:>10d}")
    print("=" * 110)


if __name__ == "__main__":
    main()

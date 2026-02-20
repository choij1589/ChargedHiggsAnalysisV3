#!/usr/bin/env python
"""
Plot 2D jet eta-phi maps before and after jet veto map application.
Compares Data and MC for DIMU (DYJets) and EMU (TTLL_powheg) channels.
"""
import os
import argparse
import logging
import json
import ROOT
import cmsstyle as CMS
from plotter import get_CoM_energy, LumiInfo

ROOT.gROOT.SetBatch(True)
CMS.setCMSStyle()

# Valid individual eras (no Run2/Run3 combination)
VALID_ERAS = ["2016preVFP", "2016postVFP", "2017", "2018", "2022", "2022EE", "2023", "2023BPix"]


def parse_args():
    parser = argparse.ArgumentParser(description="Plot jet veto map (eta vs phi)")
    parser.add_argument("--era", required=True, type=str, choices=VALID_ERAS,
                        help="Era (individual eras only, no Run2/Run3)")
    parser.add_argument("--channel", required=True, type=str, choices=["DIMU", "EMU"],
                        help="Channel: DIMU or EMU")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def load_histogram_2d(file_path, hist_path, era):
    """
    Load a single TH2F histogram from a ROOT file.

    Args:
        file_path: Path to ROOT file
        hist_path: Path to histogram within file
        era: Era name for logging

    Returns:
        TH2F histogram or None if not found
    """
    if not os.path.exists(file_path):
        logging.warning(f"File not found: {file_path}")
        return None

    f = ROOT.TFile.Open(file_path)
    if not f or f.IsZombie():
        logging.warning(f"Cannot open file: {file_path}")
        return None

    hist = f.Get(hist_path)
    if not hist:
        logging.warning(f"Histogram {hist_path} not found in {file_path}")
        f.Close()
        return None

    hist.SetDirectory(0)
    f.Close()
    logging.debug(f"Loaded {hist_path} from {file_path}")
    return hist


def load_and_merge_histograms(era, samples, hist_path, workdir, flag):
    """
    Load and merge histograms from multiple samples.

    Args:
        era: Era name
        samples: List of sample names
        hist_path: Path to histogram within file
        workdir: Base working directory
        flag: Run flag (RunDiMu or RunEMu)

    Returns:
        Merged TH2F histogram or None if all samples missing
    """
    merged = None
    for sample in samples:
        file_path = f"{workdir}/SKNanoOutput/DiLepton/{flag}_RunSyst/{era}/{sample}.root"

        hist = load_histogram_2d(file_path, hist_path, era)
        if hist is None:
            continue

        if merged is None:
            merged = hist.Clone("merged")
        else:
            merged.Add(hist)

    return merged


def calculate_jet_efficiency_run2(era, samples, workdir, flag):
    """
    Calculate jet efficiency for Run2 from before/after jet veto histograms.

    Args:
        era: Era name
        samples: List of sample names
        workdir: Base working directory
        flag: Run flag (RunDiMu or RunEMu)

    Returns:
        Tuple of (before_integral, after_integral, efficiency) or None if failed
    """
    before_hist = load_and_merge_histograms(
        era, samples, "JetEtaPhi/BeforeJetVeto/eta_phi", workdir, flag
    )
    after_hist = load_and_merge_histograms(
        era, samples, "JetEtaPhi/AfterJetVeto/eta_phi", workdir, flag
    )

    if before_hist is None or after_hist is None:
        return None

    before_int = before_hist.Integral()
    after_int = after_hist.Integral()

    if before_int <= 0:
        return None

    efficiency = after_int / before_int
    return (before_int, after_int, efficiency)


def calculate_event_efficiency_run3(era, samples, workdir, flag):
    """
    Calculate event efficiency for Run3 from cutflow histogram.
    Event veto efficiency is calculated from ALL/Central/cutflow:
    - CutStage::NoiseFilter (stage 1, bin 2): before veto
    - CutStage::VetoMap (stage 2, bin 3): after veto

    Args:
        era: Era name
        samples: List of sample names
        workdir: Base working directory
        flag: Run flag (RunDiMu or RunEMu)

    Returns:
        Tuple of (before_events, after_events, efficiency) or None if failed
    """
    before_total = 0.0
    after_total = 0.0

    for sample in samples:
        file_path = f"{workdir}/SKNanoOutput/DiLepton/{flag}_RunSyst/{era}/{sample}.root"

        if not os.path.exists(file_path):
            logging.warning(f"File not found: {file_path}")
            continue

        f = ROOT.TFile.Open(file_path)
        if not f or f.IsZombie():
            logging.warning(f"Cannot open file: {file_path}")
            continue

        # Use ALL cutflow since veto is applied before channel selection
        cutflow_path = "ALL/Central/cutflow"
        cutflow = f.Get(cutflow_path)
        if not cutflow:
            logging.warning(f"Cutflow not found: {cutflow_path} in {file_path}")
            f.Close()
            continue

        # Bin 2: NoiseFilter (before veto), Bin 3: VetoMap (after veto)
        before_total += cutflow.GetBinContent(2)
        after_total += cutflow.GetBinContent(3)
        f.Close()

    if before_total <= 0:
        return None

    efficiency = after_total / before_total
    return (before_total, after_total, efficiency)


def plot_2d_eta_phi(hist, output_path, era, sample_type, veto_status):
    """
    Create 2D eta-phi plot with CMS style.

    Args:
        hist: TH2F histogram to plot
        output_path: Output file path
        era: Era name
        sample_type: "Data" or MC sample name
        veto_status: "Before Jet Veto" or "After Jet Veto"
    """
    # Create canvas with z-axis space
    canvas = ROOT.TCanvas("c", "", 900, 800)
    canvas.SetRightMargin(0.15)
    canvas.SetLeftMargin(0.1)
    canvas.SetBottomMargin(0.1)
    canvas.SetTopMargin(0.06)

    # Set color palette (kRainbow or similar)
    ROOT.gStyle.SetPalette(ROOT.kRainBow)
    ROOT.gStyle.SetOptStat(0)

    # Set axis labels
    hist.GetXaxis().SetTitle("Jet #eta")
    hist.GetYaxis().SetTitle("Jet #phi")
    hist.GetZaxis().SetTitle("")
    hist.GetXaxis().SetTitleSize(0.06)
    hist.GetYaxis().SetTitleSize(0.06)
    hist.GetZaxis().SetTitleSize(0.06)
    hist.GetZaxis().SetLabelSize(0.035)
    hist.GetXaxis().SetTitleOffset(0.8)
    hist.GetYaxis().SetTitleOffset(0.8)
    hist.GetZaxis().SetTitleOffset(0.8)

    hist.GetXaxis().SetRangeUser(-2.5, 2.5)
    hist.GetYaxis().SetRangeUser(-3.2, 3.2)
    hist.Draw("COLZ")

    # Add labels
    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextFont(42)
    latex.SetTextSize(0.04)
    latex.DrawLatex(0.15, 0.85, sample_type)
    latex.DrawLatex(0.15, 0.80, veto_status)

    # CMS lumi label
    CMS.SetEnergy(get_CoM_energy(era))
    CMS.SetLumi(LumiInfo[era], run=era)
    CMS.SetExtraText("Preliminary")
    CMS.CMS_lumi(canvas, 0)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    canvas.SaveAs(output_path)
    logging.info(f"Saved: {output_path}")
    canvas.Close()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    WORKDIR = os.environ.get('WORKDIR')
    if not WORKDIR:
        raise RuntimeError("WORKDIR environment variable not set. Run 'source setup.sh' first.")

    # Load sample configuration
    config_path = "configs/samplegroup.json"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Sample config not found: {config_path}")

    with open(config_path) as f:
        SAMPLE_CONFIG = json.load(f)

    # Channel-specific settings
    if args.channel == "DIMU":
        FLAG = "RunDiMu"
        MC_SAMPLE = "DYJets"
    elif args.channel == "EMU":
        FLAG = "RunEMu"
        MC_SAMPLE = "TTLL_powheg"
    else:
        raise ValueError(f"Invalid channel: {args.channel}")

    # Get data samples from config
    if args.era not in SAMPLE_CONFIG:
        raise ValueError(f"Era {args.era} not found in sample config")
    if args.channel not in SAMPLE_CONFIG[args.era]:
        raise ValueError(f"Channel {args.channel} not found in config for era {args.era}")

    DATA_SAMPLES = SAMPLE_CONFIG[args.era][args.channel]["data"]
    logging.info(f"Data samples: {DATA_SAMPLES}")
    logging.info(f"MC sample: {MC_SAMPLE}")

    # Output directory
    OUTPUT_DIR = f"{WORKDIR}/DiLepton/plots/{args.era}/{args.channel}/JetVetoMap"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Histogram paths (different for Run2 vs Run3)
    RUN2_ERAS = ["2016preVFP", "2016postVFP", "2017", "2018"]
    RUN3_ERAS = ["2022", "2022EE", "2023", "2023BPix"]

    if args.era in RUN2_ERAS:
        HIST_PATHS = {
            "before": ("JetEtaPhi/BeforeJetVeto/eta_phi", "Before Jet Veto"),
            "after": ("JetEtaPhi/AfterJetVeto/eta_phi", "After Jet Veto")
        }
    elif args.era in RUN3_ERAS:
        HIST_PATHS = {
            "passed": ("JetEtaPhi/PassedEventVeto_Run3/eta_phi", "Passed Event Veto (Run3)")
        }
    else:
        raise ValueError(f"Unknown era: {args.era}")

    # Process each veto status
    for veto_key, (hist_path, veto_label) in HIST_PATHS.items():
        # Load and merge data histograms
        data_hist = load_and_merge_histograms(
            args.era, DATA_SAMPLES, hist_path, WORKDIR, FLAG
        )
        if data_hist is None:
            logging.error(f"No data histograms found for {veto_key}")
        else:
            output_path = f"{OUTPUT_DIR}/data_{veto_key}.png"
            plot_2d_eta_phi(data_hist, output_path, args.era, "Data", veto_label)

        # Load MC histogram
        mc_hist = load_and_merge_histograms(
            args.era, [MC_SAMPLE], hist_path, WORKDIR, FLAG
        )
        if mc_hist is None:
            logging.error(f"No MC histogram found for {MC_SAMPLE} {veto_key}")
        else:
            output_path = f"{OUTPUT_DIR}/{MC_SAMPLE}_{veto_key}.png"
            plot_2d_eta_phi(mc_hist, output_path, args.era, MC_SAMPLE, veto_label)

    # Calculate efficiency and prepare results
    results = {
        "era": args.era,
        "channel": args.channel,
        "mc_sample": MC_SAMPLE,
        "data_samples": DATA_SAMPLES,
    }

    print("\n" + "=" * 60)
    print(f"Efficiency Summary for {args.era} {args.channel}")
    print("=" * 60)

    if args.era in RUN2_ERAS:
        # Run2: Jet efficiency
        results["efficiency_type"] = "jet"
        print("\nJet Veto Efficiency (jets after / jets before):")

        data_eff = calculate_jet_efficiency_run2(args.era, DATA_SAMPLES, WORKDIR, FLAG)
        if data_eff:
            before, after, eff = data_eff
            print(f"  Data:  {after:.0f} / {before:.0f} = {eff * 100:.2f}%")
            results["data"] = {
                "before": before,
                "after": after,
                "efficiency": eff
            }
        else:
            print("  Data:  Failed to calculate")
            results["data"] = None

        mc_eff = calculate_jet_efficiency_run2(args.era, [MC_SAMPLE], WORKDIR, FLAG)
        if mc_eff:
            before, after, eff = mc_eff
            print(f"  {MC_SAMPLE}:  {after:.0f} / {before:.0f} = {eff * 100:.2f}%")
            results["mc"] = {
                "before": before,
                "after": after,
                "efficiency": eff
            }
        else:
            print(f"  {MC_SAMPLE}:  Failed to calculate")
            results["mc"] = None

    elif args.era in RUN3_ERAS:
        # Run3: Event efficiency
        results["efficiency_type"] = "event"
        print("\nEvent Veto Efficiency (events after / events before):")

        data_eff = calculate_event_efficiency_run3(
            args.era, DATA_SAMPLES, WORKDIR, FLAG
        )
        if data_eff:
            before, after, eff = data_eff
            print(f"  Data:  {after:.0f} / {before:.0f} = {eff * 100:.2f}%")
            results["data"] = {
                "before": before,
                "after": after,
                "efficiency": eff
            }
        else:
            print("  Data:  Failed to calculate")
            results["data"] = None

        mc_eff = calculate_event_efficiency_run3(
            args.era, [MC_SAMPLE], WORKDIR, FLAG
        )
        if mc_eff:
            before, after, eff = mc_eff
            print(f"  {MC_SAMPLE}:  {after:.0f} / {before:.0f} = {eff * 100:.2f}%")
            results["mc"] = {
                "before": before,
                "after": after,
                "efficiency": eff
            }
        else:
            print(f"  {MC_SAMPLE}:  Failed to calculate")
            results["mc"] = None

    print("=" * 60 + "\n")

    # Save results to JSON
    json_path = f"{OUTPUT_DIR}/efficiency.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logging.info(f"Saved efficiency results to {json_path}")

    logging.info("Done!")


if __name__ == "__main__":
    main()

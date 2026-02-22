"""Count events in EvtTreeProducer ROOT files under progressive selection cuts.

Scans all ROOT files in SKNanoOutput/EvtTreeProducer/ and counts events
under 4 criteria: Raw, Tight ID, B-jet, and Tight+B-jet.
Uses RDataFrame for fast columnar processing.
Outputs results as markdown tables to DataAugment/dataset.md.
"""

import os
import glob
import ROOT


WORKDIR = os.environ.get("WORKDIR")
if not WORKDIR:
    raise RuntimeError("WORKDIR not set. Run 'source setup.sh' first.")

BASEDIR = os.path.join(WORKDIR, "SKNanoOutput", "EvtTreeProducer")
OUTDIR = os.path.join(WORKDIR, "ParticleNetMD", "DataAugment")

CHANNELS = ["Run1E2Mu", "Run3Mu"]
ERAS = [
    "2016preVFP", "2016postVFP", "2017", "2018",
    "2022", "2022EE", "2023", "2023BPix",
]

SIGNAL_PREFIX = "TTToHcToWAToMuMu-"
BKG_PREFIX = "Skim_TriLep_"

BKG_CATEGORIES = [
    ("nonprompt", lambda s: s.startswith("TTLL")),
    ("diboson",   lambda s: s.startswith(("WZ", "ZZ"))),
    ("ttX",       lambda s: s.startswith(("TTZ", "TTW", "tZq"))),
    ("other",     lambda s: True),
]


def classify_sample(filename):
    """Return (category, display_name) for a ROOT file."""
    name = filename.replace(".root", "")
    if name.startswith(SIGNAL_PREFIX):
        return "signal", name[len(SIGNAL_PREFIX):]
    if name.startswith(BKG_PREFIX):
        short = name[len(BKG_PREFIX):]
        for cat, matcher in BKG_CATEGORIES:
            if matcher(short):
                return cat, short
    return "other", name


# Define C++ helpers for RDataFrame filters
ROOT.gInterpreter.Declare("""
bool allTrue(const ROOT::VecOps::RVec<bool>& v) {
    for (auto x : v) { if (!x) return false; }
    return true;
}
bool anyTrue(const ROOT::VecOps::RVec<bool>& v) {
    for (auto x : v) { if (x) return true; }
    return false;
}
""")


def count_events(filepath):
    """Count events under 4 cut levels using RDataFrame."""
    rdf = ROOT.RDataFrame("Events", filepath)

    n_raw = rdf.Count()
    rdf_tight = rdf.Filter("allTrue(MuonIsTightColl) && allTrue(ElectronIsTightColl)")
    n_tight = rdf_tight.Count()
    rdf_bjet = rdf.Filter("anyTrue(JetIsBtaggedColl)")
    n_bjet = rdf_bjet.Count()
    n_both = rdf_tight.Filter("anyTrue(JetIsBtaggedColl)").Count()

    return {
        "raw": n_raw.GetValue(),
        "tight": n_tight.GetValue(),
        "bjet": n_bjet.GetValue(),
        "both": n_both.GetValue(),
    }


def scan_all():
    """Scan all channels/eras/samples and return structured results."""
    results = {}
    for channel in CHANNELS:
        results[channel] = {}
        for era in ERAS:
            era_dir = os.path.join(BASEDIR, channel, era)
            if not os.path.isdir(era_dir):
                continue
            root_files = sorted(glob.glob(os.path.join(era_dir, "*.root")))
            for fpath in root_files:
                fname = os.path.basename(fpath)
                cat, display = classify_sample(fname)
                print(f"Processing {channel}/{era}/{fname} [{cat}]...")
                counts = count_events(fpath)
                results[channel].setdefault(cat, []).append((display, era, counts))
    return results


def write_markdown(results):
    """Write results as markdown tables to DataAugment/dataset.md."""
    os.makedirs(OUTDIR, exist_ok=True)
    outpath = os.path.join(OUTDIR, "dataset.md")

    lines = []
    lines.append("# ParticleNetMD Dataset Event Counts")
    lines.append("")
    lines.append("Event counts under progressive selection cuts.")
    lines.append("Loose lepton ID preselection is applied upstream by EvtTreeProducer.")
    lines.append("")
    lines.append("| Cut level | Definition |")
    lines.append("|-----------|-----------|")
    lines.append("| **Raw** | Total entries in the tree (loose ID from EvtTreeProducer) |")
    lines.append("| **Tight** | All muons pass `MuonIsTightColl`, all electrons pass `ElectronIsTightColl` |")
    lines.append("| **Bjet** | At least one jet with `JetIsBtaggedColl == True` |")
    lines.append("| **Tight+Bjet** | Both tight ID and b-jet requirements |")
    lines.append("")

    category_order = ["signal", "nonprompt", "diboson", "ttX", "other"]
    category_labels = {
        "signal": "Signal Samples",
        "nonprompt": "Nonprompt Background (TTLL)",
        "diboson": "Diboson Background (WZ, ZZ)",
        "ttX": "ttX Background (TTZ, TTW, tZq)",
        "other": "Other Samples (DYJets, TTH, ...)",
    }

    for channel in CHANNELS:
        lines.append(f"## {channel} Channel")
        lines.append("")
        ch_data = results.get(channel, {})

        for cat in category_order:
            entries = ch_data.get(cat, [])
            if not entries:
                continue

            lines.append(f"### {category_labels[cat]}")
            lines.append("")
            lines.append("| Sample | Era | Raw | Tight | Bjet | Tight+Bjet |")
            lines.append("|--------|-----|----:|------:|-----:|-----------:|")

            era_order = {e: i for i, e in enumerate(ERAS)}
            entries.sort(key=lambda x: (x[0], era_order.get(x[1], 99)))

            for display, era, c in entries:
                lines.append(
                    f"| {display} | {era} | {c['raw']} | {c['tight']} "
                    f"| {c['bjet']} | {c['both']} |"
                )

            lines.append("")

    with open(outpath, "w") as f:
        f.write("\n".join(lines))
    print(f"\nResults written to {outpath}")


def main():
    ROOT.gROOT.SetBatch(True)
    results = scan_all()
    write_markdown(results)


if __name__ == "__main__":
    main()

#!/bin/bash
#
# runPullPlots.sh - Generate NP pull plots from FitDiagnostics output
#
# Runs diffNuisances.py on fitDiagnostics.root to produce:
#   - nuisance_pulls.{txt,root,pdf}            (all NPs)
#   - nuisance_pulls_filtered.{txt,root,pdf}   (stat bin NPs excluded)
#
# Must be run after runFitDiagnostics.sh has produced combine_output/fitdiag/.
#
# Usage:
#   ./runPullPlots.sh --era All --channel Combined --masspoint MHc130_MA90 \
#                     --method ParticleNet --binning extended --partial-unblind
#

set -e

# Default values
ERA=""
CHANNEL=""
MASSPOINT=""
METHOD="Baseline"
BINNING="extended"
PARTIAL_UNBLIND=false
UNBLIND=false
DRY_RUN=false
VERBOSE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --era)           ERA="$2";           shift 2 ;;
        --channel)       CHANNEL="$2";       shift 2 ;;
        --masspoint)     MASSPOINT="$2";     shift 2 ;;
        --method)        METHOD="$2";        shift 2 ;;
        --binning)       BINNING="$2";       shift 2 ;;
        --partial-unblind) PARTIAL_UNBLIND=true; shift ;;
        --unblind)       UNBLIND=true;       shift ;;
        --dry-run)       DRY_RUN=true;       shift ;;
        --verbose)       VERBOSE=true;       shift ;;
        -h|--help)
            echo "Usage: $0 --era ERA --channel CHANNEL --masspoint MASSPOINT [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --method METHOD         Baseline or ParticleNet [default: Baseline]"
            echo "  --binning BINNING       extended or uniform [default: extended]"
            echo "  --partial-unblind       Use partial-unblind templates"
            echo "  --unblind               Use fully unblinded templates"
            echo "  --dry-run               Print commands without executing"
            echo "  --verbose               Enable verbose logging"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Validate
if [[ -z "$ERA" || -z "$CHANNEL" || -z "$MASSPOINT" ]]; then
    echo "ERROR: --era, --channel, and --masspoint are required"
    exit 1
fi
if [[ "$UNBLIND" == true && "$PARTIAL_UNBLIND" == true ]]; then
    echo "ERROR: --unblind and --partial-unblind are mutually exclusive"
    exit 1
fi

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Resolve binning suffix
BINNING_SUFFIX="${BINNING}"
if   [[ "$UNBLIND"         == true ]]; then BINNING_SUFFIX="${BINNING}_unblind"
elif [[ "$PARTIAL_UNBLIND" == true ]]; then BINNING_SUFFIX="${BINNING}_partial_unblind"
fi

TEMPLATE_DIR="${WORKDIR}/SignalRegionStudyV2/templates/${ERA}/${CHANNEL}/${MASSPOINT}/${METHOD}/${BINNING_SUFFIX}"
FITDIAG_DIR="${TEMPLATE_DIR}/combine_output/fitdiag"

if [[ ! -d "$FITDIAG_DIR" ]]; then
    echo "ERROR: FitDiagnostics output directory not found: $FITDIAG_DIR"
    echo "  Run runFitDiagnostics.sh first."
    exit 1
fi

FITDIAG_FILE=$(ls "${FITDIAG_DIR}"/fitDiagnostics.*.root 2>/dev/null | head -1)
if [[ -z "$FITDIAG_FILE" ]]; then
    echo "ERROR: No fitDiagnostics.root file found in ${FITDIAG_DIR}"
    exit 1
fi

DIFFNUIS="${CMSSW_BASE}/src/HiggsAnalysis/CombinedLimit/test/diffNuisances.py"
if [[ ! -f "$DIFFNUIS" ]]; then
    echo "ERROR: diffNuisances.py not found at ${DIFFNUIS}"
    echo "  Make sure CMSSW environment is set up."
    exit 1
fi

log() { [[ "$VERBOSE" == true ]] && echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" || true; }
run_cmd() {
    if [[ "$DRY_RUN" == true ]]; then echo "[DRY-RUN] $1"; else log "Running: $1"; eval "$1"; fi
}

# Regex to exclude stat bin nuisances (prop_bin*, autoMCStat*)
FILTER_REGEX='^(?!prop_bin|autoMCStat).*'

# Convert ROOT canvas to multi-page PDF (~25 NPs per page)
NPS_PER_PAGE=40

convert_to_pdf() {
    local ROOT_FILE="$1"
    local PDF_FILE="$2"
    local BASENAME
    BASENAME="$(basename "$PDF_FILE")"
    root -l -b -q -e "
        gStyle->SetOptStat(0);
        gStyle->SetPadBottomMargin(0.40);
        gStyle->SetPadLeftMargin(0.05);
        gStyle->SetPadRightMargin(0.02);
        gStyle->SetPadTopMargin(0.10);

        TFile *f = TFile::Open(\"${ROOT_FILE}\");
        if (!f || f->IsZombie()) { printf(\"ERROR: cannot open %s\\n\", \"${ROOT_FILE}\"); return; }
        TCanvas *orig = (TCanvas*)f->Get(\"nuisances\");
        if (!orig) { printf(\"WARNING: 'nuisances' canvas not found\\n\"); f->Close(); return; }

        // Extract histograms and graphs from original canvas
        TH1F *hPrefit = nullptr, *hPostfit = nullptr;
        TGraphAsymmErrors *gPrefit = nullptr, *gPostfit = nullptr;
        TLegend *leg = nullptr;
        int iHist = 0, iGraph = 0;
        TIter next(orig->GetListOfPrimitives());
        TObject *obj;
        while ((obj = next())) {
            if (obj->InheritsFrom(\"TH1F\")) {
                TH1F *h = (TH1F*)obj;
                if (iHist == 0) hPrefit = h;
                else if (iHist == 1) hPostfit = h;
                iHist++;
            } else if (obj->InheritsFrom(\"TGraphAsymmErrors\")) {
                TGraphAsymmErrors *g = (TGraphAsymmErrors*)obj;
                if (iGraph == 0) gPrefit = g;
                else if (iGraph == 1) gPostfit = g;
                iGraph++;
            } else if (obj->InheritsFrom(\"TLegend\")) {
                leg = (TLegend*)obj;
            }
        }
        if (!hPrefit || !gPostfit) {
            printf(\"WARNING: expected histogram/graph not found\\n\");
            f->Close(); return;
        }

        int nTotal = hPrefit->GetNbinsX();
        int nPerPage = ${NPS_PER_PAGE};
        int nPages = (nTotal + nPerPage - 1) / nPerPage;

        // Collect bin labels
        std::vector<TString> labels(nTotal);
        for (int i = 0; i < nTotal; i++)
            labels[i] = hPrefit->GetXaxis()->GetBinLabel(i+1);

        // Collect prefit graph data (b-only)
        int nPre = gPrefit ? gPrefit->GetN() : 0;
        std::vector<double> preX(nPre), preY(nPre), preEYL(nPre), preEYH(nPre);
        for (int i = 0; i < nPre; i++) {
            preX[i] = gPrefit->GetPointX(i);
            preY[i] = gPrefit->GetPointY(i);
            preEYL[i] = gPrefit->GetErrorYlow(i);
            preEYH[i] = gPrefit->GetErrorYhigh(i);
        }

        // Collect postfit graph data (s+b)
        int nPost = gPostfit ? gPostfit->GetN() : 0;
        std::vector<double> postX(nPost), postY(nPost), postEYL(nPost), postEYH(nPost);
        for (int i = 0; i < nPost; i++) {
            postX[i] = gPostfit->GetPointX(i);
            postY[i] = gPostfit->GetPointY(i);
            postEYL[i] = gPostfit->GetErrorYlow(i);
            postEYH[i] = gPostfit->GetErrorYhigh(i);
        }

        TCanvas *c = new TCanvas(\"c\", \"NP Pulls\", 1200, 700);

        for (int page = 0; page < nPages; page++) {
            int binLo = page * nPerPage;
            int binHi = std::min(binLo + nPerPage, nTotal);
            int nBins = binHi - binLo;

            // Frame histogram for this page
            TH1F *hFrame = new TH1F(Form(\"frame_%d\", page), \"\", nBins, 0, nBins);
            for (int i = 0; i < nBins; i++)
                hFrame->GetXaxis()->SetBinLabel(i+1, labels[binLo + i]);

            hFrame->GetYaxis()->SetRangeUser(-2.5, 2.5);
            hFrame->GetYaxis()->SetTitle(\"pull (#sigma)\");
            hFrame->GetYaxis()->SetTitleSize(0.04);
            hFrame->GetYaxis()->SetLabelSize(0.035);
            hFrame->GetXaxis()->SetLabelSize(0.022);
            hFrame->GetXaxis()->LabelsOption(\"v\");
            hFrame->SetTitle(Form(\"Nuisance parameters (page %d/%d)\", page+1, nPages));
            hFrame->SetStats(0);

            // ±1σ and ±2σ bands
            TH1F *hBand1 = new TH1F(Form(\"band1_%d\", page), \"\", nBins, 0, nBins);
            TH1F *hBand2 = new TH1F(Form(\"band2_%d\", page), \"\", nBins, 0, nBins);
            for (int i = 1; i <= nBins; i++) {
                hBand1->SetBinContent(i, 0); hBand1->SetBinError(i, 1);
                hBand2->SetBinContent(i, 0); hBand2->SetBinError(i, 2);
            }
            hBand2->SetFillColor(TColor::GetColor(\"#85D1FB\")); hBand2->SetLineColor(0); hBand2->SetMarkerSize(0);
            hBand1->SetFillColor(TColor::GetColor(\"#FFDF7F\")); hBand1->SetLineColor(0); hBand1->SetMarkerSize(0);

            // Build sub-graphs for this page
            TGraphAsymmErrors *gPre = new TGraphAsymmErrors();
            TGraphAsymmErrors *gPost = new TGraphAsymmErrors();

            for (int i = 0; i < nPre; i++) {
                int origBin = (int)(preX[i]);
                if (origBin >= binLo && origBin < binHi) {
                    int n = gPre->GetN();
                    gPre->SetPoint(n, origBin - binLo + 0.5, preY[i]);
                    gPre->SetPointError(n, 0, 0, preEYL[i], preEYH[i]);
                }
            }
            for (int i = 0; i < nPost; i++) {
                int origBin = (int)(postX[i]);
                if (origBin >= binLo && origBin < binHi) {
                    int n = gPost->GetN();
                    gPost->SetPoint(n, origBin - binLo + 0.5, postY[i]);
                    gPost->SetPointError(n, 0, 0, postEYL[i], postEYH[i]);
                }
            }

            gPre->SetMarkerStyle(20); gPre->SetMarkerSize(0.8);
            gPre->SetMarkerColor(kBlack); gPre->SetLineColor(kBlack);
            gPost->SetMarkerStyle(24); gPost->SetMarkerSize(0.8);
            gPost->SetMarkerColor(kRed); gPost->SetLineColor(kRed);

            c->Clear();
            hFrame->Draw(\"AXIS\");
            hBand2->Draw(\"E2 SAME\");
            hBand1->Draw(\"E2 SAME\");
            gPre->Draw(\"P SAME\");
            gPost->Draw(\"P SAME\");
            gPad->RedrawAxis();
            TFrame *fr = gPad->GetFrame();
            fr->SetFillStyle(0);
            fr->Draw(\"SAME\");

            // Legend on first page
            if (page == 0) {
                TLegend *l = new TLegend(0.70, 0.92, 0.98, 1.0);
                l->SetNColumns(2);
                l->SetBorderSize(0);
                l->SetFillStyle(0);
                l->SetTextSize(0.028);
                l->AddEntry(gPre, \"b-only fit\", \"lp\");
                l->AddEntry(gPost, \"s+b fit\", \"lp\");
                l->Draw();
            }

            // Multi-page PDF
            if (nPages == 1) {
                c->SaveAs(\"${PDF_FILE}\");
            } else if (page == 0) {
                c->Print(Form(\"%s(\", \"${PDF_FILE}\"));
            } else if (page == nPages - 1) {
                c->Print(Form(\"%s)\", \"${PDF_FILE}\"));
            } else {
                c->Print(\"${PDF_FILE}\");
            }
        }

        printf(\"Saved: ${BASENAME} (%d pages)\\n\", nPages);
        f->Close();
    " 2>/dev/null || echo "WARNING: PDF conversion failed — check $(basename "${ROOT_FILE}") manually"
}

echo "============================================================"
echo "NP Pull Plots (diffNuisances.py)"
echo "============================================================"
echo "  Era:       ${ERA}"
echo "  Channel:   ${CHANNEL}"
echo "  Masspoint: ${MASSPOINT}"
echo "  Method:    ${METHOD}"
echo "  Binning:   ${BINNING_SUFFIX}"
echo "  Input:     ${FITDIAG_FILE}"
echo ""

# --- All nuisances ---
echo "===== Generating text pull table (all NPs) ====="
run_cmd "python3 '${DIFFNUIS}' '${FITDIAG_FILE}' \
    --all \
    -f text \
    2>/dev/null > '${FITDIAG_DIR}/nuisance_pulls.txt'"

echo "===== Generating pull plot canvas (all NPs) ====="
run_cmd "python3 '${DIFFNUIS}' '${FITDIAG_FILE}' \
    --all \
    -g '${FITDIAG_DIR}/nuisance_pulls.root' \
    2>/dev/null"

if [[ "$DRY_RUN" == false && -f "${FITDIAG_DIR}/nuisance_pulls.root" ]]; then
    echo "===== Converting canvas to PDF (all NPs) ====="
    convert_to_pdf "${FITDIAG_DIR}/nuisance_pulls.root" "${FITDIAG_DIR}/nuisance_pulls.pdf"
fi

# --- Filtered (stat bin nuisances excluded) ---
echo "===== Generating text pull table (filtered) ====="
run_cmd "python3 '${DIFFNUIS}' '${FITDIAG_FILE}' \
    --all \
    -f text \
    --regex '${FILTER_REGEX}' \
    2>/dev/null > '${FITDIAG_DIR}/nuisance_pulls_filtered.txt'"

echo "===== Generating pull plot canvas (filtered) ====="
run_cmd "python3 '${DIFFNUIS}' '${FITDIAG_FILE}' \
    --all \
    -g '${FITDIAG_DIR}/nuisance_pulls_filtered.root' \
    --regex '${FILTER_REGEX}' \
    2>/dev/null"

if [[ "$DRY_RUN" == false && -f "${FITDIAG_DIR}/nuisance_pulls_filtered.root" ]]; then
    echo "===== Converting canvas to PDF (filtered) ====="
    convert_to_pdf "${FITDIAG_DIR}/nuisance_pulls_filtered.root" "${FITDIAG_DIR}/nuisance_pulls_filtered.pdf"
fi

echo ""
echo "============================================================"
if [[ "$DRY_RUN" == false ]]; then
    echo "Output files:"
    ls -lh "${FITDIAG_DIR}/nuisance_pulls".* 2>/dev/null || true
    ls -lh "${FITDIAG_DIR}/nuisance_pulls_filtered".* 2>/dev/null || true
fi
echo "Done."
echo "============================================================"

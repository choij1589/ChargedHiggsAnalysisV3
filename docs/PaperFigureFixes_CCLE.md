# Paper figure fixes requested by the CMS language editor (B2G-25-013, draft V6 → V7)

Task: adjust the paper plotting scripts in this repository so that the regenerated PDFs satisfy the four style requests below, then regenerate the affected PDFs. The paper repository (`~/Sync/Documents/Paper/B2G-25-013`) copies them with `scripts/parsePaperFigures.sh`, which is run by the paper author afterwards. Do not edit the paper repository from here.

The requests come from the CMS Collaboration Language Editor (CCLE). The full comment list with replies is in `~/Sync/Documents/Paper/B2G-25-013/Reviews/V6/CCLE.md`; the items marked `[Pending: figure]` are the ones handled here.

## Requests

### 1. Legend "Data" entry must look exactly like the plotted points

Currently the legend shows a dot with a vertical bar only, while the points in the plot carry both vertical and horizontal bars. Either draw the legend entry with the same error-bar style as the points, or, preferably, apply request 2 so that neither has horizontal bars. Affected scripts and outputs:

| Script | Output (paper figure) |
|---|---|
| `TriLepton/python/plotPaperMass.py` (uses `TriLepton/python/paper_plotting.py`) | `TriLepton/plots/Paper/All/{SR1E2Mu,SR3Mu}/Central/pair*_mass.pdf` (Fig. 4), `TriLepton/plots/Paper/All/{ZFake1E2Mu,ZFake3Mu,TTZ2E1Mu}/Central/ZCand_mass.pdf` (Fig. 7) |
| `SignalRegionStudyV4/python/plotPaperLRModified.py` | `SignalRegionStudyV4/results/plots/paper/SR/LR_modified_MHc{160_MA85,130_MA90,100_MA95}.pdf` (Fig. 8) |
| `SignalRegionStudyV4/python/plotPaperTemplates.py` | `SignalRegionStudyV4/results/plots/paper/templates/MHc130_MA90/{prefit,postfit_s}_mass_Run{2,3}_SR{1E2Mu,3Mu}.pdf` (Figs. 9, 10) |
| `SignalRegionStudyV4/python/plotPaperPostfitSummary.py` | `SignalRegionStudyV4/results/plots/paper/Postfit/postfit_b_only.pdf` (Fig. 11) |

### 2. No horizontal error bars on data points when the bins have equal width

All of the distributions above use uniform binning, so remove the horizontal bars from the data points (ROOT: `gStyle.SetErrorX(0)` before drawing, or draw the data histogram with the `X0` option, e.g. `"PE X0"`). The ratio panels use the same data points and should be treated the same way. Check whether the shared CMS style helper (`cmsObjectDraw` or the helper in `paper_plotting.py`) sets the draw option, since the fix may belong there rather than in each script.

Explicitly requested for Fig. 7 (`ZCand_mass.pdf`) and Fig. 11 (`postfit_b_only.pdf`). Apply it to Figs. 4, 8, 9, and 10 as well so the style is uniform.

### 3. Misidentification-rate legend: "0.0" → "0"

`MeasFakeRateV4/python/plotFakeRate.py`, line 121 builds the legend labels as `f"{abseta_bins[idx]} < {eta_label} < {abseta_bins[idx+1]}"` from `abseta_bins = [0., 0.9, 1.6, 2.4]` (muon) and `[0., 0.8, 1.479, 2.5]` (electron), so the first entry prints `0.0 < |η| < 0.9`. Format the bin edges so that `0.` prints as `0` (e.g. `f"{edge:g}"`), giving `0 < |η| < 0.9`. Alternatively print the first bin as `|η| < 0.9`. Outputs: `MeasFakeRateV4/plots/2018/{electron,muon}/fakerate.pdf` (Fig. 6). The 2018 plots are the ones in the paper, but regenerate every year for consistency if the script loops over years.

### 4. "B-only Post-fit" label → "Bkg-only post-fit"

The paper uses "bkg" for background (subscript of the LR_modified definition) and "background-only fit" in prose, so "B-only" is inconsistent. Change:

- `SignalRegionStudyV4/python/plotPaperPostfitSummary.py:87`, `FIT_STAGE_LABEL = "B-only Post-fit"` → `"Bkg-only post-fit"`.
- `SignalRegionStudyV4/python/plotPaperTemplates.py:80`, `STAGE_LABELS`: `"b": "B-only Post-fit"` → `"Bkg-only post-fit"`, and for consistency `"prefit": "Pre-fit"` and `"s": "S+B Post-fit"` → `"S+B post-fit"` (lower-case "post-fit" as in the paper text). Only the pre-fit and S+B post-fit template plots are in the paper, so the "b" label there is cosmetic, but keep the three consistent.

## Constraints that must still hold after regeneration

- No "Preliminary" label anywhere. The figures were relabeled for the journal submission.
- Luminosity labels stay `138 fb⁻¹ (13 TeV)`, `62 fb⁻¹ (13.6 TeV)`, or the combined form `138 fb⁻¹ (13 TeV) + 62 fb⁻¹ (13.6 TeV)`. Never 137.6, 137.9, or 62.4.
- Output must be vector PDF. Do not change axis ranges, binning, colors, signal normalizations (30 fb in Fig. 4 and Fig. 8, 5 fb in Figs. 9 and 10), or the physics content. Only the legend marker, horizontal bars, the `0.0` label, and the fit-stage label change.
- Do not touch the limit plots (`plotPaperLimit.py`) or the ROC plots (`ParticleNetMD/python/plotPaperROCs.py`). They were accepted as regenerated.

## How to verify

1. Text check on each regenerated PDF:
   ```bash
   pdftotext file.pdf - | grep -c -i preliminary    # must print 0
   pdftotext file.pdf - | grep -E 'B-only|0\.0 <'   # must print nothing
   ```
2. Visual check: `pdftoppm -png -r 80 -singlefile file.pdf out` and confirm that the data points and the legend "Data" entry are identical and have no horizontal bars.
3. List the regenerated files with their timestamps so the paper author can re-run `scripts/parsePaperFigures.sh` in the paper repository, which expects exactly the paths in the table above.

## Report back

Finish with a short list: which scripts changed and how, which PDFs were regenerated, and anything that could not be done, so the paper-side replies in `Reviews/V6/CCLE.md` can be switched from `[Pending: figure]` to `[Done]`.

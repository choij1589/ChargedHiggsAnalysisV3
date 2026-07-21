# Systematic Uncertainties in TriLepton

How `python/sampleBreakdown.py` (yield tables) and `python/plot.py` /
`python/paper_plotting.py` (data-MC plots) build their uncertainties.

Both go through `CorrelatedTotalBuilder` (`Common/Tools/HistoUtils.py:465`), so the
two agree by construction. Numbers here are prefit only — the full correlation model
belongs to the Combine step.

## Where sources come from

| origin | provides |
|---|---|
| `configs/systematics.json` | shape sources, per Run and lepton channel, as `{name: [Up, Down]}` |
| `Common/Data/KFactors.json` | `kFactor` (scale) and `xsecErr` (rate) per sample |
| `Common/Data/ConvSF.json` | conversion SF `central` (scale) and `total` (rate), per era |
| `Common/Data/FakeNorm.json` | nonprompt normalization, per flag and era |

## Shape sources

Loaded as `{channel}/{name}_Up|Down/{histkey}` by `load_systematic_variations`
(`HistoUtils.py:180`), scaled alongside the central by K-factor then ConvSF
(`utils.scale_with_variations`), and enveloped once after summing.

Applied to **prompt MC only** — `conv`, `ttX`, `diboson`, `others`. Nonprompt comes
from `MatrixAnalyzer` and has no variations; data and signal carry statistical errors
only.

```
source                Run2                    Run3
                      1E2Mu  2E1Mu  3Mu       1E2Mu  2E1Mu  3Mu
BtagSF_HFcorr           x      x     x          x      x     x
BtagSF_HFuncorr         x      x     x          x      x     x
BtagSF_LFcorr           x      x     x          x      x     x
BtagSF_LFuncorr         x      x     x          x      x     x
ElectronEn              x      x     x          x      x     x
ElectronRes             x      x     x          x      x     x
JetEn                   x      x     x          x      x     x
JetRes                  x      x     x          x      x     x
MuonEn                  x      x     x          x      x     x
MuonIDSF                x      x     x          x      x     x
PileupReweight          x      x     x          x      x     x
UnclusteredEn           x      x     x          x      x     x
ElectronIDSF            x      x     -          x      x     -    no electron in 3Mu
EMuTrigSF               x      x     -          x      x     -
DblMuTrigSF             -      -     x          -      -     x
L1Prefire               x      x     x          -      -     -    Run2 only
PileupJetIDSF           x      x     x          -      -     -    Run2 only
WZNjetsSF               -      -     -          x      x     x    Run3 only
```

### Missing-source substitution

A contribution that lacks a source adds its **central** histogram to that source's
up/down sum (`HistoUtils.py:554`). Without it the summed variation would silently drop
those processes and the envelope would blow up.

This is what makes cross-run aggregation work. `WZNjetsSF` exists only in Run3, so in
`--era All` the Run2 contributions sit at central inside the Run3 envelope instead of
vanishing from it. The same mechanism keeps nonprompt — which has no variations at all
— from distorting the background total.

## Rate sources

| source | value | processes | corr. samples | corr. eras |
|---|---|---|---|---|
| `nonprompt_rate` | `FakeNorm.json`, per era | data streams via `MatrixAnalyzer` | yes | no |
| `conv_rate` | `ConvSF.json` `total`, per era | `conv` category | yes | no |
| `xsec_<sample>` | `KFactors.json` `xsecErr` | samples present in `KFactors.json` | no | yes |
| `others_xsec` | flat 0.50 | the 12 `others` samples | no | yes |

The two axes are independent, and the split is by **origin**:

- A **theory prior** (a cross section) is the same number every year, so it correlates
  across eras but not across unrelated processes — a datacard writes these as separate
  per-process `lnN`.
- A **measured normalization** (ConvSF, FakeNorm) is the opposite: shared by every
  sample it scales, but re-measured each era.

`others_xsec` is flat 50% because none of `tHq`, `tHW`, `tWZ_*`, `WWW`, `WWZ`, `WZZ`,
`ZZZ`, `TTTT`, `VBFHToZZTo4L`, `ggHToZZTo4L` appears in `KFactors.json` — they would
otherwise carry no theory normalization. There is no overlap with `xsec_<sample>` in
any era or channel.

`2E1Mu` has no dedicated ConvSF measurement, so `conv_rate` falls back to scale 1.0
± 20% (`sampleBreakdown.py:260`).

### WZ normalization

Handled differently per run, and **there is no flat `WZ_rate` source**:

- **Run2** — a cross-section error, already carried by `xsec_WZTo3LNu_amcatnlo`
  (`KFactors.json`, `xsecErr` 1.075). `Common/Data/WZNjSF.json` has no Run2 entry.
- **Run3** — the measured WZ Njet reweighting, carried by the `WZNjetsSF` shape
  variation. WZ has no `KFactors.json` entry in Run3.

`ZZTo4L_powheg` is **not** a WZNjSF target — `python/measWZNjSF.py` treats it as a
subtracted background — and carries its own `xsecErr` (6.4%) in both runs.

`--exclude WZSF` drops the `WZNjetsSF` variation in `sampleBreakdown.py`; in `plot.py`
it additionally switches to the `_RunNoWZSF` inputs.

## Correlation model

| axis | treatment |
|---|---|
| across sources | quadrature — **decorrelated**, one independent nuisance each |
| across processes, within a source | shape: correlated. rate: per `correlate_samples` |
| across eras, within a source | shape: correlated. rate: per `correlate_eras` |
| across bins, within a source | yields: correlated (integral envelope). Plot band: per-bin |
| statistical vs systematic | quadrature |
| statistical across samples and eras | quadrature, via `TH1::Add` — genuinely independent |

Correlation is applied by summing the shifted histograms first and enveloping against
the summed central, so it falls out of `TH1::Add` rather than needing a covariance
matrix. Bin errors must therefore stay **pure statistical** on the way in; anything
folded into them would be double-counted.

## Envelope conventions

`sampleBreakdown.py` reports an integrated yield and uses an **integral-based**
envelope (`HistoUtils.py:623`):

```
envelope = max(|Integral(up) - Integral(central)|, |Integral(down) - Integral(central)|)
```

Plot bands draw per bin and use a **per-bin** envelope (`HistoUtils.py:597`).

The distinction matters. Summing per-bin envelopes in quadrature to get an integrated
number both invents uncertainty for shape-only sources and understates coherent ones.
For `TTZToLLNuNu` in 2018 `SR1E2Mu` (`pair/mass`, 87.8 events):

| source | integral shift | per-bin quadrature |
|---|---|---|
| `MuonEn` | 0.007 | 0.463 |
| `BtagSF_HFcorr` | 1.684 | 0.444 |

Muon scale migrates events between mass bins without changing the yield; a b-tag
reweighting changes the yield coherently.

## Differences between the two scripts

| | `sampleBreakdown.py` | `plot.py` / `paper_plotting.py` |
|---|---|---|
| envelope | integral | per-bin |
| negative bins | kept | zeroed by `clip_negative_bins` |
| `--exclude` | `Syst`, `ConvSF`, `WZSF` | `ConvSF`, `WZSF` |

Negative-bin clipping is the only remaining source of disagreement between them. It is
needed to avoid `THStack::BuildStack` warnings from negatively-weighted MC and the
matrix-method nonprompt estimate, but it biases yields upward — 544.29 vs 565.63 events
for 2018 `SR1E2Mu`, mostly `DYJets` and nonprompt. With clipping disabled the two paths
agree exactly, source by source.

## Plot band plumbing

`ComparisonCanvas` (`Common/Tools/plotter.py:343`) accepts an optional `total_syst`
histogram. It supplies **bin errors only** — contents still come from the stack, so the
band cannot drift from the histograms actually drawn. It is routed through the same
binning and overflow handling as the stack, which matters for the adaptive binning in
`paper_plotting.py`. The argument defaults to `None`, so the other call sites across
`DiLepton/`, `MeasFakeRate*/` and `ExampleRun/` are unaffected.

## Known limitations

- **Rebinning merges bin errors in quadrature**, understating a source correlated
  across the merged bins. Affects rebinned and adaptive plots only, not the yields in
  `results/`.
- **Sign information is discarded.** The envelope takes an absolute value, so sources
  pushing the same direction are understated and opposing ones overstated.
- **`BtagSF_*corr` and `*uncorr` receive identical era treatment** despite the names
  encoding a difference. Combine resolves it; the prefit table does not.
- **`FakeNorm.json` has no `Run2E1Mu` block**, so every `TTZ2E1Mu` run falls back to a
  hardcoded 30% with a warning (`sampleBreakdown.py:251`), rather than failing fast.
- **`plotNMinusOne.py`, `plotCompareHEMVeto.py` and `DiLepton/`** still fold
  systematics into per-sample bin errors via `HistoUtils.calculate_systematics`
  (`HistoUtils.py:113`) and so retain the per-process quadrature treatment.

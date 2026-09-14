# Paper mass plots

`python/plotPaperMass.py` renders the Run 2 + Run 3 combined mass distributions used
in the paper and in slides. The plotting itself lives in `python/paper_plotting.py`;
the entry point only picks which figures to make.

```bash
python python/plotPaperMass.py                    # all 10 plots + both legend panels
python python/plotPaperMass.py --plot sr1-pair    # one figure
python python/plotPaperMass.py --plot legend      # the standalone legend panels only
python python/plotPaperMass.py --dry-run          # print the output paths, render nothing
```

## Output

Default root is `$WORKDIR/TriLepton/plots/Paper` (`--output-root` overrides it).

```
plots/Paper/All/{channel}/Central/{histkey with / -> _}.pdf
plots/Paper/All/legend.pdf             # shared legend, with the signal block
plots/Paper/All/legend_nosignal.pdf    # shared legend, control-region variant
```

Ten figures: `SR1E2Mu/pair_mass`, `SR3Mu/pair_lowM_mass`, `SR3Mu/pair_highM_mass`,
and `ZCand_mass` for `ZFake1E2Mu`, `ZFake3Mu`, `ZG1E2Mu`, `ZG3Mu`, `WZ1E2Mu`,
`WZ3Mu`, `TTZ2E1Mu`.

The legend panels are written on every `--plot all` run. Each plot already carries
its own legend, so the panels exist for layouts that drop them — slides, and the
2x2 paper arrangement reachable with `--no-legends`.

## Layout

Everything below is a module constant in `paper_plotting.py`; the numbers are NDC
of the upper pad.

### CMS block

`cmsstyle` hardcodes the in-frame offsets of "CMS"/"Preliminary" at 3.5% of the
frame from its top-left corner, which at these panel sizes puts the text on the
axis ticks. The paper plots therefore place the block themselves via
`BaseCanvas._configure_cms_label()` in `Common/Tools/plotter.py`: when a config
supplies `cmsPosX`/`cmsPosY`, cmsstyle is told to draw neither string and
`_draw_cms_label()` draws both, restoring cmsstyle's globals afterwards so the
override cannot leak into the next canvas in the same process.

| | value |
|---|---|
| `CMS_LABEL_POS` | `(0.20, 0.865)` |
| `CMS_LABEL_SIZE` | `0.070` ("Preliminary" follows at 0.76x, as in cmsstyle) |
| `CHANNEL_POS` | `(0.20, 0.665)` |
| `CHANNEL_SIZE` | `0.063` |

### Legends

Both legends sit in the right-hand column, data and backgrounds above the signal
mass points:

| | value |
|---|---|
| `BKG_LEGEND` | `(0.650, 0.545, 0.970, 0.870)`, text 0.036 |
| `SIGNAL_LEGEND` | `(0.650, 0.270, 0.970, 0.515)`, text 0.034 |
| `SIGNAL_LEGEND_HEADER` | `(m_{H^{+}}, m_{A}) [GeV]` |

That column covers the top end of the mass range, which is empty in every one of
these distributions, so the legends cost no headroom. A middle column was tried
and rejected: it sits over the Z peak and forces the axis about 30% taller.

The header carries the symbols and the unit, so each signal row is just the mass
pair — `(130, 90)` — which is what lets four mass points fit one narrow column
(`format_signal_label()`).

### Vertical scale

`y_headroom` (`--y-headroom`) is the linear-scale multiplier above the tallest
bin; default **1.35**. Control regions get `y_headroom * CR_Y_HEADROOM_SCALE`
(= 1.62): their captions are long enough that `Z+nonprompt CR` reaches the Z peak,
which the short `SR` caption does not.

Note that `ComparisonCanvas.update_y_scale()` scales on the stack and the signals
only, not on data, so a data point above the stack maximum eats into the headroom.
SR1E2Mu is the case that matters (225 observed against a 198-event stack).

## Signals

Overlaid on the signal regions only, scaled by `--signal-scale` (default 2).

Drawn as **unfilled solid outlines**, width 3. A translucent fill washes out
against the stacked colours, and dashed lines break up at paper size. Hues are
disjoint from `BKG_COLORS` (blue / orange / red / grey / purple) so a signal is
never read as one of the fills it crosses; the mass point that peaks on the Z
peak, where the stack is busiest, gets black.

| mass point | colour |
|---|---|
| `MHc70_MA15` | `#d81b60` magenta |
| `MHc100_MA60` | `#2ca02c` green |
| `MHc130_MA90` | `#000000` black |
| `MHc160_MA155` | `#0099b4` teal |

## Luminosity label

`138 fb^-1 (13 TeV) + 62 fb^-1 (13.6 TeV)` — the rounded per-period values from
`LumiInfo`, i.e. what CMS quotes for a single period.

Note this differs from the other paper scripts in the repository
(`SignalRegionStudyV3/V4` `plotPaper*.py`, `plotLimits*.py`, `plotBreakdown.py`,
`plotGoFPValues.py`, `plotLEE.py`, `plotPostfitSummary.py`,
`plotParticleNetScore.py`), which still use `LumiInfoExact` and print
`137.6 / 62.4`.

## Systematics

Identical treatment to `plot.py`: contributions accumulate into
`HistoUtils.CorrelatedTotalBuilder` so a source shared between processes stays one
nuisance, and the resulting histogram supplies the band's bin errors while the
contents come from the stack. See `docs/systematics.md`.

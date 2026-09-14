# Leave-one-out closure of the shape-delta parametrisation

Instruction for a coding agent (Claude Code). Work from the module root
(`SignalRegionStudyV4/`); run `source setup.sh` first. Everything below is
JSON + numpy on a login node — no ntuples, no condor, runtime is minutes.

## Why this test exists

Shape systematics are attached to the parametric signal template through
three dimensionless deltas per (era|channel, systematic, direction) —
dm, dsig, dN — measured at every simulated mass point (stage 1,
`measInterpShapeDeltas.py`) and parametrised in mA (stage 2,
`fitInterpShapeDeltas.py`). In the production run every simulated point
was used as a fit anchor (`meta.held_out_ma == []` in every
`fits/MHc*/shape_deltas/delta_model.json`), so stage 2's built-in
held-out closure is empty, and the only genuine interpolation test of
the delta transfer is the single-point end-to-end study at
MHc160_MA90 (docs/interpolation/EXPERIMENTS.md, V1: median |Δ| 1.4e-7 /
7.3e-6 / 1.7e-5 for dm/dsig/dN over 6576 series, worst JES dN −6.3%
transferred vs −4.5% measured).

This test extends that to the full grid: a leave-one-out (LOO) closure
of the delta model over all simulated mass points, mirroring the LOO
protocol already used for the shape and yield interpolation
uncertainties. The AN (Section 8.6) will quote the resulting medians and
worst cases.

## Inputs (read-only)

- `fits/MHc{70,85,100,115,130,145,160}/shape_deltas/delta_model.json`
  - `meta`: `fit_ma` (anchor mA list), `orders` ([0, 1] — up-only F-test
    ladder), `err_floor` per quantity, `core_nsigma`.
  - `model[era|channel].systs[syst][direction][quantity]`: fitted
    `coeffs`, `order`, `chi2`, `ndf`, `err_scale`, and — crucially —
    `points`: the list of `[mA, measured_value, error]` the fit used.
    The LOO therefore needs only this file; `shape_deltas.json` is the
    same information keyed by mass point and is useful only for
    cross-checks.
  - `model[era|channel].pdf_members`: per-replica PDF series with the
    same structure. Include them behind a `--pdf-members` flag,
    summarised separately; default off (100 replicas × everything is
    noise-dominated and the AN quotes the named systematics).

## What to implement

`python/closInterpShapeDeltasLOO.py`, in the style of the existing
`clos*`/`fitInterp*` scripts. Before writing code, read
`python/fitInterpShapeDeltas.py` end to end and REUSE its fitting
machinery (`select_order`, `weighted_polyfit` — imported there from
`fitInterpPolynomials`) so the LOO refits are bit-compatible with
production. Do not re-implement the F-test or the error handling.

Protocol, per mHc and per series (era|channel key, systematic,
direction, quantity ∈ {dm, dsig, dN}):

1. Take the series' `points` from `delta_model.json`.
2. For each anchor point i: drop it, refit with the SAME procedure as
   production — same up-only order ladder `meta.orders`, same
   `err_floor`, and the same two-pass error rescaling to the residual
   RMS of the first pass (this is inside `fitInterpShapeDeltas.py`;
   factor the production fit into a reusable function if needed rather
   than copying logic).
3. Evaluate the refitted polynomial at mA_i; record
   `resid = predicted − measured` (dimensionless, same units as the
   delta itself).
4. Edge cases: a series whose remaining points cannot support order 1
   falls back to order 0 (the ladder does this naturally); skip a
   series only if fewer than 2 points remain, and count skips.

## Sanity gates (fail loudly, in order)

- **Gate A — reproduction.** With no point dropped, the refit must
  reproduce the stored production fit for every series: same selected
  `order`, `coeffs` equal within 1e-9 relative (or absolute 1e-12 for
  near-zero coefficients). If this fails, the fit procedure was not
  reused faithfully — stop and fix before trusting any LOO number.
- **Gate B — scale.** Global medians of |resid| per quantity should be
  of the same order as the V1 single-point numbers above. Order-of-
  magnitude disagreement means a bug, not a discovery.

## Outputs

- Per mHc: `fits/MHc{X}/shape_deltas/loo_closure.json` with `meta`
  (command, date, gates passed, skip count) and per-series residual
  lists.
- Printed global summary (and `fits/loo_closure_summary.json`):
  - per quantity (dm, dsig, dN): median |resid|, 95th percentile,
    worst value with its full identity (mHc, mA, era|channel,
    systematic, direction);
  - the worst 10 series overall, as a table;
  - the same summary restricted to the kinematic-tree systematics
    (JES/JER/unclustered — those with event migration; identify them
    from the systematic names), since V1 showed the worst closure
    there.

## Afterwards

- Append a numbered section to `docs/interpolation/EXPERIMENTS.md`
  describing setup, gates, and the summary table, in the house style
  (claim in the header, numbers in the body).
- Report the medians and worst cases back for the AN: Section 8.6 will
  say the transfer is validated by LOO closure over all simulated mass
  points and quote median and worst |resid| per quantity, with the V1
  end-to-end test as the corroborating datacard-level check.

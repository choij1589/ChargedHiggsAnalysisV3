# SAMPLES.md — Preprocessed Sample Layout and Production

Deep dive on the preprocessing step: what the preprocessed files contain,
how the shared layout works, and how production jobs are organized.
Summary-level usage lives in `CLAUDE.md` (Procedure 1).

## File Content

Each preprocessed `.root` file holds one TTree per systematic variation —
`Central` plus `{syst}_Up`/`{syst}_Down` trees named after
`configs/systematics.{era}.json` (signal additionally carries theory
variations: PDF members, scale variations). Every tree has double branches:

| Branch | Meaning |
|---|---|
| `mass` | The dimuon mass of the *selected* pairing (see pairing rule) |
| `pT` | The pT of the same selected pairing (must follow `mass`) |
| `mass1`, `mass2` | Both pairings, always stored |
| `weight` | Event weight × normalization (see Weights) |
| `score_{mp}_{signal,nonprompt,diboson,ttZ}` | ParticleNet scores — per-masspoint dirs only |

No selection cuts are applied at preprocess time (output entries = input
skim entries); mass windows and score cuts happen at template time.

## The SR3Mu Pairing Rule

SR3Mu events have two opposite-sign dimuon pairings. The stored `mass`/`pT`
pick one:

- `highM` (higher-mass pairing) **iff `mHc >= 100 && mA >= 60`**
- `lowM` (lower-mass pairing) otherwise

This is a 2D rule, not an mA threshold: MHc160_MA15 is `lowM`,
MHc70_MA60 is `lowM`, MHc100_MA60 is `highM`. SR1E2Mu (and TTZ2E1Mu)
always store `mass1`. Defined in `srspaths.pairing_variant()` (paths),
`preprocess.pairing_for()` (production), `env.sh:srs_pairing_variant()`
(condor wrappers) — keep the three in sync.

## Layout

```
samples/{era}/SR1E2Mu/            shared: bkg + nonprompt + data + ALL signals
samples/{era}/SR3Mu_lowM/         shared, low-pairing variant  + ALL signals
samples/{era}/SR3Mu_highM/        shared, high-pairing variant + ALL signals
samples/{era}/{channel}/{mp}/     ParticleNet per-masspoint (incl. TTZ2E1Mu)
```

- Backgrounds, nonprompt and data are **mass-independent** given the
  pairing, so they are produced once per (era, variant) — not per mass
  point (a ~78x duplication in the per-masspoint scheme).
- Signals live inside the shared dirs as `{masspoint}.root`. SR3Mu signals
  are stored in **both** variants: the template chain reads the point's own
  variant, the other exists for the interpolation study.
- ParticleNet needs per-masspoint dirs regardless: the score branches are
  per-masspoint and the inputs come from `_MHc{X}..._NoHistMode` skims.
  (Standard and NoHistMode skims carry identical events — verified
  empirically by the reproduction test for MHc130_MA90. **Known
  exceptions, found 2026-08-07**: 5 MHc145 NoHistMode files are
  deficient — missing 30–75% of the standard skim's `Events_Central`
  entries — Run1E2Mu: MA85/2016postVFP (−69%), MA90/2016preVFP (−30%),
  MA90/2017 (−75%), MA95/2016preVFP (−55%); Run3Mu: MA85/2016preVFP
  (−46%). Run3 and MA92 are clean. ParticleNet per-masspoint samples for
  those (point, era) combinations under-count signal MC until the
  NoHistMode skims are re-produced; the shared-signal dirs use standard
  skims and are unaffected.)
- An **interpolated mass point** therefore needs only a signal file dropped
  into the existing shared dirs — no background preprocessing at all.

## Weights

Applied per entry, in this exact operation order (floating-point
multiplication is not associative; the order is part of the reproduction
contract): `((weight * scale) [/ 3.0 signal] [* convSF conversion]) * kfactor`.

- Signal: `/ 3.0` normalizes the cross-section to the 5 fb reference.
- Conversion: `convSF` from `Common/Data/ConvSF.json` (era/channel).
- K-factors: `Common/Data/KFactors.json`, exact sample-name matches only.
- Data and nonprompt: weight passes through unchanged.

## Production Modes (`python/preprocess.py`)

| Mode | Inputs | Output |
|---|---|---|
| `--shared-backgrounds [--pairing lowM\|highM]` | standard skims | shared dir: backgrounds, nonprompt, data |
| `--shared-signal --masspoint MP` | standard skims | `{MP}.root` into the shared dir(s); SR3Mu writes both variants |
| `--masspoint MP` (default; ParticleNet-trained only) | `_MHc{X}_NoHistMode` skims | full per-masspoint dir with scores |

Implementation: RDataFrame `Define` + `Snapshot` (vectorized). Zero-entry
input trees get an explicit schema-only TTree (RDF Snapshot would write a
branchless tree that downstream readers cannot Filter on).

## Batch Production (`automize/preprocess.sh`)

```bash
./automize/preprocess.sh                      # backgrounds + all mass points
./automize/preprocess.sh --backgrounds-only   # 24 shared-background nodes only
./automize/preprocess.sh --masspoint MP --skip-backgrounds   # one point's signals
```

- Shared-background DAG: 8 eras × {SR1E2Mu, SR3Mu:lowM, SR3Mu:highM} =
  24 nodes, run once for the whole analysis.
- Per-masspoint DAG: 16 shared-signal nodes; ParticleNet-trained points add
  24 per-masspoint nodes (SR1E2Mu, SR3Mu, TTZ2E1Mu × 8 eras).
- The wrapper (`scripts/preprocess_wrapper.sh`) mirrors whatever the job
  produced onto pnfs via xrdcp — mode-agnostic.
- A grid proxy must be valid at submission (`voms-proxy-init --voms=cms`).

## Integrity

Per the pnfs silent-loss history: after any (re)production, verify by
opening every file — `compareToV3.py --stage samples` does this (per-tree
entry counts and branch sets for every tree of every file).

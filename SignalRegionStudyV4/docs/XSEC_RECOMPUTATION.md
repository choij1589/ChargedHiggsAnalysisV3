# Recomputation of the sigma_sig limits (charge-conjugation factor)

Instruction for a coding agent (Claude Code). Work from the module root
(`SignalRegionStudyV4/`); run `source setup.sh` first. This is a
conversion-level fix — no Combine jobs are rerun; only the r-to-sigma
conversion, its JSONs, and the xsec plots are regenerated.

## The bug

Normalisation convention of the analysis: `r = 1` corresponds to a
visible cross section of 5 fb, defined as sigma_sig x 0.5456, where
0.5456 removes the both-W-hadronic decays and sigma_sig is the AN's
Eq. (sigma_sig) quantity,

    sigma_sig = 2 x sigma(pp -> ttbar) x B_sig,

already inclusive of charge conjugation (t -> H+b or tbar -> H-bbar).

In `python/collectLimits.py`:

```python
BR_TTBAR_TO_LEPTON = 2 * 0.5456
if mode == "BR":
    return r * REFERENCE_XSEC / TTBAR_XEC_13TEV / BR_TTBAR_TO_LEPTON   # CORRECT
if mode == "xsec":
    return r * REFERENCE_XSEC / BR_TTBAR_TO_LEPTON                     # WRONG
```

The BR mode is correct: B_sig = r*5 / (0.5456 * 2 * sigma_ttbar). The
xsec mode divides the charge-conjugation 2 out a SECOND time, so every
sigma_sig output to date equals sigma_ttbar * B_sig = sigma_sig / 2.
(The in-code comment "equals B_sig * sigma_ttbar" describes the bug
faithfully.) Verified against the AN-verification pass of 2026-08-20:
`results/json/xsec/...` / `results/json/BR/...` = 833.9e3 exactly, i.e.
sigma_ttbar, where the AN equation requires 2 x sigma_ttbar.

## The fix

In `python/collectLimits.py`, correct the xsec branch to divide by the
W-decay factor only:

```python
BR_WW_NONHADRONIC = 0.5456      # non-hadronic decay of the two W bosons
BR_TTBAR_TO_LEPTON = 2 * BR_WW_NONHADRONIC  # keep for BR mode

if mode == "xsec":
    return r * REFERENCE_XSEC / BR_WW_NONHADRONIC  # fb; equals 2 * sigma_ttbar * B_sig
```

Do NOT touch the BR branch. Update the xsec comment so it states the
Eq. (sigma_sig) identity, and update the `--mode` help string
("sigma(pp->ttbar) x B_sig" is now wrong twice over — it should say
"sigma_sig = 2 sigma(ttbar) B_sig in fb").

## Regeneration

1. Recreate every xsec-mode JSON under `results/json/xsec/`
   ({All, Run2, Run3} x {Combined, SR1E2Mu, SR3Mu} x
   {Baseline, ParticleNet} x {interp-signal, and mc-signal where such
   files exist today). Mirror the exact file set currently present —
   check `docs/REPRODUCTION.md` and the collectLimits/plot command
   history there for the invocations used in production.
2. Regenerate every plot under `results/plots/xsec/` with
   `plotLimits.py` and `plotLimits2D.py` (per-mHc 1D panels used by the
   AN main text and per-channel appendix figures, plus the 2D maps).
   The y/z-axis labels already say [fb] and stay unchanged.
3. Leave `results/json/BR/` and `results/plots/BR/` strictly untouched.

## Gates (fail loudly)

- **Gate A** — BR outputs unchanged: `git status` (or a checksum before/
  after) shows no modification under `results/json/BR/`.
- **Gate B** — pure factor 2: every value in every regenerated xsec JSON
  equals exactly 2x its previous value (compare against the pre-fix
  files; same keys, same point count: 2467 Baseline + 150 ParticleNet
  for the All/Combined interp-signal pair).
- **Gate C** — identity: per point, xsec/BR = 2 * 833.9e3 fb = 1.6678e6
  fb to machine precision.

## Report back (for the AN)

Quote the recomputed sigma_sig ranges over the interp-signal scan from
the regenerated All/Combined JSONs, split as in the AN: mA < 82.5 GeV
(Baseline), 82.5 <= mA <= 97.5 GeV (ParticleNet), mA > 97.5 GeV
(Baseline); min-max of exp0 and obs in each region, in fb. Expected
result (from the verified B_sig ranges x 2 sigma_ttbar): below
exp 1.1--4.0 / obs 0.9--6.1; on-Z exp 1.8--8.5 / obs 1.4--11.4; above
exp 2.2--4.1 / obs 1.3--5.8. The AN then needs Result.tex line ~77
updated to the recomputed ranges and a rebuild to pick up the new
figures; the AN-side edit is handled in the AN review session, not
here. Also append a dated note to `docs/REPRODUCTION.md` recording the
conversion fix and the regeneration.

# HybridNew Plotting Scripts

## Overview

These scripts visualize results from the HybridNew method for computing upper limits on signal strength.

---

## 1. plotHybridNewGrid.py

**Purpose:** Plot CLs vs signal strength (r) curve to determine the 95% CL upper limit.

### Usage
```bash
# Basic usage (auto-detects r values from individual toy files)
python plotHybridNewGrid.py hybridnew_grid.root -o cls_vs_r.pdf

# With explicit r values
python plotHybridNewGrid.py hybridnew_grid.root --rvalues 0.0,0.2,0.4,0.6,0.8,1.0 --nseeds 10 -o cls_vs_r.pdf
```

### What it shows
- **X-axis:** Signal strength r (r=1 means SM cross-section)
- **Y-axis:** CLs value
- **Red dashed line:** 95% CL threshold (CLs = 0.05)
- **Green line:** Excluded region boundary

### Interpretation
- CLs ≈ 1: Data consistent with signal hypothesis
- CLs < 0.05: Signal hypothesis excluded at 95% CL
- The **95% CL upper limit** is the r value where CLs = 0.05

---

## 2. plotTestStatDist.py

**Purpose:** Visualize test statistic distributions to understand how CLs is calculated.

### Usage
```bash
# Single toy file
python plotTestStatDist.py higgsCombine.r0.2000.seed1000.HybridNew.mH120.1000.root -o test_stat.pdf

# Merge all seeds for a given r value
python plotTestStatDist.py hybridnew_grid.root --r 0.2 -o test_stat_r0.2.pdf
```

### What it shows
- **Blue histogram:** f(q | B) - test statistic distribution under background-only hypothesis
- **Red histogram:** f(q | S+B) - test statistic distribution under signal+background hypothesis
- **Black arrow:** q_obs - observed test statistic from data

### CLs Calculation (shown on plot)

```
CLs+b = P(q ≥ q_obs | S+B) = integral of red histogram right of arrow
CLb   = P(q ≥ q_obs | B)   = integral of blue histogram right of arrow
CLs   = CLs+b / CLb
```

### Physical Interpretation

| Scenario | Meaning |
|----------|---------|
| q_obs in bulk of blue | Data looks like background-only |
| q_obs in bulk of red | Data looks like signal+background |
| CLs+b small | Observed data unlikely if signal exists |
| CLb small | Observed data unlikely if no signal |
| CLs < 0.05 | Exclude signal at 95% CL |

---

## The CLs Method

### Why CLs instead of CLs+b?

Pure CLs+b can give spurious exclusions when the experiment has no sensitivity:
- If background fluctuates down, q_obs becomes large
- This makes CLs+b small even for signals we can't detect
- Dividing by CLb corrects for this: if CLb is also small, CLs stays large

### Test Statistic q

The test statistic is the **profile likelihood ratio**:
```
q_μ = -2 ln[ L(data | μ, θ̂_μ) / L(data | μ̂, θ̂) ]
```
Where:
- μ = signal strength being tested
- θ = nuisance parameters
- θ̂_μ = best-fit nuisances for fixed μ
- μ̂, θ̂ = global best-fit values

Higher q means data is less compatible with the signal hypothesis μ.

---

## Workflow

```
1. Generate toys at each r value
   └── higgsCombine.r{value}.seed{N}.HybridNew.*.root

2. Merge toys
   └── hadd hybridnew_grid.root higgsCombine.r*.root

3. Extract limits
   └── combine --readHybridResults → CLs for each quantile

4. Visualize
   ├── plotHybridNewGrid.py → CLs vs r curve
   └── plotTestStatDist.py  → Test statistic distributions
```

---

## Requirements

- ROOT with RooStats (for plotTestStatDist.py)
- numpy
- CMSSW environment recommended for full functionality

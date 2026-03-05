# ParticleNetMD Dataset Event Counts

Event counts under progressive selection cuts.
Loose lepton ID preselection is applied upstream by EvtTreeProducer.

| Cut level | Definition |
|-----------|-----------|
| **Raw** | Total entries in the tree (loose ID from EvtTreeProducer) |
| **Tight** | All muons pass `MuonIsTightColl`, all electrons pass `ElectronIsTightColl` |
| **Bjet** | At least one jet with `JetIsBtaggedColl == True` |
| **Tight+Bjet** | Both tight ID and b-jet requirements |

## Training Budget

Training target: **40,000 events per fold per class** (`max_events_per_fold_per_class` in SglConfig.json), 5 folds.
Run2/Run3 ratio per fold: ~34,000 / ~16,000 (luminosity ratio 138/62 ~ 2.2:1).
Balanced subprocess counts within each category.

> Numbers computed by `scripts/compute_training_budget.py`, output in `training_budget.json`.

### Augmentation Methods

| Class | Event source | Selection | Event weight | Reference |
|-------|-------------|-----------|--------------|-----------|
| Signal | TTToHcToWAToMuMu-* | Tight+Bjet | genW x puW x prefireW | -- |
| Nonprompt | 9 TTLL samples | LNT+Bjet (blinded) | genW x puW x prefireW x FR_wt | [nonprompt.md](nonprompt.md) |
| Diboson | WZ + ZZ | Tight+0tag+nJets>0 (blinded) | genW x puW x prefireW x cal_wt | [diboson.md](diboson.md) |
| ttX | TTZ + tZq + TTH | Tight+Bjet | genW x puW x prefireW | [ttX.md](ttX.md) |

**Blinding:** Nonprompt uses only LNT events (not genuine Tight+Bjet) and diboson uses only promoted 0-tag events (not genuine Tight+Bjet) for training, keeping the signal region unbiased for validation/test.

### A. Signal (per mass point)

Both channels (Run1E2Mu + Run3Mu) summed, Tight+Bjet selection.

| Mass point | Run2 | Run3 | Total | /fold | Run2 sumW | Run3 sumW | Total sumW |
|------------|-----:|-----:|------:|------:|----------:|----------:|-----------:|
| MHc100_MA95 | 271,376 | 221,991 | 493,367 | 98,673 | 173.9 | 44.6 | 218.5 |
| MHc115_MA87 | 272,946 | -- | 272,946 | 54,589 | 213.4 | -- | 213.4 |
| MHc130_MA90 | 283,785 | 108,012 | 391,797 | 78,359 | 223.3 | 57.5 | 280.8 |
| MHc145_MA92 | 280,421 | -- | 280,421 | 56,084 | 222.9 | -- | 222.9 |
| MHc160_MA85 | 271,740 | 109,930 | 381,670 | 76,334 | 213.7 | 57.1 | 270.8 |
| MHc160_MA98 | 262,968 | -- | 262,968 | 52,594 | 209.7 | -- | 209.7 |

### B. Nonprompt (9 TTLL, LNT+Bjet)

Both channels summed. Weight = genW x puW x prefireW x fake rate weight.

| Sample | Run2 | Run3 | Total | /fold | Run2 sumW_FR | Run3 sumW_FR |
|--------|-----:|-----:|------:|------:|-------------:|-------------:|
| TTLL_powheg | 77,457 | 22,906 | 100,363 | 20,073 | 432.2 | 120.4 |
| TTLL_mtop171p5 | 31,021 | 8,912 | 39,933 | 7,987 | 427.7 | 111.9 |
| TTLL_mtop173p5 | 32,543 | 9,204 | 41,747 | 8,349 | 434.8 | 117.1 |
| TTLL_TuneCP5up | 29,648 | 10,422 | 40,070 | 8,014 | 426.8 | 119.3 |
| TTLL_TuneCP5down | 30,235 | 10,134 | 40,369 | 8,074 | 431.0 | 117.8 |
| TTLL_TuneCP5CR1 | 30,482 | 9,025 | 39,507 | 7,901 | 416.0 | 118.5 |
| TTLL_TuneCP5CR2 | 31,186 | 9,509 | 40,695 | 8,139 | 428.3 | 122.4 |
| TTLL_hdamp_up | 30,216 | 9,099 | 39,315 | 7,863 | 432.9 | 119.2 |
| TTLL_hdamp_down | 30,735 | 8,518 | 39,253 | 7,851 | 420.6 | 113.6 |
| **Total** | **323,523** | **97,729** | **421,252** | **84,250** | **3,850.3** | **1,060.0** |

**Per-fold feasibility:**

| Metric | Run2 | Run3 | Total |
|--------|-----:|-----:|------:|
| Available /fold | 64,705 | 19,546 | 84,250 |
| Target /fold | 34,000 | 16,000 | 40,000 |
| Headroom | 190% | 122% | 211% |

**Subprocess balance:** 9 samples, cap each at ~40,000/9 ~ 4,444/fold. Nominal TTLL_powheg has ~2x the stats of systematics samples, so cap it to match. All 9 samples cover all 8 eras.

### C. Diboson (WZ + ZZ, promoted 0-tag)

Both channels summed, Tight + 0-tag + nJets>0 (events for b-jet rank promotion).
Base sumW shown (before promotion calibration weights, which are normalization-preserving shape corrections).

| Sample | Run2 | Run3 | Total | /fold | Run2 sumW | Run3 sumW | Total sumW |
|--------|-----:|-----:|------:|------:|----------:|----------:|-----------:|
| WZ | 162,426 | 19,812 | 182,238 | 36,448 | 2,210.4 | 319.0 | 2,529.3 |
| ZZ | 375,810 | 79,176 | 454,986 | 90,997 | 212.2 | 63.0 | 275.3 |
| **Total** | **538,236** | **98,988** | **637,224** | **127,445** | **2,422.6** | **382.0** | **2,804.6** |

**Per-fold feasibility:**

| Metric | Run2 | Run3 | Total |
|--------|-----:|-----:|------:|
| Available /fold | 107,647 | 19,798 | 127,445 |
| Target /fold | 34,000 | 16,000 | 40,000 |
| Headroom | 317% | 124% | 319% |

**Subprocess balance:** ZZ has ~2.5x the raw events of WZ. Cap ZZ to match WZ (~18k/fold each for Run2; ~8k/fold each for Run3). Physics yield is WZ-dominated (sumW 2,529 vs 275), so this balances event counts while WZ naturally dominates the effective weight.

### D. ttX (TTZ + tZq + TTH, Tight+Bjet)

Both channels summed. From [ttX.md](ttX.md): TTW excluded (Run3 stats too low).

| Sample | Run2 | Run3 | Total | /fold | Run2 sumW | Run3 sumW | Total sumW | % of ttX |
|--------|-----:|-----:|------:|------:|----------:|----------:|-----------:|---------:|
| TTZ | 479,096 | 59,087 | 538,183 | 107,637 | 348.5 | 107.7 | 456.2 | 72% |
| tZq | 341,656 | 39,453 | 381,109 | 76,222 | 109.5 | 28.0 | 137.5 | 22% |
| TTH | 15,653 | 24,289 | 39,942 | 7,988 | 27.4 | 10.2 | 37.7 | 6% |
| **Total** | **836,405** | **122,829** | **959,234** | **191,847** | **485.4** | **145.9** | **631.3** | **100%** |

**Per-fold feasibility:**

| Metric | Run2 | Run3 | Total |
|--------|-----:|-----:|------:|
| Available /fold | 167,281 | 24,566 | 191,847 |
| Target /fold | 34,000 | 16,000 | 40,000 |
| Headroom | 492% | 154% | 480% |

**Subprocess balance:** TTZ ~108k/fold, tZq ~76k/fold, TTH ~8k/fold. Cap TTZ and tZq to equalize. TTH is limiting but still ~8k/fold (adequate given its 18% physics weight).

### Budget Summary

| Class | Run2/fold | % of 34k | Run3/fold | % of 16k | Total/fold | % of 40k | Status |
|-------|----------:|---------:|----------:|---------:|-----------:|---------:|--------|
| Signal (typical) | ~55k | 162% | ~22k | 138% | ~77k | 193% | Capped at 40k |
| Nonprompt | 64,705 | 190% | 19,546 | 122% | 84,250 | 211% | Capped at 40k |
| Diboson | 107,647 | 317% | 19,798 | 124% | 127,445 | 319% | Capped at 40k |
| ttX | 167,281 | 492% | 24,566 | 154% | 191,847 | 480% | Capped at 40k |

All classes comfortably meet the 40k/fold target. Run3 is the tighter constraint across all categories but still meets 16k/fold targets with 124-154% headroom.

## Run1E2Mu Channel

### Signal Samples

| Sample | Era | Raw | Tight | Bjet | Tight+Bjet |
|--------|-----|----:|------:|-----:|-----------:|
| MHc100_MA95 | 2016preVFP | 42139 | 31762 | 36749 | 27884 |
| MHc100_MA95 | 2016postVFP | 37181 | 28618 | 32947 | 25535 |
| MHc100_MA95 | 2017 | 83315 | 61862 | 75183 | 56118 |
| MHc100_MA95 | 2018 | 77363 | 58555 | 69651 | 53031 |
| MHc100_MA95 | 2022 | 28084 | 20187 | 23748 | 17121 |
| MHc100_MA95 | 2022EE | 99346 | 70981 | 83891 | 60235 |
| MHc100_MA95 | 2023 | 61216 | 46166 | 51710 | 39225 |
| MHc100_MA95 | 2023BPix | 25761 | 19276 | 21815 | 16407 |
| MHc115_MA87 | 2016preVFP | 41941 | 30790 | 35228 | 26008 |
| MHc115_MA87 | 2016postVFP | 38187 | 28483 | 32690 | 24478 |
| MHc115_MA87 | 2017 | 82538 | 59949 | 72468 | 52820 |
| MHc115_MA87 | 2018 | 80647 | 59215 | 70596 | 52095 |
| MHc130_MA90 | 2016preVFP | 45821 | 33711 | 37062 | 27473 |
| MHc130_MA90 | 2016postVFP | 40630 | 30417 | 33562 | 25326 |
| MHc130_MA90 | 2017 | 89381 | 64867 | 75861 | 55404 |
| MHc130_MA90 | 2018 | 87434 | 64662 | 73667 | 54729 |
| MHc130_MA90 | 2022 | 18385 | 12735 | 13266 | 9205 |
| MHc130_MA90 | 2022EE | 53757 | 37187 | 38769 | 27042 |
| MHc130_MA90 | 2023 | 32654 | 23655 | 23701 | 17217 |
| MHc130_MA90 | 2023BPix | 15296 | 11016 | 10888 | 7850 |
| MHc145_MA92 | 2016preVFP | 48504 | 35601 | 36906 | 27291 |
| MHc145_MA92 | 2016postVFP | 42566 | 31859 | 32797 | 24778 |
| MHc145_MA92 | 2017 | 94996 | 68646 | 76048 | 55288 |
| MHc145_MA92 | 2018 | 93141 | 68487 | 74162 | 54849 |
| MHc160_MA85 | 2016preVFP | 52188 | 38403 | 35532 | 26319 |
| MHc160_MA85 | 2016postVFP | 47064 | 35208 | 32880 | 24717 |
| MHc160_MA85 | 2017 | 102333 | 74305 | 74133 | 54058 |
| MHc160_MA85 | 2018 | 101658 | 75188 | 73244 | 54423 |
| MHc160_MA85 | 2022 | 21996 | 15510 | 13434 | 9521 |
| MHc160_MA85 | 2022EE | 65642 | 45936 | 40112 | 28136 |
| MHc160_MA85 | 2023 | 40649 | 29843 | 24901 | 18327 |
| MHc160_MA85 | 2023BPix | 19882 | 14520 | 12210 | 8941 |
| MHc160_MA98 | 2016preVFP | 50812 | 38123 | 34825 | 26274 |
| MHc160_MA98 | 2016postVFP | 44824 | 33904 | 31445 | 23911 |
| MHc160_MA98 | 2017 | 97313 | 71478 | 70888 | 52449 |
| MHc160_MA98 | 2018 | 93567 | 70145 | 67818 | 51223 |

### Nonprompt Background (TTLL)

| Sample | Era | Raw | Tight | Bjet | Tight+Bjet |
|--------|-----|----:|------:|-----:|-----------:|
| TTLL_TuneCP5CR1_powheg | 2016preVFP | 3910 | 501 | 2731 | 357 |
| TTLL_TuneCP5CR1_powheg | 2016postVFP | 4658 | 614 | 3334 | 452 |
| TTLL_TuneCP5CR1_powheg | 2017 | 8857 | 1138 | 6607 | 860 |
| TTLL_TuneCP5CR1_powheg | 2018 | 12719 | 1584 | 9397 | 1177 |
| TTLL_TuneCP5CR1_powheg | 2022 | 1168 | 176 | 807 | 122 |
| TTLL_TuneCP5CR1_powheg | 2022EE | 4573 | 540 | 3192 | 411 |
| TTLL_TuneCP5CR1_powheg | 2023 | 2595 | 375 | 1782 | 265 |
| TTLL_TuneCP5CR1_powheg | 2023BPix | 1382 | 206 | 969 | 153 |
| TTLL_TuneCP5CR2_powheg | 2016preVFP | 4276 | 531 | 2953 | 387 |
| TTLL_TuneCP5CR2_powheg | 2016postVFP | 4702 | 573 | 3354 | 412 |
| TTLL_TuneCP5CR2_powheg | 2017 | 9120 | 1143 | 6751 | 869 |
| TTLL_TuneCP5CR2_powheg | 2018 | 12803 | 1654 | 9458 | 1242 |
| TTLL_TuneCP5CR2_powheg | 2022 | 1279 | 152 | 867 | 109 |
| TTLL_TuneCP5CR2_powheg | 2022EE | 4733 | 614 | 3262 | 423 |
| TTLL_TuneCP5CR2_powheg | 2023 | 2787 | 412 | 1909 | 286 |
| TTLL_TuneCP5CR2_powheg | 2023BPix | 1368 | 195 | 990 | 140 |
| TTLL_TuneCP5_RTT_powheg | 2016preVFP | 11693 | 1563 | 8317 | 1131 |
| TTLL_TuneCP5_RTT_powheg | 2016postVFP | 13634 | 1748 | 9834 | 1284 |
| TTLL_TuneCP5_RTT_powheg | 2017 | 24741 | 3075 | 18503 | 2360 |
| TTLL_TuneCP5_RTT_powheg | 2018 | 35193 | 4549 | 26385 | 3493 |
| TTLL_TuneCP5_erdOn_powheg | 2016preVFP | 4236 | 558 | 2975 | 390 |
| TTLL_TuneCP5_erdOn_powheg | 2016postVFP | 4466 | 595 | 3166 | 439 |
| TTLL_TuneCP5_erdOn_powheg | 2017 | 9122 | 1127 | 6881 | 871 |
| TTLL_TuneCP5_erdOn_powheg | 2018 | 13358 | 1680 | 9865 | 1278 |
| TTLL_TuneCP5_erdOn_powheg | 2022 | 1195 | 165 | 827 | 124 |
| TTLL_TuneCP5_erdOn_powheg | 2022EE | 4830 | 580 | 3270 | 408 |
| TTLL_TuneCP5_erdOn_powheg | 2023 | 2708 | 431 | 1841 | 287 |
| TTLL_TuneCP5_erdOn_powheg | 2023BPix | 1250 | 169 | 864 | 101 |
| TTLL_TuneCP5down_powheg | 2016preVFP | 5426 | 675 | 3844 | 500 |
| TTLL_TuneCP5down_powheg | 2016postVFP | 4789 | 618 | 3490 | 465 |
| TTLL_TuneCP5down_powheg | 2017 | 8418 | 1070 | 6265 | 814 |
| TTLL_TuneCP5down_powheg | 2018 | 13224 | 1735 | 9835 | 1316 |
| TTLL_TuneCP5down_powheg | 2022 | 2592 | 356 | 1766 | 236 |
| TTLL_TuneCP5down_powheg | 2022EE | 4419 | 568 | 2977 | 387 |
| TTLL_TuneCP5down_powheg | 2023 | 2733 | 416 | 1848 | 283 |
| TTLL_TuneCP5down_powheg | 2023BPix | 1269 | 165 | 879 | 116 |
| TTLL_TuneCP5up_powheg | 2016preVFP | 2601 | 330 | 1822 | 213 |
| TTLL_TuneCP5up_powheg | 2016postVFP | 4517 | 600 | 3260 | 427 |
| TTLL_TuneCP5up_powheg | 2017 | 9373 | 1135 | 6915 | 861 |
| TTLL_TuneCP5up_powheg | 2018 | 12562 | 1579 | 9321 | 1175 |
| TTLL_TuneCP5up_powheg | 2022 | 2636 | 305 | 1783 | 195 |
| TTLL_TuneCP5up_powheg | 2022EE | 4545 | 561 | 3072 | 385 |
| TTLL_TuneCP5up_powheg | 2023 | 2682 | 373 | 1875 | 277 |
| TTLL_TuneCP5up_powheg | 2023BPix | 1294 | 190 | 898 | 131 |
| TTLL_hdamp_down_powheg | 2016preVFP | 4303 | 541 | 3029 | 388 |
| TTLL_hdamp_down_powheg | 2016postVFP | 4416 | 574 | 3163 | 424 |
| TTLL_hdamp_down_powheg | 2017 | 8727 | 1094 | 6432 | 807 |
| TTLL_hdamp_down_powheg | 2018 | 13118 | 1707 | 9721 | 1265 |
| TTLL_hdamp_down_powheg | 2022 | 1098 | 133 | 767 | 95 |
| TTLL_hdamp_down_powheg | 2022EE | 4306 | 546 | 2988 | 392 |
| TTLL_hdamp_down_powheg | 2023 | 2457 | 362 | 1702 | 238 |
| TTLL_hdamp_down_powheg | 2023BPix | 1171 | 170 | 822 | 122 |
| TTLL_hdamp_up_powheg | 2016preVFP | 3795 | 498 | 2660 | 348 |
| TTLL_hdamp_up_powheg | 2016postVFP | 4949 | 644 | 3560 | 471 |
| TTLL_hdamp_up_powheg | 2017 | 8825 | 1082 | 6582 | 824 |
| TTLL_hdamp_up_powheg | 2018 | 12436 | 1635 | 9193 | 1267 |
| TTLL_hdamp_up_powheg | 2022 | 1238 | 166 | 859 | 126 |
| TTLL_hdamp_up_powheg | 2022EE | 4724 | 629 | 3224 | 428 |
| TTLL_hdamp_up_powheg | 2023 | 2685 | 338 | 1813 | 228 |
| TTLL_hdamp_up_powheg | 2023BPix | 1284 | 171 | 888 | 119 |
| TTLL_mtop166p5_powheg | 2022 | 1128 | 135 | 769 | 91 |
| TTLL_mtop166p5_powheg | 2022EE | 4192 | 558 | 2832 | 371 |
| TTLL_mtop166p5_powheg | 2023 | 2414 | 331 | 1571 | 213 |
| TTLL_mtop166p5_powheg | 2023BPix | 1256 | 180 | 859 | 126 |
| TTLL_mtop169p5_powheg | 2016preVFP | 2771 | 366 | 1927 | 271 |
| TTLL_mtop169p5_powheg | 2016postVFP | 4351 | 601 | 3123 | 453 |
| TTLL_mtop169p5_powheg | 2017 | 8796 | 1096 | 6505 | 820 |
| TTLL_mtop169p5_powheg | 2018 | 13017 | 1754 | 9607 | 1316 |
| TTLL_mtop169p5_powheg | 2022 | 1247 | 177 | 844 | 124 |
| TTLL_mtop169p5_powheg | 2022EE | 4637 | 630 | 3153 | 426 |
| TTLL_mtop169p5_powheg | 2023 | 2587 | 384 | 1732 | 256 |
| TTLL_mtop169p5_powheg | 2023BPix | 1206 | 153 | 793 | 103 |
| TTLL_mtop171p5_powheg | 2016preVFP | 4181 | 538 | 2896 | 375 |
| TTLL_mtop171p5_powheg | 2016postVFP | 4421 | 590 | 3153 | 431 |
| TTLL_mtop171p5_powheg | 2017 | 9052 | 1095 | 6695 | 814 |
| TTLL_mtop171p5_powheg | 2018 | 13199 | 1712 | 9827 | 1317 |
| TTLL_mtop171p5_powheg | 2022 | 1187 | 156 | 796 | 105 |
| TTLL_mtop171p5_powheg | 2022EE | 4655 | 627 | 3160 | 455 |
| TTLL_mtop171p5_powheg | 2023 | 2689 | 388 | 1814 | 259 |
| TTLL_mtop171p5_powheg | 2023BPix | 1279 | 182 | 847 | 115 |
| TTLL_mtop173p5_powheg | 2016preVFP | 4274 | 575 | 2981 | 415 |
| TTLL_mtop173p5_powheg | 2016postVFP | 4974 | 661 | 3611 | 490 |
| TTLL_mtop173p5_powheg | 2017 | 9440 | 1227 | 7088 | 927 |
| TTLL_mtop173p5_powheg | 2018 | 13553 | 1708 | 10073 | 1288 |
| TTLL_mtop173p5_powheg | 2022 | 1121 | 144 | 755 | 100 |
| TTLL_mtop173p5_powheg | 2022EE | 4814 | 636 | 3287 | 456 |
| TTLL_mtop173p5_powheg | 2023 | 2728 | 361 | 1852 | 246 |
| TTLL_mtop173p5_powheg | 2023BPix | 1194 | 181 | 795 | 113 |
| TTLL_mtop175p5_powheg | 2016preVFP | 4211 | 550 | 2902 | 412 |
| TTLL_mtop175p5_powheg | 2016postVFP | 4775 | 644 | 3433 | 458 |
| TTLL_mtop175p5_powheg | 2017 | 9482 | 1185 | 7088 | 908 |
| TTLL_mtop175p5_powheg | 2018 | 13631 | 1784 | 10063 | 1358 |
| TTLL_mtop175p5_powheg | 2022 | 1225 | 176 | 855 | 136 |
| TTLL_mtop175p5_powheg | 2022EE | 4455 | 544 | 3074 | 390 |
| TTLL_mtop175p5_powheg | 2023 | 2719 | 389 | 1852 | 273 |
| TTLL_mtop175p5_powheg | 2023BPix | 1340 | 190 | 914 | 130 |
| TTLL_mtop178p5_powheg | 2022 | 1247 | 160 | 832 | 113 |
| TTLL_mtop178p5_powheg | 2022EE | 5042 | 606 | 3412 | 423 |
| TTLL_mtop178p5_powheg | 2023 | 2430 | 338 | 1708 | 246 |
| TTLL_mtop178p5_powheg | 2023BPix | 1270 | 175 | 865 | 116 |
| TTLL_powheg | 2016preVFP | 10153 | 1399 | 7100 | 1001 |
| TTLL_powheg | 2016postVFP | 11681 | 1579 | 8304 | 1178 |
| TTLL_powheg | 2017 | 24601 | 3279 | 18271 | 2494 |
| TTLL_powheg | 2018 | 34999 | 4733 | 26056 | 3589 |
| TTLL_powheg | 2022 | 3082 | 422 | 2095 | 295 |
| TTLL_powheg | 2022EE | 11582 | 1455 | 7960 | 1028 |
| TTLL_powheg | 2023 | 6681 | 935 | 4580 | 674 |
| TTLL_powheg | 2023BPix | 3322 | 500 | 2317 | 357 |
| TTLL_powheg_ext1 | 2022 | 3189 | 427 | 2195 | 295 |
| TTLL_powheg_ext1 | 2022EE | 12012 | 1543 | 8149 | 1080 |
| TTLL_widthx0p7_powheg | 2016preVFP | 3742 | 489 | 2590 | 334 |
| TTLL_widthx0p7_powheg | 2016postVFP | 4915 | 638 | 3488 | 473 |
| TTLL_widthx0p7_powheg | 2017 | 9005 | 1174 | 6623 | 886 |
| TTLL_widthx0p7_powheg | 2018 | 13402 | 1678 | 10051 | 1288 |
| TTLL_widthx0p85_powheg | 2016preVFP | 4007 | 503 | 2781 | 345 |
| TTLL_widthx0p85_powheg | 2016postVFP | 4854 | 671 | 3444 | 472 |
| TTLL_widthx0p85_powheg | 2017 | 9019 | 1145 | 6650 | 837 |
| TTLL_widthx0p85_powheg | 2018 | 13136 | 1668 | 9738 | 1263 |
| TTLL_widthx1p15_powheg | 2016preVFP | 4314 | 554 | 3024 | 394 |
| TTLL_widthx1p15_powheg | 2016postVFP | 4631 | 583 | 3333 | 422 |
| TTLL_widthx1p15_powheg | 2017 | 8854 | 1165 | 6610 | 887 |
| TTLL_widthx1p15_powheg | 2018 | 12908 | 1699 | 9495 | 1275 |
| TTLL_widthx1p3_powheg | 2016preVFP | 4286 | 544 | 3001 | 376 |
| TTLL_widthx1p3_powheg | 2016postVFP | 4661 | 601 | 3341 | 429 |
| TTLL_widthx1p3_powheg | 2017 | 8799 | 1169 | 6518 | 888 |
| TTLL_widthx1p3_powheg | 2018 | 13263 | 1714 | 9776 | 1288 |

### Diboson Background (WZ, ZZ)

> **Note:** Diboson events with zero b-tagged jets are used for data augmentation via conditional rank-based promotion. See [`diboson.md`](diboson.md) for the full augmentation strategy.

| Sample | Era | Raw | Tight | Bjet | Tight+Bjet |
|--------|-----|----:|------:|-----:|-----------:|
| WZTo3LNu_amcatnlo | 2016preVFP | 28638 | 22090 | 2401 | 1817 |
| WZTo3LNu_amcatnlo | 2016postVFP | 32343 | 25504 | 2694 | 2110 |
| WZTo3LNu_amcatnlo | 2017 | 33426 | 25318 | 2625 | 2010 |
| WZTo3LNu_amcatnlo | 2018 | 31059 | 24026 | 2391 | 1871 |
| WZTo3LNu_powheg | 2022 | 1882 | 1381 | 142 | 100 |
| WZTo3LNu_powheg | 2022EE | 8128 | 5958 | 566 | 424 |
| WZTo3LNu_powheg | 2023 | 3934 | 2978 | 302 | 228 |
| WZTo3LNu_powheg | 2023BPix | 1965 | 1470 | 153 | 115 |
| WZTo3LNu_powheg_mllmin4p0 | 2016preVFP | 2165 | 1724 | 182 | 143 |
| WZTo3LNu_powheg_mllmin4p0 | 2016postVFP | 1899 | 1479 | 181 | 144 |
| WZTo3LNu_powheg_mllmin4p0 | 2017 | 4482 | 3407 | 385 | 300 |
| WZTo3LNu_powheg_mllmin4p0 | 2018 | 4146 | 3230 | 356 | 287 |
| ZZTo4L_powheg | 2016preVFP | 58571 | 44100 | 7646 | 5737 |
| ZZTo4L_powheg | 2016postVFP | 67401 | 51519 | 9367 | 7139 |
| ZZTo4L_powheg | 2017 | 134739 | 98755 | 17260 | 12582 |
| ZZTo4L_powheg | 2018 | 127862 | 95917 | 17272 | 12832 |
| ZZTo4L_powheg | 2022 | 9723 | 6863 | 789 | 545 |
| ZZTo4L_powheg | 2022EE | 45261 | 31774 | 3332 | 2374 |
| ZZTo4L_powheg | 2023 | 18273 | 13635 | 1554 | 1155 |
| ZZTo4L_powheg | 2023BPix | 8965 | 6558 | 747 | 551 |

### ttX Background (TTZ, TTW, tZq, TTH)

| Sample | Era | Raw | Tight | Bjet | Tight+Bjet |
|--------|-----|----:|------:|-----:|-----------:|
| TTWToLNu | 2016preVFP | 3885 | 2526 | 3284 | 2171 |
| TTWToLNu | 2016postVFP | 4775 | 3181 | 4127 | 2845 |
| TTWToLNu | 2017 | 10257 | 6634 | 9076 | 5973 |
| TTWToLNu | 2018 | 14444 | 9500 | 12586 | 8441 |
| TTWToLNu | 2022 | 112 | 68 | 89 | 58 |
| TTWToLNu | 2022EE | 275 | 151 | 230 | 123 |
| TTWToLNu | 2023 | 445 | 289 | 359 | 241 |
| TTWToLNu | 2023BPix | 235 | 149 | 193 | 123 |
| TTZToLLNuNu | 2016preVFP | 54498 | 40040 | 45557 | 33771 |
| TTZToLLNuNu | 2016postVFP | 59481 | 44582 | 50411 | 38085 |
| TTZToLLNuNu | 2017 | 138012 | 100183 | 120077 | 87816 |
| TTZToLLNuNu | 2018 | 191638 | 142699 | 166025 | 124537 |
| TTZ_M50 | 2022 | 8406 | 5994 | 6204 | 4462 |
| TTZ_M50 | 2022EE | 31264 | 22223 | 23399 | 16709 |
| TTZ_M50 | 2023 | 16985 | 12716 | 12648 | 9531 |
| TTZ_M50 | 2023BPix | 8509 | 6288 | 6410 | 4742 |
| tZq | 2016preVFP | 35765 | 25847 | 27005 | 19974 |
| tZq | 2016postVFP | 42528 | 31248 | 32599 | 24437 |
| tZq | 2017 | 101527 | 72699 | 80723 | 58886 |
| tZq | 2018 | 125587 | 91912 | 99432 | 74040 |
| tZq | 2022 | 5236 | 3643 | 3732 | 2680 |
| tZq | 2022EE | 19432 | 13472 | 14063 | 9934 |
| tZq | 2023 | 10754 | 7766 | 7765 | 5710 |
| tZq | 2023BPix | 4851 | 3494 | 3558 | 2585 |
| TTHToNonbb | 2016preVFP | 2199 | 1406 | 1871 | 1214 |
| TTHToNonbb | 2016postVFP | 2404 | 1498 | 2068 | 1319 |
| TTHToNonbb | 2017 | 5331 | 3219 | 4693 | 2890 |
| TTHToNonbb | 2018 | 7780 | 4867 | 6787 | 4318 |
| TTHToNonbb | 2022 | 3490 | 2016 | 2723 | 1603 |
| TTHToNonbb | 2022EE | 12699 | 7373 | 10035 | 5907 |
| TTHToNonbb | 2023 | 10577 | 6417 | 8237 | 5138 |
| TTHToNonbb | 2023BPix | 5152 | 3108 | 3996 | 2463 |

### Other Samples (DYJets, ...)

| Sample | Era | Raw | Tight | Bjet | Tight+Bjet |
|--------|-----|----:|------:|-----:|-----------:|
| DYJets | 2016preVFP | 823 | 224 | 169 | 47 |
| DYJets | 2016postVFP | 667 | 188 | 140 | 25 |
| DYJets | 2017 | 1676 | 376 | 343 | 79 |
| DYJets | 2018 | 1590 | 409 | 365 | 83 |
| DYJets | 2022 | 283 | 63 | 39 | 8 |
| DYJets | 2022EE | 919 | 203 | 147 | 25 |
| DYJets | 2023 | 552 | 144 | 122 | 39 |
| DYJets | 2023BPix | 300 | 87 | 58 | 19 |

## Run3Mu Channel

### Signal Samples

| Sample | Era | Raw | Tight | Bjet | Tight+Bjet |
|--------|-----|----:|------:|-----:|-----------:|
| MHc100_MA95 | 2016preVFP | 28033 | 22278 | 24464 | 19573 |
| MHc100_MA95 | 2016postVFP | 24614 | 19744 | 21739 | 17557 |
| MHc100_MA95 | 2017 | 52945 | 40403 | 47739 | 36665 |
| MHc100_MA95 | 2018 | 49516 | 38756 | 44480 | 35013 |
| MHc100_MA95 | 2022 | 18844 | 13067 | 15911 | 11131 |
| MHc100_MA95 | 2022EE | 67655 | 46337 | 57011 | 39355 |
| MHc100_MA95 | 2023 | 42331 | 31746 | 35728 | 27028 |
| MHc100_MA95 | 2023BPix | 18105 | 13487 | 15340 | 11489 |
| MHc115_MA87 | 2016preVFP | 31725 | 24312 | 26681 | 20593 |
| MHc115_MA87 | 2016postVFP | 28621 | 22015 | 24481 | 18938 |
| MHc115_MA87 | 2017 | 60364 | 44162 | 52855 | 38903 |
| MHc115_MA87 | 2018 | 59327 | 44553 | 51831 | 39111 |
| MHc130_MA90 | 2016preVFP | 33896 | 26028 | 27387 | 21123 |
| MHc130_MA90 | 2016postVFP | 29931 | 23110 | 24659 | 19164 |
| MHc130_MA90 | 2017 | 64497 | 47308 | 54578 | 40218 |
| MHc130_MA90 | 2018 | 63280 | 47568 | 53350 | 40348 |
| MHc130_MA90 | 2022 | 13924 | 9526 | 9943 | 6850 |
| MHc130_MA90 | 2022EE | 41493 | 27562 | 29702 | 19836 |
| MHc130_MA90 | 2023 | 26087 | 19069 | 18817 | 13783 |
| MHc130_MA90 | 2023BPix | 12139 | 8753 | 8610 | 6229 |
| MHc145_MA92 | 2016preVFP | 35382 | 27303 | 26655 | 20785 |
| MHc145_MA92 | 2016postVFP | 30716 | 23940 | 23608 | 18557 |
| MHc145_MA92 | 2017 | 66422 | 48943 | 52973 | 39250 |
| MHc145_MA92 | 2018 | 65888 | 49692 | 52226 | 39623 |
| MHc160_MA85 | 2016preVFP | 35540 | 27800 | 24313 | 19172 |
| MHc160_MA85 | 2016postVFP | 32025 | 25144 | 22295 | 17615 |
| MHc160_MA85 | 2017 | 68976 | 51288 | 49882 | 37285 |
| MHc160_MA85 | 2018 | 69049 | 52551 | 49815 | 38151 |
| MHc160_MA85 | 2022 | 15259 | 10459 | 9366 | 6467 |
| MHc160_MA85 | 2022EE | 45967 | 30965 | 28183 | 19030 |
| MHc160_MA85 | 2023 | 28531 | 21098 | 17468 | 12994 |
| MHc160_MA85 | 2023BPix | 14266 | 10544 | 8746 | 6514 |
| MHc160_MA98 | 2016preVFP | 35349 | 27801 | 24123 | 19153 |
| MHc160_MA98 | 2016postVFP | 31467 | 24866 | 21951 | 17458 |
| MHc160_MA98 | 2017 | 66324 | 49755 | 48317 | 36455 |
| MHc160_MA98 | 2018 | 64611 | 49430 | 46917 | 36045 |

### Nonprompt Background (TTLL)

| Sample | Era | Raw | Tight | Bjet | Tight+Bjet |
|--------|-----|----:|------:|-----:|-----------:|
| TTLL_TuneCP5CR1_powheg | 2016preVFP | 2101 | 295 | 1434 | 208 |
| TTLL_TuneCP5CR1_powheg | 2016postVFP | 2494 | 331 | 1738 | 230 |
| TTLL_TuneCP5CR1_powheg | 2017 | 4705 | 596 | 3425 | 452 |
| TTLL_TuneCP5CR1_powheg | 2018 | 6764 | 964 | 5003 | 727 |
| TTLL_TuneCP5CR1_powheg | 2022 | 634 | 84 | 426 | 57 |
| TTLL_TuneCP5CR1_powheg | 2022EE | 2493 | 289 | 1696 | 195 |
| TTLL_TuneCP5CR1_powheg | 2023 | 1505 | 199 | 1022 | 143 |
| TTLL_TuneCP5CR1_powheg | 2023BPix | 794 | 96 | 544 | 67 |
| TTLL_TuneCP5CR2_powheg | 2016preVFP | 2318 | 320 | 1578 | 225 |
| TTLL_TuneCP5CR2_powheg | 2016postVFP | 2543 | 358 | 1802 | 254 |
| TTLL_TuneCP5CR2_powheg | 2017 | 4920 | 584 | 3575 | 439 |
| TTLL_TuneCP5CR2_powheg | 2018 | 6821 | 927 | 5047 | 674 |
| TTLL_TuneCP5CR2_powheg | 2022 | 648 | 67 | 435 | 43 |
| TTLL_TuneCP5CR2_powheg | 2022EE | 2614 | 236 | 1772 | 162 |
| TTLL_TuneCP5CR2_powheg | 2023 | 1643 | 217 | 1108 | 144 |
| TTLL_TuneCP5CR2_powheg | 2023BPix | 769 | 96 | 538 | 65 |
| TTLL_TuneCP5_RTT_powheg | 2016preVFP | 6516 | 886 | 4568 | 648 |
| TTLL_TuneCP5_RTT_powheg | 2016postVFP | 7424 | 1022 | 5335 | 745 |
| TTLL_TuneCP5_RTT_powheg | 2017 | 12899 | 1661 | 9539 | 1209 |
| TTLL_TuneCP5_RTT_powheg | 2018 | 18751 | 2439 | 13866 | 1782 |
| TTLL_TuneCP5_erdOn_powheg | 2016preVFP | 2420 | 352 | 1689 | 244 |
| TTLL_TuneCP5_erdOn_powheg | 2016postVFP | 2553 | 350 | 1795 | 263 |
| TTLL_TuneCP5_erdOn_powheg | 2017 | 4821 | 596 | 3585 | 456 |
| TTLL_TuneCP5_erdOn_powheg | 2018 | 7102 | 945 | 5202 | 684 |
| TTLL_TuneCP5_erdOn_powheg | 2022 | 673 | 87 | 447 | 57 |
| TTLL_TuneCP5_erdOn_powheg | 2022EE | 2708 | 291 | 1809 | 193 |
| TTLL_TuneCP5_erdOn_powheg | 2023 | 1614 | 187 | 1088 | 123 |
| TTLL_TuneCP5_erdOn_powheg | 2023BPix | 756 | 74 | 508 | 56 |
| TTLL_TuneCP5down_powheg | 2016preVFP | 3010 | 416 | 2083 | 295 |
| TTLL_TuneCP5down_powheg | 2016postVFP | 2622 | 314 | 1864 | 222 |
| TTLL_TuneCP5down_powheg | 2017 | 4481 | 585 | 3292 | 445 |
| TTLL_TuneCP5down_powheg | 2018 | 7172 | 938 | 5265 | 685 |
| TTLL_TuneCP5down_powheg | 2022 | 1452 | 160 | 980 | 114 |
| TTLL_TuneCP5down_powheg | 2022EE | 2459 | 273 | 1652 | 181 |
| TTLL_TuneCP5down_powheg | 2023 | 1634 | 205 | 1099 | 147 |
| TTLL_TuneCP5down_powheg | 2023BPix | 712 | 95 | 457 | 60 |
| TTLL_TuneCP5up_powheg | 2016preVFP | 1492 | 201 | 1016 | 128 |
| TTLL_TuneCP5up_powheg | 2016postVFP | 2572 | 354 | 1827 | 257 |
| TTLL_TuneCP5up_powheg | 2017 | 4829 | 624 | 3548 | 462 |
| TTLL_TuneCP5up_powheg | 2018 | 6857 | 900 | 5024 | 678 |
| TTLL_TuneCP5up_powheg | 2022 | 1448 | 139 | 989 | 87 |
| TTLL_TuneCP5up_powheg | 2022EE | 2539 | 267 | 1739 | 199 |
| TTLL_TuneCP5up_powheg | 2023 | 1586 | 211 | 1087 | 148 |
| TTLL_TuneCP5up_powheg | 2023BPix | 726 | 94 | 462 | 61 |
| TTLL_hdamp_down_powheg | 2016preVFP | 2348 | 312 | 1591 | 218 |
| TTLL_hdamp_down_powheg | 2016postVFP | 2542 | 315 | 1799 | 223 |
| TTLL_hdamp_down_powheg | 2017 | 4452 | 557 | 3265 | 423 |
| TTLL_hdamp_down_powheg | 2018 | 7099 | 949 | 5268 | 717 |
| TTLL_hdamp_down_powheg | 2022 | 625 | 60 | 415 | 41 |
| TTLL_hdamp_down_powheg | 2022EE | 2462 | 268 | 1637 | 176 |
| TTLL_hdamp_down_powheg | 2023 | 1463 | 178 | 963 | 122 |
| TTLL_hdamp_down_powheg | 2023BPix | 722 | 104 | 480 | 70 |
| TTLL_hdamp_up_powheg | 2016preVFP | 2207 | 298 | 1544 | 207 |
| TTLL_hdamp_up_powheg | 2016postVFP | 2709 | 356 | 1923 | 241 |
| TTLL_hdamp_up_powheg | 2017 | 4593 | 589 | 3359 | 434 |
| TTLL_hdamp_up_powheg | 2018 | 6517 | 871 | 4790 | 656 |
| TTLL_hdamp_up_powheg | 2022 | 649 | 69 | 445 | 40 |
| TTLL_hdamp_up_powheg | 2022EE | 2528 | 277 | 1631 | 175 |
| TTLL_hdamp_up_powheg | 2023 | 1611 | 199 | 1059 | 138 |
| TTLL_hdamp_up_powheg | 2023BPix | 759 | 95 | 501 | 67 |
| TTLL_mtop166p5_powheg | 2022 | 643 | 75 | 434 | 53 |
| TTLL_mtop166p5_powheg | 2022EE | 2415 | 258 | 1580 | 174 |
| TTLL_mtop166p5_powheg | 2023 | 1418 | 199 | 931 | 131 |
| TTLL_mtop166p5_powheg | 2023BPix | 730 | 90 | 481 | 65 |
| TTLL_mtop169p5_powheg | 2016preVFP | 1598 | 228 | 1142 | 160 |
| TTLL_mtop169p5_powheg | 2016postVFP | 2308 | 326 | 1634 | 245 |
| TTLL_mtop169p5_powheg | 2017 | 4543 | 570 | 3333 | 444 |
| TTLL_mtop169p5_powheg | 2018 | 6921 | 916 | 5050 | 685 |
| TTLL_mtop169p5_powheg | 2022 | 629 | 57 | 411 | 41 |
| TTLL_mtop169p5_powheg | 2022EE | 2435 | 259 | 1653 | 180 |
| TTLL_mtop169p5_powheg | 2023 | 1624 | 209 | 1095 | 144 |
| TTLL_mtop169p5_powheg | 2023BPix | 714 | 92 | 475 | 64 |
| TTLL_mtop171p5_powheg | 2016preVFP | 2319 | 331 | 1588 | 226 |
| TTLL_mtop171p5_powheg | 2016postVFP | 2459 | 325 | 1713 | 218 |
| TTLL_mtop171p5_powheg | 2017 | 4594 | 571 | 3336 | 423 |
| TTLL_mtop171p5_powheg | 2018 | 6966 | 894 | 5151 | 686 |
| TTLL_mtop171p5_powheg | 2022 | 629 | 71 | 407 | 52 |
| TTLL_mtop171p5_powheg | 2022EE | 2632 | 305 | 1716 | 220 |
| TTLL_mtop171p5_powheg | 2023 | 1555 | 167 | 1067 | 116 |
| TTLL_mtop171p5_powheg | 2023BPix | 748 | 118 | 511 | 84 |
| TTLL_mtop173p5_powheg | 2016preVFP | 2391 | 308 | 1626 | 220 |
| TTLL_mtop173p5_powheg | 2016postVFP | 2785 | 374 | 1920 | 276 |
| TTLL_mtop173p5_powheg | 2017 | 4878 | 625 | 3612 | 493 |
| TTLL_mtop173p5_powheg | 2018 | 7244 | 938 | 5368 | 699 |
| TTLL_mtop173p5_powheg | 2022 | 626 | 54 | 409 | 29 |
| TTLL_mtop173p5_powheg | 2022EE | 2751 | 303 | 1841 | 196 |
| TTLL_mtop173p5_powheg | 2023 | 1607 | 183 | 1089 | 120 |
| TTLL_mtop173p5_powheg | 2023BPix | 737 | 107 | 508 | 72 |
| TTLL_mtop175p5_powheg | 2016preVFP | 2492 | 352 | 1771 | 238 |
| TTLL_mtop175p5_powheg | 2016postVFP | 2629 | 322 | 1851 | 238 |
| TTLL_mtop175p5_powheg | 2017 | 4944 | 615 | 3652 | 451 |
| TTLL_mtop175p5_powheg | 2018 | 7444 | 992 | 5521 | 722 |
| TTLL_mtop175p5_powheg | 2022 | 671 | 70 | 464 | 49 |
| TTLL_mtop175p5_powheg | 2022EE | 2448 | 263 | 1647 | 178 |
| TTLL_mtop175p5_powheg | 2023 | 1590 | 184 | 1084 | 125 |
| TTLL_mtop175p5_powheg | 2023BPix | 734 | 106 | 502 | 64 |
| TTLL_mtop178p5_powheg | 2022 | 679 | 72 | 470 | 49 |
| TTLL_mtop178p5_powheg | 2022EE | 2721 | 304 | 1825 | 194 |
| TTLL_mtop178p5_powheg | 2023 | 1357 | 165 | 937 | 124 |
| TTLL_mtop178p5_powheg | 2023BPix | 795 | 106 | 531 | 69 |
| TTLL_powheg | 2016preVFP | 5345 | 746 | 3722 | 530 |
| TTLL_powheg | 2016postVFP | 6372 | 828 | 4513 | 589 |
| TTLL_powheg | 2017 | 12259 | 1509 | 9042 | 1118 |
| TTLL_powheg | 2018 | 17254 | 2279 | 12650 | 1702 |
| TTLL_powheg | 2022 | 1636 | 191 | 1080 | 129 |
| TTLL_powheg | 2022EE | 6478 | 684 | 4350 | 473 |
| TTLL_powheg | 2023 | 3962 | 495 | 2698 | 344 |
| TTLL_powheg | 2023BPix | 1929 | 248 | 1294 | 168 |
| TTLL_powheg_ext1 | 2022 | 1764 | 198 | 1191 | 148 |
| TTLL_powheg_ext1 | 2022EE | 6464 | 679 | 4360 | 463 |
| TTLL_widthx0p7_powheg | 2016preVFP | 2044 | 258 | 1424 | 181 |
| TTLL_widthx0p7_powheg | 2016postVFP | 2791 | 353 | 1968 | 258 |
| TTLL_widthx0p7_powheg | 2017 | 4651 | 610 | 3391 | 430 |
| TTLL_widthx0p7_powheg | 2018 | 7168 | 959 | 5296 | 718 |
| TTLL_widthx0p85_powheg | 2016preVFP | 2116 | 291 | 1431 | 199 |
| TTLL_widthx0p85_powheg | 2016postVFP | 2725 | 375 | 1919 | 257 |
| TTLL_widthx0p85_powheg | 2017 | 4783 | 588 | 3512 | 440 |
| TTLL_widthx0p85_powheg | 2018 | 7035 | 883 | 5132 | 653 |
| TTLL_widthx1p15_powheg | 2016preVFP | 2463 | 357 | 1699 | 266 |
| TTLL_widthx1p15_powheg | 2016postVFP | 2572 | 357 | 1782 | 251 |
| TTLL_widthx1p15_powheg | 2017 | 4578 | 560 | 3337 | 406 |
| TTLL_widthx1p15_powheg | 2018 | 6994 | 955 | 5037 | 709 |
| TTLL_widthx1p3_powheg | 2016preVFP | 2279 | 307 | 1570 | 215 |
| TTLL_widthx1p3_powheg | 2016postVFP | 2671 | 347 | 1911 | 244 |
| TTLL_widthx1p3_powheg | 2017 | 4698 | 610 | 3497 | 438 |
| TTLL_widthx1p3_powheg | 2018 | 7053 | 891 | 5156 | 678 |

### Diboson Background (WZ, ZZ)

> **Note:** Diboson events with zero b-tagged jets are used for data augmentation via conditional rank-based promotion. See [`diboson.md`](diboson.md) for the full augmentation strategy.

| Sample | Era | Raw | Tight | Bjet | Tight+Bjet |
|--------|-----|----:|------:|-----:|-----------:|
| WZTo3LNu_amcatnlo | 2016preVFP | 22978 | 18892 | 1830 | 1501 |
| WZTo3LNu_amcatnlo | 2016postVFP | 25622 | 21028 | 1983 | 1648 |
| WZTo3LNu_amcatnlo | 2017 | 25999 | 20283 | 2010 | 1579 |
| WZTo3LNu_amcatnlo | 2018 | 24154 | 19326 | 1863 | 1505 |
| WZTo3LNu_powheg | 2022 | 1575 | 1071 | 109 | 78 |
| WZTo3LNu_powheg | 2022EE | 6839 | 4719 | 443 | 290 |
| WZTo3LNu_powheg | 2023 | 3341 | 2489 | 260 | 196 |
| WZTo3LNu_powheg | 2023BPix | 1688 | 1266 | 117 | 89 |
| WZTo3LNu_powheg_mllmin4p0 | 2016preVFP | 1714 | 1435 | 133 | 114 |
| WZTo3LNu_powheg_mllmin4p0 | 2016postVFP | 1529 | 1270 | 135 | 115 |
| WZTo3LNu_powheg_mllmin4p0 | 2017 | 3439 | 2700 | 274 | 208 |
| WZTo3LNu_powheg_mllmin4p0 | 2018 | 3331 | 2703 | 246 | 198 |
| ZZTo4L_powheg | 2016preVFP | 29960 | 22904 | 3396 | 2575 |
| ZZTo4L_powheg | 2016postVFP | 32189 | 24681 | 4135 | 3130 |
| ZZTo4L_powheg | 2017 | 65294 | 47134 | 8367 | 6000 |
| ZZTo4L_powheg | 2018 | 62752 | 46682 | 8080 | 5887 |
| ZZTo4L_powheg | 2022 | 4831 | 3042 | 394 | 239 |
| ZZTo4L_powheg | 2022EE | 23158 | 14312 | 1867 | 1135 |
| ZZTo4L_powheg | 2023 | 9796 | 6659 | 911 | 608 |
| ZZTo4L_powheg | 2023BPix | 4720 | 3226 | 422 | 286 |

### ttX Background (TTZ, TTW, tZq, TTH)

| Sample | Era | Raw | Tight | Bjet | Tight+Bjet |
|--------|-----|----:|------:|-----:|-----------:|
| TTWToLNu | 2016preVFP | 2630 | 1814 | 2205 | 1548 |
| TTWToLNu | 2016postVFP | 3144 | 2201 | 2707 | 1930 |
| TTWToLNu | 2017 | 6584 | 4450 | 5757 | 3979 |
| TTWToLNu | 2018 | 9425 | 6471 | 8260 | 5792 |
| TTWToLNu | 2022 | 80 | 38 | 62 | 29 |
| TTWToLNu | 2022EE | 205 | 114 | 168 | 99 |
| TTWToLNu | 2023 | 293 | 184 | 236 | 156 |
| TTWToLNu | 2023BPix | 180 | 110 | 140 | 88 |
| TTZToLLNuNu | 2016preVFP | 36947 | 28842 | 30694 | 24217 |
| TTZToLLNuNu | 2016postVFP | 39815 | 31143 | 33793 | 26643 |
| TTZToLLNuNu | 2017 | 90070 | 67586 | 78390 | 59303 |
| TTZToLLNuNu | 2018 | 126297 | 97408 | 109006 | 84724 |
| TTZ_M50 | 2022 | 5782 | 3978 | 4266 | 2954 |
| TTZ_M50 | 2022EE | 21465 | 14737 | 16022 | 11107 |
| TTZ_M50 | 2023 | 11554 | 8615 | 8628 | 6445 |
| TTZ_M50 | 2023BPix | 5759 | 4208 | 4248 | 3137 |
| tZq | 2016preVFP | 32462 | 25322 | 24862 | 19703 |
| tZq | 2016postVFP | 38276 | 30086 | 29925 | 23835 |
| tZq | 2017 | 86484 | 65035 | 69347 | 52853 |
| tZq | 2018 | 108837 | 83771 | 87161 | 67928 |
| tZq | 2022 | 4553 | 3044 | 3314 | 2250 |
| tZq | 2022EE | 17253 | 11603 | 12583 | 8605 |
| tZq | 2023 | 9750 | 7172 | 7063 | 5308 |
| tZq | 2023BPix | 4382 | 3205 | 3185 | 2381 |
| TTHToNonbb | 2016preVFP | 1313 | 854 | 1105 | 737 |
| TTHToNonbb | 2016postVFP | 1410 | 903 | 1205 | 791 |
| TTHToNonbb | 2017 | 3082 | 1972 | 2726 | 1777 |
| TTHToNonbb | 2018 | 4548 | 2945 | 3963 | 2607 |
| TTHToNonbb | 2022 | 2140 | 1177 | 1654 | 935 |
| TTHToNonbb | 2022EE | 7746 | 4302 | 6091 | 3429 |
| TTHToNonbb | 2023 | 6836 | 4152 | 5354 | 3328 |
| TTHToNonbb | 2023BPix | 3274 | 1898 | 2519 | 1486 |

### Other Samples (DYJets, ...)

| Sample | Era | Raw | Tight | Bjet | Tight+Bjet |
|--------|-----|----:|------:|-----:|-----------:|
| DYJets | 2016preVFP | 313 | 67 | 123 | 25 |
| DYJets | 2016postVFP | 254 | 58 | 82 | 18 |
| DYJets | 2017 | 563 | 122 | 218 | 50 |
| DYJets | 2018 | 604 | 137 | 220 | 50 |
| DYJets | 2022 | 119 | 22 | 35 | 5 |
| DYJets | 2022EE | 370 | 86 | 114 | 20 |
| DYJets | 2023 | 262 | 53 | 94 | 16 |
| DYJets | 2023BPix | 132 | 30 | 42 | 6 |

# Cross-Topology FDIA Detection — Results

All numbers below come from files in `results/`. "Prior" = earlier notebook (`final_cleaned_transfer_results.csv`), not rerun, different split and model.

## 1. Data (verified, `results/audit/`)
- 3 grids × 500,000 rows; 36 / 68 / 132 features (`Flow_i_j`, `Inj_k` at lags t-0..t-3). No NaN, no inf, no duplicate rows.
- Label 0 = normal (≈40%), labels 1–30 = attack (≈60%). Attack share is within 1 point across train/val/test.
- Rows are one per minute, sorted. Row *i+1*'s `t-1` columns equal row *i*'s `t-0` → a random split would leak; we use a **chronological 70/15/15 split**.
- Name-shared columns: 5∩9 = 20, 5∩14 = 28, 9∩14 = 44, all three = 20. **Same name ≠ same physical bus/line** across grids.

## 2. Method
- **Within-grid:** RF, ExtraTrees (ET), XGBoost; 4 configs each tuned on a 150k sample of train, scored on validation at threshold 0.5; best config refit on full train; test used once for reporting.
- **Selection rule:** validation balanced accuracy (BA); ties (<0.001) → lower FPR → faster.
- **Transfer model:** ExtraTrees (200 trees, no depth limit, min leaf 5) for every source — *researcher choice* for a constant algorithm across directions (the rule picked RF for 9-bus; see §3).
- **Main transfer:** per directed pair, a new model trained on source train restricted to the name-shared columns (sorted, identical order on target).
- **Ablation (topology-agnostic):** 40 features — for each lag, mean/std/min/max/mean|x| over all Flow columns and over all Inj columns. One model per source.
- **Deployable thresholds (source validation only):** fixed 0.5; p95 of source-val normal scores; max source-val BA with source-val FPR ≤ 5%.
- **Oracle sweep (diagnostic only):** thresholds 0.1–0.9 scored with target labels. Never used to pick a deployed threshold.

## 3. Within-grid comparison (`results/within/test.csv`)
| Grid | Model | Val BA | Val FPR | Test BA | Test FPR | Test recall | Train s |
|---|---|---|---|---|---|---|---|
| 5 | RF | 0.9009 | 0.0005 | 0.8991 | 0.0002 | 0.798 | 121 |
| 5 | **ET** (rule) | 0.9004 | 0.0000 | 0.8983 | 0.0000 | 0.797 | 95 |
| 5 | XGB | 0.9006 | 0.0011 | 0.8987 | 0.0012 | 0.799 | 14 |
| 9 | **RF** (rule) | 0.8994 | 0.0009 | 0.8997 | 0.0007 | 0.800 | 168 |
| 9 | ET | 0.8974 | 0.0001 | 0.8972 | 0.0000 | 0.795 | 169 |
| 9 | XGB | 0.8992 | 0.0015 | 0.8994 | 0.0011 | 0.800 | 19 |
| 14 | RF | 0.9007 | 0.0001 | 0.8993 | 0.0001 | 0.799 | 216 |
| 14 | **ET** (rule) | 0.8997 | 0.0000 | 0.8986 | 0.0000 | 0.797 | 1088 |
| 14 | XGB | 0.9007 | 0.0002 | 0.8991 | 0.0006 | 0.799 | 29 |

- All three algorithms are within 0.003 BA on every grid; the "winner" is effectively decided by the FPR tie-break. Validation and test agree (no overfitting signal).
- **Recall ceiling ≈ 0.80 for every model.** Per-type recall was computed for the *transfer* models only (`recall_by_attack_type.csv`, target data @0.5): attack types **5, 10, 15, 20, 25, 30** are almost never detected there (recall ≈ 0 on 5- and 14-bus targets). These 6 of 30 types are ≈ 20% of attack rows (0.199 on 5-bus), which is *consistent with* the within-grid 0.80 ceiling — per-type recall of the within-grid models was not computed. In the best transfers, types 2, 7, 12, 17, 22, 27 are caught ≈ 80% (as low as 0.48 in main 5→14). Hypothesis (not verified — attack generator not available): these are stealthy, measurement-consistent injections.
- XGB matches BA at 6–40× less training time.

## 4. Transfers (`results/transfer/transfer_metrics.csv`)
Target balanced accuracy (FPR in brackets).

| Pair | n feat | Prior @0.5 | Main @0.5 | Main p95 | Main best-BA | Ablation @0.5 | Ablation p95 | Ablation best-BA |
|---|---|---|---|---|---|---|---|---|
| 5→9 | 20 | 0.535 (0.730) | 0.556 (0.556) | 0.529 (0.782) | 0.549 (0.617) | 0.530 (0.922) | 0.507 (0.981) | 0.519 (0.951) |
| 5→14 | 28 | 0.744 (0.185) | 0.787 (0.034) | 0.726 (0.245) | 0.779 (0.064) | **0.872** (0.023) | 0.798 (0.218) | 0.858 (0.058) |
| 9→5 | 20 | 0.863 (0.001) | 0.845 (0.045) | 0.834 (0.075) | 0.857 (0.003) | **0.880** (0.000) | 0.865 (0.039) | 0.880 (0.000) |
| 9→14 | 44 | 0.836 (0.000) | 0.835 (0.000) | 0.835 (0.007) | 0.836 (0.000) | **0.880** (0.000) | 0.874 (0.016) | 0.880 (0.000) |
| 14→5 | 28 | 0.880 (0.000) | 0.877 (0.001) | 0.859 (0.053) | 0.877 (0.002) | 0.880 (0.000) | 0.834 (0.122) | 0.880 (0.001) |
| 14→9 | 44 | 0.507 (0.981) | 0.562 (0.834) | 0.510 (0.975) | 0.544 (0.884) | 0.572 (0.812) | 0.511 (0.972) | 0.531 (0.919) |

Target ROC-AUC: main 0.621 / 0.799 / 0.861 / 0.837 / 0.879 / 0.812; ablation 0.871 / 0.883 / 0.881 / 0.882 / 0.883 / 0.877 (same pair order).

**Findings**
1. **Four of six directions transfer well** (BA 0.79–0.88 main). Every transfer **into 9-bus fails** (BA 0.51–0.57) under every deployable threshold, both feature sets.
2. **The ablation helps wherever the target is not 9-bus**: 5→14 +0.085 BA, 9→5 +0.035, 9→14 +0.045, 14→5 ≈ 0. With grid-agnostic features, four directions reach ≈ 0.88 BA, about 0.02 below within-grid test BA (0.90); the gap is mainly types 2, 7, 12, … (recall ≈ 0.80 after transfer).
3. **Source-calibrated thresholds did not help.** p95 is always below 0.5 (0.27–0.48) because source normal scores are tightly packed low; lowering the threshold raises target FPR. Best-BA@FPR≤5% roughly matches 0.5 where transfer works and cannot rescue →9-bus.
4. New main @0.5 results agree with the prior run in direction and rough size; 5→14 and 14→9 differ most (different split, model and hyperparameters).

## 5. Why the asymmetry? Calibration vs discrimination
- **→9-bus is a scale problem.** On *normal* rows, shared-column means match across grids (|shift| ≤ 0.07 source-std) but **9-bus spread is 2.8–6× wider than 5-bus and 1.3–28.6× wider than 14-bus** (`normal_feature_shift.csv`). Trees learned "large value ⇒ attack", so 9-bus normal rows score high: FPR 0.56–0.92 at 0.5 (`figures/hist_main_14-bus_9-bus.png` — target normal scores spread across 0.3–1.0).
- **It is mostly miscalibration, not lost ranking, for 14→9 and ablation 5→9**: target AUC 0.81–0.88 while BA ≈ 0.55. The oracle sweep (diagnostic only) reaches 0.747 (main 14→9) and 0.781 (ablation 14→9) at t = 0.9, the top of the swept range, so these are lower bounds — still below the ≈0.88 of other directions at the swept thresholds, and the needed threshold is invisible from source data.
- **Main 5→9 also loses ranking** (AUC 0.62): with 5 shared base signals, 9-bus normal noise and attacks overlap.
- **Out of 9-bus works** because 5- and 14-bus normal data are quieter (std ratio 0.03–0.89): normal stays "normal".
- **Shared-column coverage matters.** On its own validation, the 9-bus model restricted to the 20 columns shared with 5-bus reaches only BA 0.63 (AUC 0.63) — many 9-bus attacks hit buses 6–9 — yet it scores 0.845 on 5-bus, where every attack lands on buses 2–5.

## 6. Limitations
- Name-matched columns are not physically equivalent between grids; the main transfer is a naming coincidence, not a physical mapping.
- Synthetic data; attacks appear nearly per-row random (mean label run ≈ 1.2 rows), so temporal detection is not tested.
- Small tuning budget (4 configs/algorithm, 150k-row sample). Transfer model fixed to ET by choice.
- Only the threshold, not the model, is calibrated; calibration uses source data only by design.
- Single seed; no confidence intervals.
- Stealthy-attack hypothesis for types ≡ 0 (mod 5) is unverified.

## 7. Future work
- **Per-grid scale normalisation from unlabeled target normal-period data** (e.g., z-score each feature per grid). The shift analysis suggests this targets the main failure — but it uses unlabeled target data, a transductive setting distinct from the zero-shot one here.
- Physically aligned features (per-unit, graph/topology-aware features, or a GNN over the bus graph).
- Residual-based features (state-estimation residuals) and analysis of the undetected attack types.
- Multiple seeds and wider tuning; probability calibration (isotonic/Platt) on source validation.

# Extra results — seeds, bootstrap intervals, label-free adaptation

Follow-up to `REPORT.md`, which is unchanged. It addresses two gaps listed there: a single random seed with no intervals, and every transfer into 9-bus failing.
Scripts: `05_adapt.py`, `06_seeds.py`, `07_bootstrap.py`. Outputs: `results/extra/`.

## 1. Are the results luck? (seeds + bootstrap)

**Seeds.** The zero-shot transfers from `03_transfer.py` were rerun with ExtraTrees seeds 42–46, unchanged otherwise.
Seed 42 reproduces `results/transfer/transfer_metrics.csv` exactly (max |BA diff| = 0), which confirms the
shared code in `fdia_common.run_transfers` is equivalent to `03`.

| Pair | Main BA (mean ± std, thr 0.5) | Ablation BA |
|---|---|---|
| 5→9 | 0.557 ± 0.001 | 0.531 ± 0.001 |
| 5→14 | 0.786 ± 0.001 | 0.873 ± 0.002 |
| 9→5 | 0.844 ± 0.001 | 0.880 ± 0.000 |
| 9→14 | 0.835 ± 0.000 | 0.880 ± 0.000 |
| 14→5 | 0.877 ± 0.000 | 0.880 ± 0.000 |
| 14→9 | 0.565 ± 0.003 | 0.571 ± 0.002 |

The largest std across all 36 setting/threshold combinations is 0.010 (ablation 14→9, best-BA threshold).
Seed variation is far smaller than the gaps the report draws conclusions from (e.g. 0.88 vs 0.56).
Full data: `seeds_metrics.csv`, `seeds_summary.csv`.

**Bootstrap.** These are 95% intervals from resampling target rows (2000 draws), for every saved score file at each of the 3 thresholds.
The widest half-width is 0.0014 BA. Caveat: rows are treated as independent. Neighbouring time steps are
correlated, so these intervals are too narrow; a block bootstrap would be the upgrade. File: `bootstrap_ci.csv`.

**Conclusion:** the original conclusions hold under seed changes and test-set resampling. Only 5 seeds were run, and the tuning budget is still small.

## 2. Fixing transfers into 9-bus — label-free adaptation (NOT zero-shot)

The report found that 9-bus normal data has 2.8–28.6× wider spread, so normal rows score as attacks.
The fix is to robust-scale each grid by its own reference rows before training and scoring: per column, (x − median) / IQR. The model, features and thresholds are the same as in `03`.

- **scaled_all:** reference = the grid's first 70% of rows in time order. **No labels used.**
- **scaled_clean:** reference = normal rows in the grid's first 15% of time. This assumes a known-clean calibration window, so it uses labels to pick those rows.

The same rule applies whether a grid is source or target. Target metrics are computed on all target rows, as in `03`.

**This is not zero-shot.** The target grid's unlabeled measurements are used to compute scaling statistics
(transductive / unsupervised domain adaptation). The zero-shot results in `REPORT.md` remain the main result.

Target BA at threshold 0.5 (seed 42):

| Pair | Zero-shot main | scaled_all main | scaled_clean main | Zero-shot ablation | scaled_all ablation | scaled_clean ablation |
|---|---|---|---|---|---|---|
| 5→9 | 0.556 | 0.624 | 0.627 | 0.530 | **0.878** | **0.878** |
| 5→14 | 0.787 | 0.797 | 0.797 | 0.872 | 0.880 | 0.880 |
| 9→5 | 0.845 | 0.837 | 0.837 | 0.880 | 0.880 | 0.880 |
| 9→14 | 0.835 | 0.826 | 0.836 | 0.880 | 0.880 | 0.880 |
| 14→5 | 0.877 | 0.877 | 0.878 | 0.880 | 0.880 | 0.880 |
| 14→9 | 0.562 | **0.856** | **0.863** | 0.572 | **0.878** | **0.879** |

Findings:
1. **The 9-bus failure was mostly scale miscalibration.** Scaling raises 3 of the 4 into-9-bus transfers from
   0.53–0.57 to 0.86–0.88, and 9-bus FPR at 0.5 falls from 0.56–0.92 to 0.00–0.02.
2. **With the scaled ablation features, all 6 directions reach ≈ 0.88 BA**, about 0.02 below within-grid (≈ 0.90).
3. **Labels are not needed for the fix.** scaled_all (no labels) is within 0.01 BA of scaled_clean in every direction.
4. **5→9 main stays weak (0.62, AUC 0.63).** Its ranking is poor, not just miscalibrated. It has only 20 shared
   columns (5 measurements × 4 lags). The agnostic ablation features do not have this limit.
5. Scaling does not hurt the directions that already worked; the largest drop is 0.009 BA (9→14 main, scaled_all).
6. The pattern is the same for the source-calibrated thresholds (p95, best-BA with FPR ≤ 5%). File: `adapt_metrics.csv`.

Limits: the adaptation runs used seed 42 only. Their bootstrap intervals are in `bootstrap_ci.csv`, with the same independence caveat as section 1.
Recall stays at ≈ 0.76. The attack types that went undetected in `REPORT.md` were not re-analysed here.

## Reproduce
```
python 05_adapt.py      # ~25 min, writes results/extra/adapt_metrics.csv + scores_scaled_*.npz
python 06_seeds.py      # ~65 min for 5 seeds; resumable (keeps seeds already in seeds_metrics.csv)
python 07_bootstrap.py  # seconds; needs .npz from 03 and 05
```

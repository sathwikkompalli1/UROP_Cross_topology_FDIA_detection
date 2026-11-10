# Cross-Topology FDIA Detection — Design Spec

Date: 2026-11-07. Source of requirements: `plan.md`.

## Goal
Reproducible comparison of RF / XGBoost / Extra Trees per grid (5, 9, 14-bus), then six
directed zero-shot transfers, threshold study, and honest write-up. Prior notebook results
are reported as "prior, not rerun" — never mixed with new numbers.

## Verified facts (2026-11-07)
- Files: `measurements_{5,9,14}bus_final.csv`, 500,000 rows each, 38/70/134 columns, no NaN.
- Label `label_multi` in 0..30. 0 = normal (y=0), 1..30 = attack (y=1). ~40% normal / 60% attack.
- Features: `Flow_i_j_t-k`, `Inj_n_t-k`, k = 0..3. `Timestamp` excluded (unique, monotonic, 1 per minute).
- Lag structure: row i+1 `t-1` == row i `t-0` (verified on 5-bus).
- Name-shared features: 5∩9 = 20, 5∩14 = 28, 9∩14 = 44, all three = 20 (matches prior report).
- Machine: 14 logical cores, 15.5 GB RAM.

## Decisions
1. **Split:** chronological 70 / 15 / 15 (train / val / test) per grid. Reason: lag columns
   copy previous rows, so a random split leaks test values into training rows.
2. **Labels:** binary as above. Features: every numeric column except `Timestamp`, `label_multi`.
   Loaded as float32.
3. **Models:** RandomForest, ExtraTrees (sklearn), XGBClassifier (hist). Seed 42.
4. **Tuning budget:** ~4 configs per model, fitted on a 150k-row seeded random sample of the
   train split, scored on val. Best config refit on full train.
5. **Selection metric:** val balanced accuracy at threshold 0.5. Guardrails: val FPR and recall
   reported; ties (< 0.001 BA) broken by lower FPR, then shorter training time.
6. **Test split:** used once, after selection, for within-grid reporting only.
7. **Transfers (main):** for each source→target, ordered name intersection; new instance of the
   source winner (its tuned config) fit on source train restricted to that intersection;
   thresholds calibrated on source val; scored on full target dataset.
7b. **User decision (2026-11-10):** the rule picked ET (5-bus), RF (9-bus), ET (14-bus); all three
   algos were within 0.003 val BA. Transfers use **ExtraTrees for every source** (n_estimators=200,
   max_depth=None, min_samples_leaf=5 — best ET config on all grids) so algorithm is not a confound
   across directions. Reported as researcher choice, not as the rule's output.
8. **Ablation (option B, labelled "ablation"):** topology-agnostic features — per lag k, over all
   Flow columns and over all Inj columns: mean, std, min, max, mean |x| → 2 × 5 × 4 = 40 features.
   Same six transfers, same protocol.
9. **Thresholds:** (a) 0.5; (b) 95th percentile of source-val normal scores; (c) argmax source-val
   balanced accuracy subject to source-val FPR ≤ 5%; (d) sweep 0.1..0.9 step 0.1.
   (a–c) are the valid zero-shot results. Sweep on target labels = **oracle/diagnostic only**.
10. **Metrics:** accuracy, precision, recall, F1, specificity, FPR, balanced accuracy, TN/FP/FN/TP,
    ROC-AUC, PR-AUC, train time.

## Layout
```
fdia_common.py        load, split, features, metrics, threshold rules (+ assert self-checks)
01_audit.py           -> results/audit/
02_within.py          -> results/within/  (tuning table, test table, winners.json)
03_transfer.py        -> results/transfer/ (metrics CSV, saved scores .npz), main + ablation
04_thresholds.py      -> results/thresholds/, figures/ (sweeps, hist, ROC/PR, confusion)
REPORT.md, README.md, requirements.txt
```
Notebook `FDIA_Final_Experiments.ipynb` left untouched.

## Checkpoints with user
- After stage 2: report winner per grid; wait for approval before stages 3+.

## Known limitations (to discuss in report)
- Name-matched features are not physically equivalent across grids (bus/line numbering differs).
- Synthetic data; attacks appear nearly i.i.d. per row (mean label run ≈ 1.2 rows).
- Small tuning budget.

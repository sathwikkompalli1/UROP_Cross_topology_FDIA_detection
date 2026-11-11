# Handoff — Cross-Topology FDIA Detection

Date: 2026-11-11. Status: **main study complete and on GitHub; extra stages 05–07 done, not yet committed.**
The project can stop here — everything left is optional future research (section 7).

## 1. What this project does
Detect False Data Injection Attacks (FDIA) on 5-, 9- and 14-bus grids, then test whether a detector
trained on one grid works on another grid with no retraining (zero-shot transfer). Requirements: `plan.md`.

## 2. State right now
- All code, results, figures and write-ups are in this folder.
- Git: commit `b551e65` ("project") is pushed to `origin/main`
  (`github.com/sathwikkompalli1/UROP_Cross_topology_FDIA_detection`). It contains stages 01–04.
- **Not committed yet:** `05_adapt.py`, `06_seeds.py`, `07_bootstrap.py`, `REPORT_EXTRA.md`, `results/extra/`,
  this file, and additions to `fdia_common.py` + `tests/test_common.py`.
- Existing results, figures, `REPORT.md` and `README.md` were not changed by the extra work.
  The additions to `fdia_common.py` are append-only; `make_model` gained an optional `seed` argument (default unchanged).
- `.gitignore` blocks: the 3 data CSVs (~1.5 GB), `results/models/`, `*.npz`, caches, `.superpowers/`.
- Tests: 8/8 passing (`python -m pytest tests -q`).

## 3. Files
| File | Role |
|---|---|
| `fdia_common.py` | Shared code: load, chronological split, features, metrics, threshold rules, model factory; plus `robust_scale`, `run_transfers` for stages 05–06 |
| `01_audit.py` | Data checks → `results/audit/` |
| `02_within.py` | RF / ET / XGB per grid → `results/within/` |
| `03_transfer.py` | 6 directed transfers, main + ablation → `results/transfer/` |
| `04_thresholds.py` | Threshold sweeps, figures, per-attack recall, feature shift → `results/thresholds/`, `figures/` |
| `05_adapt.py` | Extra: label-free robust scaling per grid (NOT zero-shot) → `results/extra/adapt_metrics.csv` |
| `06_seeds.py` | Extra: zero-shot transfers with seeds 42–46, resumable → `results/extra/seeds_*.csv` |
| `07_bootstrap.py` | Extra: 95% bootstrap intervals from saved scores → `results/extra/bootstrap_ci.csv` |
| `tests/test_common.py` | Unit tests for `fdia_common.py` |
| `REPORT.md` | Main results and analysis (read this first) |
| `REPORT_EXTRA.md` | Seeds, bootstrap intervals, 9-bus fix |
| `README.md` | Setup and run order (stages 01–04 only; does not mention 05–07 yet) |
| `docs/superpowers/specs/…design.md` | Design decisions |
| `docs/superpowers/plans/…md` | Implementation plan |
| `FDIA_Final_Experiments.ipynb`, `final_cleaned_transfer_results.csv` | Earlier Colab baseline — "prior, not rerun" |

Run order and timings: see `README.md` (step 2 ~1.5 h, step 3 ~20 min). Step 4 needs the `.npz` files
from step 3, which are not in git, so rerun step 3 first on a fresh clone. Extra stages: `05` ~25 min,
`06` ~65 min, `07` seconds (needs `.npz` from 03 and 05). Running 05 and 06 together hit low memory once
on this 15.5 GB machine; run them one at a time with other apps closed.

## 4. Key decisions (all approved by the user)
1. **Chronological 70/15/15 split** — lag columns copy the previous row, so a random split leaks.
2. Binary label: `label_multi` 0 = normal, 1–30 = attack. `Timestamp` excluded.
3. Model selection on validation balanced accuracy (BA) only; test used once.
4. **ExtraTrees for every transfer source** (200 trees, no depth limit, min leaf 5) — user choice so
   the algorithm is constant across directions (the selection rule had picked RF for 9-bus).
5. Option B: main transfer on name-shared columns **plus** an ablation with 40 topology-agnostic
   features (per lag: mean/std/min/max/mean|x| over Flow and over Inj columns).
6. Deployable thresholds use source validation only (0.5, p95 of normal scores, best BA with FPR ≤ 5%).
   Target-label sweeps are labelled ORACLE / diagnostic only.
7. **9-bus fix is reported separately as "label-free adaptation, NOT zero-shot"**; zero-shot stays the
   main result. Two variants, both shown: `scaled_all` (median/IQR from the grid's first 70% of rows, no labels)
   and `scaled_clean` (median/IQR from normal rows in the grid's first 15%, a known-clean window).
   Same rule for a grid whether it is source or target.
8. Uncertainty: bootstrap intervals on saved scores **and** 5 seeds (42–46) for the zero-shot transfers only.

## 5. Results in one screen
Within-grid: RF, ET and XGB all reach ≈ 0.90 test BA with FPR ≈ 0; recall ceiling ≈ 0.80.

Transfers (ExtraTrees, threshold 0.5, target BA). Zero-shot = 5-seed mean; scaled = `scaled_all`, seed 42:

| Pair | Zero-shot main | Zero-shot ablation | Scaled main | Scaled ablation |
|---|---|---|---|---|
| 5→9 | 0.557 | 0.531 | 0.624 | **0.878** |
| 5→14 | 0.786 | 0.873 | 0.797 | 0.880 |
| 9→5 | 0.844 | 0.880 | 0.837 | 0.880 |
| 9→14 | 0.835 | 0.880 | 0.826 | 0.880 |
| 14→5 | 0.877 | 0.880 | 0.877 | 0.880 |
| 14→9 | 0.565 | 0.571 | **0.856** | **0.878** |

- **Results are not luck.** Seed std ≤ 0.003 BA at 0.5 (≤ 0.010 at any threshold); bootstrap half-width
  ≤ 0.0014. Seed 42 reproduces `transfer_metrics.csv` exactly.
- **Zero-shot:** 4/6 directions transfer well; every transfer into 9-bus fails (FPR 0.56–0.92) because
  9-bus normal data has 2.8–28.6× wider spread.
- **With per-grid scaling, the 9-bus failure is mostly gone:** 3 of 4 into-9-bus transfers reach 0.86–0.88,
  9-bus FPR drops to 0.00–0.02, and the scaled ablation reaches ≈ 0.88 in all 6 directions (≈ 0.02 below
  within-grid). No labels are needed: `scaled_all` is within 0.01 of `scaled_clean` everywhere.
- 5→9 main stays weak (0.62, AUC 0.63): only 20 shared columns, poor ranking — not a calibration problem.
- Attack types 5, 10, 15, 20, 25, 30 (≈ 20% of attacks) are almost never detected by the transfer models.

## 6. Known gaps / open issues
- Bootstrap treats rows as independent; time-correlated rows make intervals too narrow (block bootstrap = upgrade).
- Adaptation (05) ran with seed 42 only; small tuning budget overall (4 configs, 150k-row sample).
- Name-matched columns are not physically the same bus/line across grids.
- Per-attack-type recall was computed for transfer models only, not within-grid models.
- "Stealthy attack" explanation for the undetected types is a hypothesis (attack generator not available).
- Minor code notes: `thr_p95` can flag > 5% when scores tie; `metrics` returns BA on single-class input.
- 14-bus ET refit took 1088 s (others ~100–200 s); cause not checked.

## 7. Suggested next steps
1. Commit and push the extra work (check `git status`: no `measurements_*.csv`, no `.npz`).
2. Link `REPORT_EXTRA.md` from `README.md` and `REPORT.md`; add stages 05–07 to the README run order.
3. Optional research: per-attack recall for within-grid models (cheap), then analyse the undetected attack
   types (e.g. state-estimation residual features).
4. Optional: physically aligned / graph-aware features (per-unit values, GNN over the bus graph).

## 8. Environment
Python 3.13, pandas 3.0.5, numpy 2.5.2, scikit-learn 1.9.0, xgboost 3.4.1, matplotlib 3.11.1,
pytest 9.1.1 (pinned in `requirements.txt`). Seeds 42 (main), 42–46 (06). Ran on 14 cores, 15.5 GB RAM.

# Cross-Topology FDIA Detection

Zero-shot transfer of false-data-injection-attack detectors between 5-, 9- and 14-bus grids.
Findings: [`REPORT.md`](REPORT.md). Follow-up (5 seeds, bootstrap intervals, 9-bus fix): [`REPORT_EXTRA.md`](REPORT_EXTRA.md). Design: [`docs/superpowers/specs/2026-11-07-fdia-cross-topology-design.md`](docs/superpowers/specs/2026-11-07-fdia-cross-topology-design.md).
Plan: [`docs/superpowers/plans/2026-11-07-fdia-cross-topology.md`](docs/superpowers/plans/2026-11-07-fdia-cross-topology.md).

## Setup
```bash
pip install -r requirements.txt
```
Put the three data files in the repo root (not in git, ~1.5 GB):
`measurements_5bus_final.csv`, `measurements_9bus_final.csv`, `measurements_14bus_final.csv`.

## Run (in order)
| Step | Command | Time* | Output |
|---|---|---|---|
| tests | `python -m pytest tests -q` | 5 s | 8 passing |
| 1 audit | `python 01_audit.py` | 30 s | `results/audit/` |
| 2 within-grid | `python 02_within.py [grid ...]` | ~1.5 h | `results/within/` (tuning, test, winners.json) |
| 3 transfers | `python 03_transfer.py` | ~20 min | `results/transfer/transfer_metrics.csv`, `scores_*.npz` |
| 4 thresholds | `python 04_thresholds.py` | ~5 min | `results/thresholds/`, `figures/` (48 PNGs) |

*14 logical cores, 15.5 GB RAM. Seed 42 everywhere. Step 4 needs the `.npz` files from step 3 (gitignored).

Extra stages `05_adapt.py`, `06_seeds.py`, `07_bootstrap.py` → `results/extra/`: see [`REPORT_EXTRA.md`](REPORT_EXTRA.md#reproduce).

## Outputs
- `results/audit/audit.csv`, `shared_features.csv` — data checks, shared-column matrix.
- `results/within/test.csv` — RF/ET/XGB × 3 grids, val + test metrics; `tuning.csv` all 36 configs.
- `results/transfer/transfer_metrics.csv` — 2 modes × 6 pairs × 3 deployable thresholds.
- `results/thresholds/sweep_source_val.csv` — valid sweep. `sweep_target_ORACLE.csv`,
  `oracle_best_threshold_DIAGNOSTIC.csv` — **target-label diagnostics, not deployable**.
- `results/thresholds/recall_by_attack_type.csv`, `normal_feature_shift.csv`.
- `figures/{sweep,hist,rocpr,cm}_{main|ablation}_{src}_{tgt}.png`.

`FDIA_Final_Experiments.ipynb` is the earlier Colab baseline; kept unchanged. Its results are in
`final_cleaned_transfer_results.csv` ("prior, not rerun").

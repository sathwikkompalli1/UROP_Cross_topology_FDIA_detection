"""Extra stage 5: label-free adaptation -- NOT zero-shot. Leaves 03_transfer.py results untouched.

Each grid is robust-scaled (per column: (x - median) / IQR) by its own reference rows, then the same
ExtraTrees transfers as 03_transfer.py are run (6 main + 3 ablation fits, seed 42).
  scaled_all_*   : ref = first 70% of the grid's rows (time order), no labels used
  scaled_clean_* : ref = normal rows in the grid's first 15% of time (assumes a known-clean window)
Same rule for a grid whether it is source or target; target metrics are on all target rows, as in 03.
Outputs: results/extra/adapt_metrics.csv, results/extra/scores_scaled_*.npz
"""
import numpy as np
import pandas as pd

from fdia_common import GRIDS, ROOT, agnostic_features, load_grid, robust_scale, run_transfers

OUT = ROOT / "results" / "extra"
OUT.mkdir(parents=True, exist_ok=True)

data = {g: load_grid(g) for g in GRIDS}
y = {g: data[g][1] for g in GRIDS}


def ref_rows(g, variant):
    n = len(y[g])
    idx = np.arange(n)
    return idx < int(n * 0.70) if variant == "all" else (idx < int(n * 0.15)) & (y[g] == 0)


rows = []
for variant in ("all", "clean"):
    X = {g: robust_scale(data[g][0], ref_rows(g, variant)) for g in GRIDS}
    agn = {g: agnostic_features(X[g]) for g in GRIDS}
    rows += run_transfers(X, agn, y, prefix=f"scaled_{variant}_", save_dir=OUT)
    pd.DataFrame(rows).to_csv(OUT / "adapt_metrics.csv", index=False)
    del X, agn

print(pd.DataFrame(rows).query("threshold_method == 'fixed_0.5'")
      [["mode", "source", "target", "balanced_accuracy", "recall", "fpr", "roc_auc"]].round(4).to_string(index=False))

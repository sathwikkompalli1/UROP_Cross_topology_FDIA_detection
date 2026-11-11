"""Extra stage 6: rerun the zero-shot transfers of 03_transfer.py with 5 seeds (42-46) -> mean +- std.

Resumable: seeds already in seeds_metrics.csv are kept, not rerun.
Seed 42 must reproduce results/transfer/transfer_metrics.csv; the check is printed at the end.
Outputs: results/extra/seeds_metrics.csv (every run), results/extra/seeds_summary.csv (mean/std/min/max)
"""
import pandas as pd

from fdia_common import GRIDS, ROOT, agnostic_features, load_grid, run_transfers

OUT = ROOT / "results" / "extra"
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [42, 43, 44, 45, 46]
KEY = ["mode", "source", "target", "threshold_method"]

data = {g: load_grid(g) for g in GRIDS}
X = {g: data[g][0] for g in GRIDS}
y = {g: data[g][1] for g in GRIDS}
agn = {g: agnostic_features(X[g]) for g in GRIDS}

done = OUT / "seeds_metrics.csv"
rows = pd.read_csv(done).to_dict("records") if done.exists() else []  # resume: keep finished seeds
for seed in [s for s in SEEDS if s not in {r["seed"] for r in rows}]:
    rows += run_transfers(X, agn, y, seed=seed)
    pd.DataFrame(rows).to_csv(OUT / "seeds_metrics.csv", index=False)

df = pd.DataFrame(rows)
summ = df.groupby(KEY)[["balanced_accuracy", "recall", "fpr", "roc_auc"]].agg(["mean", "std", "min", "max"])
summ.columns = ["_".join(c) for c in summ.columns]
summ.reset_index().to_csv(OUT / "seeds_summary.csv", index=False)

old = pd.read_csv(ROOT / "results" / "transfer" / "transfer_metrics.csv").set_index(KEY)["balanced_accuracy"]
new = df[df.seed == 42].set_index(KEY)["balanced_accuracy"]
print(f"seed-42 reproduction: max |BA diff| vs transfer_metrics.csv = {(old - new.reindex(old.index)).abs().max():.2e}")
print(summ.reset_index().query("threshold_method == 'fixed_0.5'")
      [KEY[:3] + ["balanced_accuracy_mean", "balanced_accuracy_std"]].round(4).to_string(index=False))

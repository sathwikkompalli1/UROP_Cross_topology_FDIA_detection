"""Stage 2: within-topology RF / ET / XGB comparison.

Usage: python 02_within.py [grid ...]   (default: all grids)
Tuning on a 150k sample of the chronological train split, scored on val @0.5.
Best config per algo refit on full train. Winner per grid by val BA (tie <0.001 -> lower FPR -> faster).
Test split is scored only after selection, for reporting.
"""
import json
import sys
import time

import numpy as np
import pandas as pd

from fdia_common import GRIDS, ROOT, SEARCH, SEED, chrono_split, load_grid, make_model, metrics

OUT = ROOT / "results" / "within"
OUT.mkdir(parents=True, exist_ok=True)
TUNE_N = 150_000


def pick(df):
    """Row with best val BA; ties (<0.001) broken by lower val FPR, then lower train time."""
    near = df[df.val_balanced_accuracy >= df.val_balanced_accuracy.max() - 0.001]
    return near.sort_values(["val_fpr", "train_sec"]).iloc[0]


def fit_score(algo, params, Xtr, ytr, Xva, yva):
    m = make_model(algo, params)
    t0 = time.perf_counter()
    m.fit(Xtr, ytr)
    sec = time.perf_counter() - t0
    return m, sec, {f"val_{k}": v for k, v in metrics(yva, m.predict_proba(Xva)[:, 1], 0.5).items()}


for grid in sys.argv[1:] or GRIDS:
    X, y = load_grid(grid)
    tr, va, te = chrono_split(len(y))
    Xtr, ytr, Xva, yva, Xte, yte = X[tr], y[tr], X[va], y[va], X[te], y[te]
    idx = np.sort(np.random.default_rng(SEED).choice(len(ytr), TUNE_N, replace=False))
    Xs, ys = Xtr.iloc[idx], ytr[idx]

    tuning = []
    for algo, configs in SEARCH.items():
        for params in configs:
            _, sec, m = fit_score(algo, params, Xs, ys, Xva, yva)
            tuning.append(dict(grid=grid, algo=algo, params=json.dumps(params), train_sec=sec, **m))
            print(f"[tune] {grid} {algo} {params} BA={m['val_balanced_accuracy']:.4f} FPR={m['val_fpr']:.4f} {sec:.0f}s", flush=True)
    tuning = pd.DataFrame(tuning)
    tuning.to_csv(OUT / f"tuning_{grid}.csv", index=False)

    final = []
    for algo in SEARCH:
        best = pick(tuning[tuning.algo == algo])
        params = json.loads(best.params)
        model, sec, m = fit_score(algo, params, Xtr, ytr, Xva, yva)
        p_te = model.predict_proba(Xte)[:, 1]
        final.append(dict(grid=grid, algo=algo, params=best.params, train_sec=sec, **m,
                          **{f"test_{k}": v for k, v in metrics(yte, p_te, 0.5).items()}))
        print(f"[final] {grid} {algo} valBA={m['val_balanced_accuracy']:.4f} {sec:.0f}s", flush=True)
    final = pd.DataFrame(final)
    final["winner"] = final.algo == pick(final).algo
    final.to_csv(OUT / f"test_{grid}.csv", index=False)
    del X, Xtr, Xva, Xte, Xs

# Combine whatever grids have been run so far.
parts = [pd.read_csv(OUT / f"test_{g}.csv") for g in GRIDS if (OUT / f"test_{g}.csv").exists()]
allf = pd.concat(parts, ignore_index=True)
allf.to_csv(OUT / "test.csv", index=False)
pd.concat([pd.read_csv(OUT / f"tuning_{g}.csv") for g in GRIDS if (OUT / f"tuning_{g}.csv").exists()]).to_csv(OUT / "tuning.csv", index=False)
winners = {r.grid: dict(algo=r.algo, params=json.loads(r.params), val_ba=float(r.val_balanced_accuracy))
           for r in allf[allf.winner].itertuples()}
(OUT / "winners.json").write_text(json.dumps(winners, indent=2))
cols = ["grid", "algo", "val_balanced_accuracy", "val_fpr", "val_recall", "test_balanced_accuracy",
        "test_fpr", "test_recall", "test_roc_auc", "train_sec", "winner"]
print(allf[cols].round(4).to_string(index=False))

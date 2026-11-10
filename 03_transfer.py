"""Stage 3: six directed zero-shot transfers (main = name-matched shared columns, ablation = 40 topology-agnostic features).

Model: ExtraTrees for every source (user decision, see spec 7b). Fit on source train only;
thresholds from source val only; target labels used only for scoring.
Outputs: results/transfer/transfer_metrics.csv, results/transfer/scores_{mode}_{src}_{tgt}.npz
"""
import time

import numpy as np
import pandas as pd

from fdia_common import (GRIDS, PAIRS, ROOT, agnostic_features, chrono_split, load_grid, make_model,
                         metrics, thr_best_ba, thr_p95, transfer_features)

OUT = ROOT / "results" / "transfer"
OUT.mkdir(parents=True, exist_ok=True)
ALGO, PARAMS = "et", dict(n_estimators=200, max_depth=None, min_samples_leaf=5)

data = {g: load_grid(g) for g in GRIDS}  # ~0.6 GB float32 for all three
agn = {g: agnostic_features(data[g][0]) for g in GRIDS}

rows = []


def run(mode, src, targets, cols_for):
    """Fit one model on source train, score source val + each target on the same ordered columns."""
    Xs, ys = (agn[src], data[src][1]) if mode == "ablation" else data[src]
    cols = cols_for(targets[0])
    tr, va, _ = chrono_split(len(ys))
    model = make_model(ALGO, PARAMS)
    t0 = time.perf_counter()
    model.fit(Xs.iloc[tr][cols], ys[tr])
    sec = time.perf_counter() - t0
    p_val, y_val = model.predict_proba(Xs.iloc[va][cols])[:, 1], ys[va]
    thr = {"fixed_0.5": 0.5, "p95_normal": thr_p95(p_val, y_val), "best_ba_fpr5": thr_best_ba(p_val, y_val)}
    for tgt in targets:
        assert cols_for(tgt) == cols  # same ordered features for fit and predict
        Xt, yt = (agn[tgt], data[tgt][1]) if mode == "ablation" else data[tgt]
        p_tgt = model.predict_proba(Xt[cols])[:, 1]
        np.savez_compressed(OUT / f"scores_{mode}_{src}_{tgt}.npz", p_val=p_val, y_val=y_val, p_tgt=p_tgt, y_tgt=yt)
        for name, t in thr.items():
            rows.append(dict(mode=mode, source=src, target=tgt, n_features=len(cols), algo=ALGO,
                             train_sec=sec, threshold_method=name, threshold=t, **metrics(yt, p_tgt, t)))
        m = metrics(yt, p_tgt, 0.5)
        print(f"[{mode}] {src}->{tgt} feats={len(cols)} fit={sec:.0f}s BA@0.5={m['balanced_accuracy']:.4f} "
              f"FPR@0.5={m['fpr']:.4f} AUC={m['roc_auc']:.4f}", flush=True)
        pd.DataFrame(rows).to_csv(OUT / "transfer_metrics.csv", index=False)


for src, tgt in PAIRS:  # main: shared columns differ per pair -> one fresh model per pair
    run("main", src, [tgt], lambda t, s=src: transfer_features(s, t))
for src in GRIDS:  # ablation: same 40 columns for every target -> one model per source serves both
    run("ablation", src, [t for s, t in PAIRS if s == src], lambda t: list(agn[src].columns))

print(pd.DataFrame(rows).query("threshold_method == 'fixed_0.5'")
      [["mode", "source", "target", "n_features", "balanced_accuracy", "recall", "fpr", "roc_auc"]].round(4).to_string(index=False))

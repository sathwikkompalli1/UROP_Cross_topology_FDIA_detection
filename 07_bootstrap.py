"""Extra stage 7: 95% bootstrap intervals for target BA / recall / FPR from saved scores (no retraining).

Reads results/transfer/scores_*.npz (zero-shot, from 03) and results/extra/scores_*.npz (from 05).
Thresholds are recomputed from source validation exactly as in 03. Resampling target rows with replacement
only changes the TP/FN/TN/FP counts, so each bootstrap draw is one multinomial draw over those 4 cells.
Output: results/extra/bootstrap_ci.csv
"""
import numpy as np
import pandas as pd

from fdia_common import ROOT, SEED, metrics, thr_best_ba, thr_p95

B = 2000
OUT = ROOT / "results" / "extra"
OUT.mkdir(parents=True, exist_ok=True)
files = sorted((ROOT / "results" / "transfer").glob("scores_*.npz")) + sorted(OUT.glob("scores_*.npz"))
rng = np.random.default_rng(SEED)


def boot(c, size):
    """c = [TP, FN, TN, FP] -> bootstrap samples of (BA, recall, FPR)."""
    # shortcut: rows treated as independent; time-correlated rows make these intervals too narrow, upgrade = block bootstrap
    s = rng.multinomial(c.sum(), c / c.sum(), size=size)
    rec, fpr = s[:, 0] / (s[:, 0] + s[:, 1]), s[:, 3] / (s[:, 2] + s[:, 3])
    return (rec + 1 - fpr) / 2, rec, fpr


rows = []
for f in files:
    parts = f.stem.split("_")
    mode, src, tgt = "_".join(parts[1:-2]), parts[-2], parts[-1]
    z = np.load(f)
    thr = {"fixed_0.5": 0.5, "p95_normal": thr_p95(z["p_val"], z["y_val"]), "best_ba_fpr5": thr_best_ba(z["p_val"], z["y_val"])}
    for name, t in thr.items():
        m = metrics(z["y_tgt"], z["p_tgt"], t)
        samples = boot(np.array([m["TP"], m["FN"], m["TN"], m["FP"]], float), B)
        for metric, s in zip(("balanced_accuracy", "recall", "fpr"), samples):
            rows.append(dict(mode=mode, source=src, target=tgt, threshold_method=name, threshold=t, metric=metric,
                             value=m[metric], ci_low=np.quantile(s, 0.025), ci_high=np.quantile(s, 0.975)))

df = pd.DataFrame(rows)
df.to_csv(OUT / "bootstrap_ci.csv", index=False)
print(df.query("threshold_method == 'fixed_0.5' and metric == 'balanced_accuracy'")
      [["mode", "source", "target", "value", "ci_low", "ci_high"]].round(4).to_string(index=False))

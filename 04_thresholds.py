"""Stage 4: threshold sweeps and figures from saved transfer scores (no retraining).

Valid (source-only): sweep on source validation.  ORACLE (diagnostic only): sweep on target labels.
Outputs: results/thresholds/sweep_source_val.csv, results/thresholds/sweep_target_ORACLE.csv, figures/*.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from fdia_common import GRIDS, PAIRS, ROOT, metrics, transfer_features

TR = ROOT / "results" / "transfer"
OUT = ROOT / "results" / "thresholds"
FIG = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(exist_ok=True)
SWEEP = np.round(np.arange(1, 10) / 10, 1)
SHOW = ["recall", "fpr", "specificity", "precision", "f1", "balanced_accuracy"]
deploy = pd.read_csv(TR / "transfer_metrics.csv")

val_rows, tgt_rows = [], []
for mode in ("main", "ablation"):
    for src, tgt in PAIRS:
        tag = f"{mode}_{src}_{tgt}"
        z = np.load(TR / f"scores_{tag}.npz")
        sv = pd.DataFrame([dict(mode=mode, source=src, target=tgt, threshold=t, **metrics(z["y_val"], z["p_val"], t)) for t in SWEEP])
        st = pd.DataFrame([dict(mode=mode, source=src, target=tgt, threshold=t, **metrics(z["y_tgt"], z["p_tgt"], t)) for t in SWEEP])
        val_rows.append(sv); tgt_rows.append(st)
        d = deploy[(deploy["mode"] == mode) & (deploy.source == src) & (deploy.target == tgt)]
        title = f"{mode}: {src} -> {tgt}"

        # 1) threshold vs metrics
        fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        for a, s, name in ((ax[0], sv, "source validation (valid)"), (ax[1], st, "target - ORACLE, diagnostic only")):
            for m in SHOW:
                a.plot(s.threshold, s[m], marker="o", label=m)
            a.set(title=name, xlabel="threshold", ylim=(0, 1.02)); a.grid(alpha=.3)
        ax[0].legend(fontsize=8); fig.suptitle(title); fig.tight_layout()
        fig.savefig(FIG / f"sweep_{tag}.png", dpi=110); plt.close(fig)

        # 2) score histograms with deployable thresholds
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for a, p, y, name in ((ax[0], z["p_val"], z["y_val"], f"source val ({src})"), (ax[1], z["p_tgt"], z["y_tgt"], f"target ({tgt})")):
            a.hist(p[y == 0], bins=50, range=(0, 1), alpha=.6, density=True, label="normal")
            a.hist(p[y == 1], bins=50, range=(0, 1), alpha=.6, density=True, label="attack")
            for r in d.itertuples():
                a.axvline(r.threshold, ls="--", c="k", lw=.8)
            a.set(title=name, xlabel="attack probability"); a.legend()
        fig.suptitle(f"{title}  (dashed: 0.5 / p95 / best-BA thresholds)"); fig.tight_layout()
        fig.savefig(FIG / f"hist_{tag}.png", dpi=110); plt.close(fig)

        # 3) ROC + PR
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        for p, y, name in ((z["p_val"], z["y_val"], "source val"), (z["p_tgt"], z["y_tgt"], "target")):
            fpr, tpr, _ = roc_curve(y, p); ax[0].plot(fpr, tpr, label=name)
            pr, rc, _ = precision_recall_curve(y, p); ax[1].plot(rc, pr, label=name)
        ax[0].plot([0, 1], [0, 1], "k:", lw=.8)
        ax[0].set(title="ROC", xlabel="FPR", ylabel="TPR"); ax[1].set(title="PR", xlabel="recall", ylabel="precision")
        ax[0].legend(); fig.suptitle(title); fig.tight_layout()
        fig.savefig(FIG / f"rocpr_{tag}.png", dpi=110); plt.close(fig)

        # 4) target confusion matrices at deployable thresholds
        fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
        for a, r in zip(ax, d.itertuples()):
            cm = np.array([[r.TN, r.FP], [r.FN, r.TP]])
            a.imshow(cm, cmap="Blues")
            for i in range(2):
                for j in range(2):
                    a.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color="red")
            a.set(title=f"{r.threshold_method} (t={r.threshold:.2f})", xticks=[0, 1], yticks=[0, 1],
                  xticklabels=["pred normal", "pred attack"], yticklabels=["normal", "attack"])
        fig.suptitle(f"{title}  target confusion"); fig.tight_layout()
        fig.savefig(FIG / f"cm_{tag}.png", dpi=110); plt.close(fig)
        print("done", tag, flush=True)

pd.concat(val_rows, ignore_index=True).to_csv(OUT / "sweep_source_val.csv", index=False)
tgt = pd.concat(tgt_rows, ignore_index=True)  # unique index needed for idxmax below
tgt.to_csv(OUT / "sweep_target_ORACLE.csv", index=False)
best = tgt.loc[tgt.groupby(["mode", "source", "target"], sort=False).balanced_accuracy.idxmax()]
best.to_csv(OUT / "oracle_best_threshold_DIAGNOSTIC.csv", index=False)
print(best[["mode", "source", "target", "threshold", "balanced_accuracy", "recall", "fpr"]].round(3).to_string(index=False))

# Recall per attack type (label_multi 1..30) on the target @0.5: which attacks are never caught?
labels = {g: pd.read_csv(ROOT / f, usecols=["label_multi"])["label_multi"].to_numpy() for g, f in GRIDS.items()}
rec = {}
for mode in ("main", "ablation"):
    for src, tgt in PAIRS:
        p = np.load(TR / f"scores_{mode}_{src}_{tgt}.npz")["p_tgt"]
        lab = labels[tgt]
        rec[f"{mode} {src}->{tgt}"] = pd.Series({k: (p[lab == k] >= 0.5).mean() for k in range(1, 31)})
rec = pd.DataFrame(rec).rename_axis("attack_type")
rec.to_csv(OUT / "recall_by_attack_type.csv")
print(rec.round(2).to_string())

# Feature-distribution shift on NORMAL rows of the name-shared t-0 columns: target std / source std, mean shift in source-std units.
shift = []
for src, tgt in PAIRS:
    cols = [c for c in transfer_features(src, tgt) if c.endswith("_t-0")]
    s, t = (pd.read_csv(ROOT / GRIDS[g], usecols=cols + ["label_multi"]).query("label_multi == 0")[cols] for g in (src, tgt))
    for c in cols:
        shift.append(dict(source=src, target=tgt, feature=c, std_ratio=t[c].std() / s[c].std(),
                          mean_shift_in_src_std=(t[c].mean() - s[c].mean()) / s[c].std()))
shift = pd.DataFrame(shift)
shift.to_csv(OUT / "normal_feature_shift.csv", index=False)
print(shift.groupby(["source", "target"], sort=False)[["std_ratio", "mean_shift_in_src_std"]]
      .agg(lambda x: f"{x.abs().min():.2f}-{x.abs().max():.2f}").to_string())

"""Shared helpers for the cross-topology FDIA experiments (see docs/superpowers/specs)."""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

SEED = 42
ROOT = Path(__file__).resolve().parent
GRIDS = {
    "5-bus": "measurements_5bus_final.csv",
    "9-bus": "measurements_9bus_final.csv",
    "14-bus": "measurements_14bus_final.csv",
}
PAIRS = [("5-bus", "9-bus"), ("5-bus", "14-bus"), ("9-bus", "5-bus"),
         ("9-bus", "14-bus"), ("14-bus", "5-bus"), ("14-bus", "9-bus")]
NON_FEATURES = {"Timestamp", "label_multi"}

SEARCH = {
    "rf": [dict(n_estimators=200, max_depth=d, min_samples_leaf=l) for d in (20, None) for l in (1, 5)],
    "et": [dict(n_estimators=200, max_depth=d, min_samples_leaf=l) for d in (20, None) for l in (1, 5)],
    "xgb": [dict(n_estimators=400, learning_rate=0.1, max_depth=d, subsample=s) for d in (6, 10) for s in (0.8, 1.0)],
}


def feature_columns(name):
    cols = pd.read_csv(ROOT / GRIDS[name], nrows=0).columns
    return [c for c in cols if c not in NON_FEATURES]


def load_grid(name):
    """Rows stay in file order (sorted by Timestamp). Label 0 = normal, 1..30 = attack."""
    feats = feature_columns(name)
    df = pd.read_csv(ROOT / GRIDS[name], usecols=feats + ["label_multi"],
                     dtype={**{c: np.float32 for c in feats}, "label_multi": np.int16})
    assert df["label_multi"].notna().all() and df["label_multi"].between(0, 30).all()
    y = (df.pop("label_multi").to_numpy() > 0).astype(np.int8)
    return df, y


def chrono_split(n):
    a, b = int(n * 0.70), int(n * 0.85)
    return slice(0, a), slice(a, b), slice(b, n)


def transfer_features(src, tgt):
    return sorted(set(feature_columns(src)) & set(feature_columns(tgt)))


def agnostic_features(X):
    """Per lag: mean/std/min/max/mean|x| over all Flow and all Inj columns -> 40 cols."""
    out = {}
    for k in range(4):
        for grp in ("Flow", "Inj"):
            v = X[[c for c in X.columns if c.startswith(grp + "_") and c.endswith(f"_t-{k}")]].to_numpy()
            out[f"{grp}_mean_t-{k}"] = v.mean(1)
            out[f"{grp}_std_t-{k}"] = v.std(1)  # ddof=0: no NaN
            out[f"{grp}_min_t-{k}"] = v.min(1)
            out[f"{grp}_max_t-{k}"] = v.max(1)
            out[f"{grp}_absmean_t-{k}"] = np.abs(v).mean(1)
    return pd.DataFrame(out, index=X.index).astype(np.float32)


def metrics(y, p, t):
    y = np.asarray(y)
    pred = np.asarray(p) >= t
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum()); tn = int((~pred & (y == 0)).sum())
    div = lambda a, b: a / b if b else 0.0
    rec, spec, prec = div(tp, tp + fn), div(tn, tn + fp), div(tp, tp + fp)
    both = len(np.unique(y)) == 2
    return dict(accuracy=div(tp + tn, len(y)), precision=prec, recall=rec,
                f1=div(2 * prec * rec, prec + rec), specificity=spec, fpr=div(fp, fp + tn),
                balanced_accuracy=(rec + spec) / 2,
                roc_auc=roc_auc_score(y, p) if both else np.nan,
                pr_auc=average_precision_score(y, p) if both else np.nan,
                TN=tn, FP=fp, FN=fn, TP=tp)


def thr_p95(p_val, y_val):
    return float(np.quantile(p_val[y_val == 0], 0.95))


def thr_best_ba(p_val, y_val, max_fpr=0.05):
    """Max source-val BA s.t. FPR <= max_fpr; if none feasible, restrict to lowest-FPR candidates."""
    p_n, p_a = np.sort(p_val[y_val == 0]), np.sort(p_val[y_val == 1])
    ts = np.round(np.arange(1, 100) / 100, 2)
    fpr = 1 - np.searchsorted(p_n, ts, side="left") / len(p_n)
    rec = 1 - np.searchsorted(p_a, ts, side="left") / len(p_a)
    ba = (rec + 1 - fpr) / 2
    ok = fpr <= max_fpr if (fpr <= max_fpr).any() else fpr == fpr.min()
    return float(ts[np.flatnonzero(ok)[np.argmax(ba[ok])]])


def make_model(algo, params, seed=SEED):
    if algo == "rf":
        return RandomForestClassifier(**params, class_weight="balanced", random_state=seed, n_jobs=-1)
    if algo == "et":
        return ExtraTreesClassifier(**params, class_weight="balanced", random_state=seed, n_jobs=-1)
    if algo == "xgb":
        return XGBClassifier(**params, tree_method="hist", random_state=seed, n_jobs=-1)
    raise ValueError(algo)


# ---- extra stages 05-07 (added after the main study; 01-04 do not use these) ----
TRANSFER_ET = dict(n_estimators=200, max_depth=None, min_samples_leaf=5)  # same as 03_transfer.py


def robust_scale(X, ref):
    """(X - median) / IQR per column, stats from rows where boolean mask `ref` is True. IQR 0 -> 1."""
    q1, med, q3 = np.quantile(X.to_numpy()[ref], [0.25, 0.5, 0.75], axis=0)
    iqr = np.where(q3 - q1 > 0, q3 - q1, 1.0)
    return ((X - med) / iqr).astype(np.float32)


def run_transfers(X, agn, y, prefix="", seed=SEED, save_dir=None):
    """Same 6 main + 3 ablation ExtraTrees fits as 03_transfer.py. X/agn/y: dicts keyed by grid. Returns metric rows."""
    rows = []
    jobs = [("main", s, [t], transfer_features(s, t)) for s, t in PAIRS] + \
           [("ablation", s, [t for a, t in PAIRS if a == s], list(agn[s].columns)) for s in GRIDS]
    for kind, src, tgts, cols in jobs:
        F, mode = (agn if kind == "ablation" else X), prefix + kind
        tr, va, _ = chrono_split(len(y[src]))
        model = make_model("et", TRANSFER_ET, seed)
        t0 = time.perf_counter()
        model.fit(F[src].iloc[tr][cols], y[src][tr])
        sec = time.perf_counter() - t0
        p_val, y_val = model.predict_proba(F[src].iloc[va][cols])[:, 1], y[src][va]
        thr = {"fixed_0.5": 0.5, "p95_normal": thr_p95(p_val, y_val), "best_ba_fpr5": thr_best_ba(p_val, y_val)}
        for tgt in tgts:
            p = model.predict_proba(F[tgt][cols])[:, 1]
            if save_dir:
                np.savez_compressed(save_dir / f"scores_{mode}_{src}_{tgt}.npz", p_val=p_val, y_val=y_val, p_tgt=p, y_tgt=y[tgt])
            for name, t in thr.items():
                rows.append(dict(mode=mode, seed=seed, source=src, target=tgt, n_features=len(cols), algo="et",
                                 train_sec=sec, threshold_method=name, threshold=t, **metrics(y[tgt], p, t)))
            m = metrics(y[tgt], p, 0.5)
            print(f"[{mode} seed={seed}] {src}->{tgt} fit={sec:.0f}s BA@0.5={m['balanced_accuracy']:.4f} "
                  f"FPR@0.5={m['fpr']:.4f} AUC={m['roc_auc']:.4f}", flush=True)
    return rows

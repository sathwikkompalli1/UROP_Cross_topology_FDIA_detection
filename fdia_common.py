"""Shared helpers for the cross-topology FDIA experiments (see docs/superpowers/specs)."""
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


def make_model(algo, params):
    if algo == "rf":
        return RandomForestClassifier(**params, class_weight="balanced", random_state=SEED, n_jobs=-1)
    if algo == "et":
        return ExtraTreesClassifier(**params, class_weight="balanced", random_state=SEED, n_jobs=-1)
    if algo == "xgb":
        return XGBClassifier(**params, tree_method="hist", random_state=SEED, n_jobs=-1)
    raise ValueError(algo)

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fdia_common import (agnostic_features, chrono_split, metrics, thr_best_ba,
                         transfer_features)


def test_split_sizes():
    a, b, c = chrono_split(1000)
    assert (a.stop, b.start, b.stop, c.start, c.stop) == (700, 700, 850, 850, 1000)


def test_threshold_inclusive():
    m = metrics(np.array([0, 1]), np.array([0.5, 0.5]), 0.5)
    assert m["TP"] == 1 and m["FP"] == 1


def test_single_class_no_crash():
    m = metrics(np.array([1, 1]), np.array([0.2, 0.9]), 0.5)
    assert np.isnan(m["roc_auc"]) and m["recall"] == 0.5


def test_transfer_features_known_counts():
    assert len(transfer_features("5-bus", "9-bus")) == 20
    assert len(transfer_features("5-bus", "14-bus")) == 28
    assert len(transfer_features("9-bus", "14-bus")) == 44
    assert transfer_features("9-bus", "5-bus") == transfer_features("5-bus", "9-bus")


def test_agnostic_shape_no_nan():
    cols = [f"{b}_t-{k}" for k in range(4) for b in ["Flow_1_2", "Flow_2_3", "Inj_2", "Inj_3"]]
    X = pd.DataFrame(np.random.default_rng(0).normal(size=(10, 16)), columns=cols)
    A = agnostic_features(X)
    assert A.shape == (10, 40) and not A.isna().any().any()


def test_thr_best_ba_respects_fpr():
    y = np.array([0] * 100 + [1] * 100)
    p = np.r_[np.linspace(0, .6, 100), np.linspace(.4, 1, 100)]
    assert metrics(y, p, thr_best_ba(p, y, 0.05))["fpr"] <= 0.05


def test_thr_best_ba_fallback():
    y = np.array([0, 1])
    p = np.array([0.999, 0.999])
    assert 0 < thr_best_ba(p, y, 0.05) <= 0.99


def test_robust_scale_ref_stats_and_zero_iqr():
    from fdia_common import robust_scale
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0, 100.0], "b": [7.0] * 6})
    ref = np.array([True] * 5 + [False])
    S = robust_scale(X, ref)
    assert S["a"].iloc[2] == 0 and S["a"].iloc[3] == 0.5  # median 3, IQR 2 from ref rows only
    assert (S["b"] == 0).all() and np.isfinite(S.to_numpy()).all()  # constant column: IQR 0 -> 1, no inf

"""Stage 1: data audit -> results/audit/{audit,shared_features}.csv"""
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from fdia_common import GRIDS, ROOT, chrono_split, feature_columns, load_grid, transfer_features

OUT = ROOT / "results" / "audit"
OUT.mkdir(parents=True, exist_ok=True)

rows = []
for name, file in GRIDS.items():
    X, y = load_grid(name)
    ts = pd.to_datetime(pd.read_csv(ROOT / file, usecols=["Timestamp"])["Timestamp"], format="%d/%m/%Y %H:%M")
    flow0 = next(c for c in X.columns if c.startswith("Flow_") and c.endswith("_t-0"))
    lag1 = flow0[:-1] + "1"
    tr, va, te = chrono_split(len(y))
    rows.append(dict(
        grid=name, rows=len(y), n_features=X.shape[1],
        nan_cells=int(X.isna().sum().sum()), inf_cells=int(np.isinf(X.to_numpy()).sum()),
        duplicate_rows=int(X.assign(_y=y).duplicated().sum()),
        normal=int((y == 0).sum()), attack=int((y == 1).sum()), attack_share=y.mean(),
        attack_share_train=y[tr].mean(), attack_share_val=y[va].mean(), attack_share_test=y[te].mean(),
        timestamp_monotonic=bool(ts.is_monotonic_increasing), timestamp_unique=bool(ts.is_unique),
        ts_start=ts.iloc[0], ts_end=ts.iloc[-1],
        lag_copy_check=bool(np.array_equal(X[lag1].to_numpy()[1:], X[flow0].to_numpy()[:-1])),
    ))
    del X
audit = pd.DataFrame(rows)
audit.to_csv(OUT / "audit.csv", index=False)

pairs = []
for a, b in combinations(GRIDS, 2):
    s = transfer_features(a, b)
    pairs.append(dict(pair=f"{a} <-> {b}", n_shared=len(s),
                      shared_bases=" ".join(sorted({c.rsplit("_t-", 1)[0] for c in s}))))
all3 = set.intersection(*(set(feature_columns(g)) for g in GRIDS))
pairs.append(dict(pair="all three", n_shared=len(all3),
                  shared_bases=" ".join(sorted({c.rsplit("_t-", 1)[0] for c in all3}))))
shared = pd.DataFrame(pairs)
shared.to_csv(OUT / "shared_features.csv", index=False)

pd.set_option("display.width", 200, "display.max_columns", 30)
print(audit.T.to_string())
print(shared.to_string(index=False))
for _, r in audit.iterrows():
    for s in ("train", "val", "test"):
        if abs(r[f"attack_share_{s}"] - r.attack_share) > 0.05:
            print(f"FLAG: {r.grid} {s} attack share {r[f'attack_share_{s}']:.3f} vs overall {r.attack_share:.3f}")

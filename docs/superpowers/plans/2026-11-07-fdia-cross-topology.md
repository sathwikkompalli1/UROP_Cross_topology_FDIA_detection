# Cross-Topology FDIA Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reproducible RF/XGB/ET comparison per grid, six zero-shot transfers (name-matched + topology-agnostic ablation), and a threshold study with report.

**Architecture:** One shared module `fdia_common.py` (load, split, features, metrics, thresholds) used by four numbered scripts run in order. Each script reads prior outputs from `results/` and writes CSV/JSON/NPZ/PNG. No target labels touch training, tuning or deployable thresholds.

**Tech Stack:** Python 3.13, pandas 3.0, numpy, scikit-learn 1.9, xgboost 3.4, matplotlib 3.11, pytest 9.1.

**Spec:** `docs/superpowers/specs/2026-11-07-fdia-cross-topology-design.md`

## Global Constraints
- Seed 42 everywhere (`SEED = 42` in `fdia_common.py`).
- Data files: `measurements_{5,9,14}bus_final.csv` in repo root. Never written to. Gitignored.
- Label: `y = (label_multi > 0)`; features = all columns except `Timestamp`, `label_multi`; dtype float32.
- Split: chronological 70/15/15 by row order (file is sorted by Timestamp).
- Tuning: 4 configs/model, fit on 150k-row seeded sample of train, scored on val; refit best on full train.
- Selection: val balanced accuracy @0.5; tie (<0.001) → lower FPR → lower train time.
- Thresholds: 0.5; p95 of source-val normal scores; argmax source-val BA s.t. source-val FPR ≤ 0.05; sweep 0.1..0.9 step 0.1 (target sweep = oracle/diagnostic only).
- Ablation features: per lag k∈{0..3}, over Flow cols and Inj cols: mean, std, min, max, mean|x| → 40 cols.
- Prior notebook results labelled "prior, not rerun"; notebook untouched.
- `n_jobs=-1`; RAM 15.5 GB — keep at most two grids' frames in memory.
- **Checkpoint:** stop after Task 3, report winners, wait for user approval.

## Review Focus
1. Shared-feature order differs between source and target → `transfer_features` returns a sorted list, symmetric in (src, tgt); test pins it.
2. Threshold rule (c) when no candidate meets FPR ≤ 0.05 → fall back to the lowest-FPR candidate; test pins it.
3. Single-class slice → `metrics` must not crash; ROC/PR-AUC = NaN; test pins it.
4. Probability exactly equal to threshold → Attack (`p >= t`); test pins it.
5. Ablation std → `ddof=0` so no NaN; test asserts no NaN and 40 cols.

---

### Task 1: Shared module + tests + setup files

**Files:**
- Create: `fdia_common.py`, `tests/test_common.py`, `requirements.txt`

**Interfaces (produced):**
- `SEED = 42`; `GRIDS = {"5-bus": "measurements_5bus_final.csv", "9-bus": "measurements_9bus_final.csv", "14-bus": "measurements_14bus_final.csv"}`; `PAIRS` = the six (source, target) tuples in plan.md order.
- `load_grid(name: str) -> tuple[pd.DataFrame, np.ndarray]` — X float32, y int8.
- `feature_columns(name: str) -> list[str]` — header read only.
- `chrono_split(n: int) -> tuple[slice, slice, slice]` — 70/15/15.
- `transfer_features(src: str, tgt: str) -> list[str]` — sorted intersection.
- `agnostic_features(X: pd.DataFrame) -> pd.DataFrame` — 40 cols named `{Flow|Inj}_{mean|std|min|max|absmean}_t-{k}`.
- `metrics(y, p, t: float) -> dict` — keys: accuracy, precision, recall, f1, specificity, fpr, balanced_accuracy, roc_auc, pr_auc, TN, FP, FN, TP.
- `thr_p95(p_val, y_val) -> float`; `thr_best_ba(p_val, y_val, max_fpr=0.05) -> float` (candidates 0.01..0.99 step 0.01).
- `make_model(algo: str, params: dict)` — algo ∈ {"rf","et","xgb"}; RF/ET `class_weight="balanced"`, XGB `tree_method="hist"`; all seeded, `n_jobs=-1`.
- `SEARCH: dict[str, list[dict]]` — 4 configs each:
  - rf, et: `n_estimators=200` × `max_depth∈{20, None}` × `min_samples_leaf∈{1, 5}`
  - xgb: `n_estimators=400, learning_rate=0.1` × `max_depth∈{6, 10}` × `subsample∈{0.8, 1.0}`

- [ ] **Step 1: Write tests** in `tests/test_common.py`:

```python
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
    y = np.array([0, 1]); p = np.array([0.999, 0.999])
    assert 0 < thr_best_ba(p, y, 0.05) <= 0.99
```

- [ ] **Step 2:** `pytest tests -q` → FAIL (import error).
- [ ] **Step 3:** Implement the interfaces in `fdia_common.py`.
- [ ] **Step 4:** `pytest tests -q` → 7 passed.
- [ ] **Step 5:** `requirements.txt` pinned to installed versions. Commit `feat: shared fdia module with tests`.

### Task 2: Data audit (`01_audit.py`)

**Files:** Create `01_audit.py` → `results/audit/audit.csv`, `results/audit/shared_features.csv`

- [ ] **Step 1:** Per grid: rows, n_features, NaN cells, inf cells, exact duplicate rows, normal/attack counts, attack share per split, timestamp monotonic, lag-copy check on first Flow column. Pair table: pair, n_shared, shared base names; plus all-three count.
- [ ] **Step 2:** `python 01_audit.py`. Expected: 500000 rows; 36/68/132 features; shared 20/28/44; all-3 = 20; flag any split whose attack share differs > 5 points from overall.
- [ ] **Step 3:** Commit `feat: data audit`.

### Task 3: Within-topology comparison (`02_within.py`) — CHECKPOINT

**Files:** Create `02_within.py` → `results/within/tuning.csv`, `results/within/test.csv`, `results/within/winners.json`

**Produces:** `winners.json` = `{"5-bus": {"algo": "xgb", "params": {...}, "val_ba": 0.0}, ...}` — read by Task 4.

- [ ] **Step 1:** Per grid (optional CLI arg limits to one grid): split; sample 150k train rows with `rng(SEED)`; for each algo × config: fit, time, val metrics @0.5 → `tuning.csv`. Best config per algo by val BA → refit on full train → val + test metrics + train_sec → `test.csv`. Winner among the three by val BA with tie rules → `winners.json`.
- [ ] **Step 2:** Run `python 02_within.py 5-bus`, report runtime to user; then `python 02_within.py`.
- [ ] **Step 3:** Commit `feat: within-topology comparison`.
- [ ] **Step 4: STOP.** Show user 3×3 table (val BA, val FPR, val recall, test BA, train time) + winners. Wait for approval.

### Task 4: Transfers + ablation (`03_transfer.py`)

**Files:** Create `03_transfer.py` → `results/transfer/transfer_metrics.csv`, `results/transfer/scores_{mode}_{src}_{tgt}.npz` (gitignored)

**Produces:** CSV columns: mode ∈ {main, ablation}, source, target, n_features, algo, threshold_method ∈ {fixed_0.5, p95_normal, best_ba_fpr5}, threshold, + all `metrics` keys. NPZ keys: `p_val, y_val, p_tgt, y_tgt`.

- [ ] **Step 1:** Per mode × pair: features = `transfer_features` (main) or `agnostic_features` output (ablation); `make_model(**winner)` fit on source train only; score source val + full target; three thresholds from source val only; metrics on target. `assert list(Xs.columns) == list(Xt.columns)` before predict.
- [ ] **Step 2:** Run; compare main-mode direction of results with prior CSV and note differences (no forcing).
- [ ] **Step 3:** Commit `feat: six transfers with ablation`.

### Task 5: Threshold study + figures (`04_thresholds.py`)

**Files:** Create `04_thresholds.py` → `results/thresholds/sweep_source_val.csv`, `results/thresholds/sweep_target_ORACLE.csv`, `figures/*.png`

- [ ] **Step 1:** Per mode × pair from NPZ: sweep 0.1..0.9 on source val and target (oracle); 4 figures: threshold-vs-metrics (recall, FPR, specificity, precision, F1, BA; target panel titled "ORACLE – diagnostic only"), target score histogram (normal vs attack), ROC + PR, confusion matrices at the three deployable thresholds.
- [ ] **Step 2:** Run; expect 2 × 6 × 4 = 48 PNGs.
- [ ] **Step 3:** Commit `feat: threshold study and figures`.

### Task 6: Report + README

**Files:** Create `REPORT.md`, `README.md`

- [ ] **Step 1:** README: install, data placement, run order 01→04, `pytest tests`, outputs map, link to spec + plan.
- [ ] **Step 2:** REPORT: methodology, within-grid table, transfer table (main vs ablation vs prior-not-rerun), threshold findings, asymmetry + shift (AUC + histograms to separate calibration failure from poor discrimination), limitations, future work. Only numbers present in `results/`.
- [ ] **Step 3:** Commit `docs: report and README`.

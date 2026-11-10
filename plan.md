# Claude Code (Opus) — Cross-Topology FDIA Detection Research

You are assisting with a reproducible research implementation of **Cross-Topology False Data Injection Attack Detection in Smart Grids**. Work with the existing repository and dataset; inspect them before modifying code. Do not invent results or claim improvements without experiments.

## Context

The existing baseline uses a Random Forest classifier on the FDIA-SyntheticMicroGrid dataset with 5-bus, 9-bus, and 14-bus topologies. The previous study performed directed cross-topology testing using shared measurement columns and compared a fixed 0.5 probability threshold against a source-calibrated threshold (95th percentile of source-validation normal scores). Some transfers had severe false positives: for 14→9, source-calibrated recall was 99.93% but false-positive rate was 99.70%. Reported fixed-threshold balanced accuracies included 14→5: 88.01%, 5→9: 53.51%, and 14→9: 50.73%. Treat these as prior reported baselines, not automatically reproducible findings.

The cleaned datasets were described as approximately 500,000 rows each, with 38, 70, and 134 columns respectively. Verify actual schemas and label distributions from the files before coding.

## Research goals

1. Compare **at least three classifiers** independently for each source topology: Random Forest (existing baseline), XGBoost, and Extra Trees. Recommend alternatives only with justification.
2. Select the best model **separately for each source topology** using source-only validation evidence.
3. Evaluate all **six directed transfers** without fitting or tuning on target labels:

| Source (train) | Target (test) |
|---|---|
| 5-bus | 9-bus |
| 5-bus | 14-bus |
| 9-bus | 5-bus |
| 9-bus | 14-bus |
| 14-bus | 5-bus |
| 14-bus | 9-bus |

4. Evaluate how probability thresholds influence attack recall and false alarms, and whether source-only calibration improves generalization.
5. Explain transfer asymmetry, feature-distribution shift, and limitations honestly.

## Critical experimental design

### A. Audit the repository first

- Locate data files, notebooks, scripts, requirements, baseline results, and current preprocessing.
- Inspect labels, data types, missing values, duplicates, timestamps, feature naming, class balance, and potential leakage.
- Identify the normal label and map it to 0; merge attack categories into 1.
- Exclude target-derived columns, identifiers, and any leakage-prone metadata from features.
- Report shared-feature intersections for each topology pair. Prior report: 5↔9 = 20, 5↔14 = 28, 9↔14 = 44; verify from actual datasets.
- Present a short implementation plan and identify ambiguities before significant changes.

### B. Fair within-topology comparison

For each of the 5-, 9-, and 14-bus datasets:

- Establish train, validation, and untouched test partitions; use stratification where appropriate, and assess whether chronological or grouped splitting is preferable to avoid temporal leakage.
- Fit Random Forest, XGBoost, and Extra Trees using consistent partitions and reproducible seeds.
- Use source **validation** data for hyperparameter tuning, model selection, and threshold calibration. Use the held-out source **test** partition only for final within-topology reporting; do not select the winner by repeatedly checking test scores.
- Report accuracy, precision, recall, F1, specificity, false-positive rate, balanced accuracy, confusion matrix, ROC-AUC and PR-AUC where meaningful. Include training time if feasible.
- Choose an explicit primary selection metric (prefer balanced accuracy with false-positive rate and recall as guardrails), justify it, and document ties.

### C. Shared-feature issue — avoid incompatible models

A model fitted on all 5-bus features cannot directly receive only the 5↔9 shared features. For each **directed source→target transfer**, derive the feature intersection first, then fit a **new instance of the source-selected algorithm** on source training records restricted to that intersection. Apply that fitted model to target records with exactly the same ordered features. This means six pair-specific fitted models, even though each source topology has one selected algorithm. Do not silently reuse an incompatible full-feature fitted model.

Optionally run a separate common-to-all-features ablation for apples-to-apples comparisons, clearly labeled as such.

### D. Cross-topology evaluation

- Train and calibrate exclusively with source training/validation data.
- Never retrain on target labels or use target labels for feature engineering, hyperparameter tuning, or selection of a deployable threshold.
- Evaluate all six directions, including each transfer's shared-feature count and the same metrics as above.
- Record fixed-threshold (0.5) baseline before trying calibrated thresholds.
- Preserve model scores/probabilities to support post-hoc analysis and plots.

### E. Threshold research

For attack probability `p`, classify Attack when `p >= threshold`. **Increasing** the threshold generally reduces attack predictions (often lowering both recall and false-positive rate); decreasing it generally increases both. Verify observed behavior rather than assuming monotonic metric improvements.

Evaluate:

1. Fixed threshold `0.5`.
2. Existing source-normal 95th-percentile score threshold.
3. Source-validation-selected threshold using a prespecified objective, such as maximizing balanced accuracy subject to an acceptable source-validation FPR; clearly document objective and constraints.
4. Sensitivity sweep over thresholds `0.1, 0.2, ..., 0.9`, optionally finer around relevant regions.

For each transfer, plot threshold versus recall, FPR, specificity, precision, F1 and balanced accuracy. Include confusion matrices at selected thresholds. Keep two analyses separate:

- **Valid zero-shot/source-only results:** thresholds chosen using source validation labels only.
- **Oracle/diagnostic target sensitivity:** compute target metrics over thresholds to understand potential performance; do **not** choose a threshold from target test labels and present it as a deployable zero-shot result.

A threshold cannot fix a model that does not separate normal and attack score distributions; analyze score histograms/distributions and ROC/PR behavior to distinguish calibration failure from poor discrimination.

## Deliverables

1. Clean, modular, runnable Python code (scripts or refactored notebooks), dependency instructions, and reproducible seeds/configuration.
2. Data audit and shared-feature matrix.
3. Within-topology comparison table: three algorithms × three source topologies, with winner and selection rationale.
4. Six directed transfer results with fixed and source-calibrated thresholds.
5. Threshold-sweep tables, score-distribution plots, ROC/PR curves, and confusion matrices.
6. A concise research-style methodology, results, discussion, limitations, and future-work summary.
7. A README explaining execution order, inputs/outputs, and how to reproduce tables and figures.

## Implementation constraints

- Preserve working baseline code; make incremental, reviewable changes.
- Prefer Python, pandas, scikit-learn, XGBoost, matplotlib/seaborn as appropriate to the repository.
- Be mindful of 500k-row datasets: control memory, runtime, and parallelism; use sensible initial hyperparameters and a documented tuning budget.
- Save experiment configuration, model artifacts, metrics CSVs, and figures in organized directories.
- Check for class imbalance and report metrics beyond raw accuracy.
- Do not fabricate missing measurements, feature names, dataset paths, or performance numbers.
- Clearly distinguish verified baseline findings from new experimental results.

## Workflow instruction

**Start by inspecting the existing project files and dataset schemas.** Summarize what exists, flag any design problems, and propose a staged execution plan. Then implement and validate stage by stage: data audit → within-topology model comparison → six shared-feature transfers → threshold analysis → results and report. Ask for missing files only when needed. Do not skip directly to writing a large unverified pipeline.

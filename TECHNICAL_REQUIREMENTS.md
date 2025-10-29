# TECHNICAL_DOCUMENTATION.md

project: Poverty-Graduation Eligibility & Aid Recommendation
dataset: Participant_Selection_Final.csv (+ dataset description)
last updated: 2025-10-29 (Asia/Dhaka)

---

## 1. Problem statement

We automate two linked decisions for a poverty-graduation program:

1. Task 1 — Eligibility scoring
   Predict a calibrated probability that a household qualifies for IGA support.

2. Task 2 — Aid recommendation (run only for eligible households)
   Recommend the most suitable aid type among four actionable categories:
   Cash Grant, Livelihood Asset, Health Support, Training.

We use an ensemble-first approach with human-in-the-loop review, strict leakage control, and transparent explanations.

---

## 2. Data and targets

2.1 Source and shape
Single tabular dataset (rows = households). Columns include demographics, health, income, loans, savings, assets. Targets come from existing program labels and engineered rules.

2.2 Canonical columns used
Participant_Selected_For_AID (Yes/No)
Aid_Type_Recomended (Enterprise Development, Both, Skill Development)
Has_Chronic_Dissease, Chronic_Patient_Num, Disabled_Yes, Disabled_Num
Total_Income_Annualy, Income_Monthly_per_head
Loans_Taken_Yes, Loans_Running_Yes, Loans_Num, Loans_Outstanding
has_Savings, Savings_Amt
Asset_Yes, Productive_Asset_Num, Productive_Asset_Value
Family_Size, has_under5, has_50_plus, has_18_50_Family_member
Migrant, Family_Migrant_Tenure(Years)
Gurdian_Age, Marrital_Status

2.3 Target definitions

Task 1 (binary, probabilistic)
y₁ = 1 if Participant_Selected_For_AID == "Yes"; else 0
Model outputs calibrated P(eligible) ∈ [0,1] and a decision at a program-chosen threshold.

Task 2 (multi-label → optionally single-label)
We construct four heads with weak labels consistent with the data dictionary.

• Direct mappings from Aid_Type_Recomended
Enterprise Development → Livelihood Asset = 1
Skill Development → Training = 1
Both → Livelihood Asset = 1 and Training = 1

• Weak-label rules for new heads
Health Support = 1 if any of:
Has_Chronic_Dissease == "Yes"
or Chronic_Patient_Num > 0
or Disabled_Yes == "Yes"
or Disabled_Num > 0

Cash Grant = 1 if acute liquidity stress (trigger any two of):
Income_Monthly_per_head ≤ p10(Income_Monthly_per_head)
debt_to_income ≥ p80, where debt_to_income = Loans_Outstanding / (Total_Income_Annualy + 1)
has_Savings == "No" and Savings_Amt == 0
Asset_Yes == "No" and Productive_Asset_Value == 0

Percentiles (p10, p80) are computed on the training split and stored in config for auditability.

Optional single-label collapse when MIS needs one class, using priority:
Health Support > Cash Grant > Livelihood Asset > Training

---

## 3. Feature engineering

3.1 Cleaning and typing
strip whitespace; normalize Yes/No to {1,0}; coerce numerics; validate ranges.
create missingness indicators for key fields (e.g., Savings_Amt_is_null).
winsorize heavy-tailed money columns at p99; consider log1p for skew.

3.2 Engineered features
debt_to_income = Loans_Outstanding / (Total_Income_Annualy + 1)
savings_to_income = Savings_Amt / (Total_Income_Annualy + 1)
asset_value_pc = Productive_Asset_Value / (Family_Size + 1)
dependency_ratio = (has_under5 + has_50_plus) / (has_18_50_Family_member + 1)
health_burden = Chronic_Patient_Num + Disabled_Num + 1[Has_Chronic_Dissease] + 1[Disabled_Yes]
tenure_bins from Family_Migrant_Tenure(Years) (0, 1–2, 3–5, >5)

3.3 Leakage control
Do not use post-intervention columns to predict eligibility.
Do not use Aid_Type_Recomended for Task 1.
Task 2 trained only on eligible records with non-null mapping-derived labels; weak labels use pre-aid attributes only.

---

## 4. Modeling approach

4.1 Overview
Hierarchical pipeline:

1. Task 1 — predict P(eligible).
2. If eligible, Task 2 — predict per-type probabilities for four aid heads.

4.2 Ensembles (stacking with out-of-fold meta-learning)
Level-0 base learners for tabular data:
Logistic Regression (L1), Random Forest, LightGBM, XGBoost, CatBoost.
We generate 5-fold out-of-fold (OOF) predictions per base model.
Level-1 meta-learner: Logistic or LightGBM trained on concatenated OOF predictions.
Final probabilities are calibrated (isotonic or Platt).

4.3 Losses and class imbalance
Task 1: class_weight='balanced' for linear models; scale_pos_weight for GBDTs; threshold tuning to program capacity; primary metric AUC-PR.
Task 2: class weights per head (inverse prevalence); focal loss for minority heads (if supported); stratified CV by “any positive label”.

4.4 Probability calibration
Fit CalibratedClassifierCV (isotonic preferred) using internal validation folds; validate with Brier score and reliability curves.

4.5 Thresholding and capacity alignment
Select operational thresholds via grid search maximizing Fβ or recall@capacity.
Provide gains/lift charts and TP/FP/FN trade-off tables.

---

## 5. Validation and metrics

5.1 Data splits
Stratified 5-fold cross-validation.
One locked hold-out set for final reporting.

5.2 Task 1 metrics
Primary: AUC-PR
Secondary: ROC-AUC, F1, Recall@capacity, Precision@capacity, Brier score
Calibration: reliability curves; expected calibration error (ECE)

5.3 Task 2 metrics
Primary: macro-F1 across four heads (or across single-label if collapsed)
Per-class precision/recall; PR curves per head
Top-2 accuracy for recommendation UX
Confusion matrices for single-label view

5.4 Fairness and robustness
Slice metrics by available attributes (age bands, migrant, family size bins, region if present).
Report disparities and confidence intervals.
Stability check across folds (variance of scores, SHAP stability).

---

## 6. Explainability

Global importance: SHAP summary on the strongest tree learner per task; rank features by mean |SHAP| averaged across folds.
Local explanations: top 3–5 reason codes per prediction (e.g., very_low_income, no_savings, chronic_patient, high_debt_to_income).
Monotonic constraints (optional) in eligibility model to align with policy (e.g., higher income should not increase eligibility).

---

## 7. Outputs and interfaces

7.1 Batch CSVs
eligibility_scored.csv
household_id, p_eligible, decision, threshold, reason_codes[], model_version, scored_at

aid_recommendations.csv
household_id, p_cash_grant, p_livelihood_asset, p_health_support, p_training, top1, top2, reason_codes[], model_version, scored_at

7.2 API (optional FastAPI)
POST /score → {household_id, p_eligible, decision, reasons, version}
POST /recommend → {household_id, probabilities{4}, top1, top2, reasons, version}

7.3 Reporting artifacts
Fold-wise metrics; calibration plots; gains/lift charts; per-class PR curves; SHAP plots; fairness slices; confusion matrices; threshold memo.

---

## 8. Governance and ethics

Human-in-the-loop: staff may override recommendations with a reason; overrides logged.
Transparency: publish model card (data cut, features, limitations, calibration date).
Do no harm: never fully automate denial; eligibility decisions should be reviewed when near threshold or when sensitive attributes are present.
Drift monitoring: data, prediction, and calibration drift alerts; retraining or recalibration policy.
Auditability: pin random seeds, versions, config, and percentiles; keep an immutable scoring log.

---

## 9. Project structure

```
poverty-graduation-ml/
├─ README.md
├─ TECHNICAL_DOCUMENTATION.md
├─ pyproject.toml               # or requirements.txt + setup.cfg
├─ data/
│  ├─ raw/                      # original CSV(s)
│  ├─ interim/                  # cleaned splits
│  └─ processed/                # feature matrices
├─ config/
│  ├─ schema.yaml               # dtypes, valid ranges
│  ├─ thresholds.yaml           # p10/p20/p80 anchors, class thresholds, priority order
│  └─ params.yaml               # model/search params
├─ notebooks/
│  ├─ 01_eda.ipynb
│  ├─ 02_features.ipynb
│  └─ 03_models.ipynb
├─ src/
│  ├─ config/settings.py
│  ├─ data/load.py
│  ├─ data/preprocess.py
│  ├─ features/build_features.py
│  ├─ labels/build_task2_labels.py
│  ├─ models/train_eligibility.py
│  ├─ models/train_aid_multilabel.py
│  ├─ models/stacking.py
│  ├─ models/calibrate.py
│  ├─ eval/metrics.py
│  ├─ eval/reporting.py
│  ├─ explain/shap_utils.py
│  ├─ serve/api.py
│  └─ jobs/batch_score.py
├─ experiments/
│  ├─ elig_lgbm.yaml
│  └─ aid_multilabel_lgbm.yaml
├─ artifacts/
│  ├─ models/                  # serialized models, encoders
│  └─ reports/                 # HTML/JSON metrics
└─ mlflow/                     # optional local tracking
```

---

## 10. Training and evaluation procedure

10.1 Preprocessing
load → schema validate → clean types → impute (median/most-freq) + missing flags → winsorize/log1p → encode binaries (0/1) and low-card categorical (one-hot) → build engineered features.

10.2 Task 1 (eligibility)
train 5-fold base models with class weighting → generate OOF preds → fit meta-learner → calibrate → choose operating threshold via capacity-aware search → evaluate (AUC-PR, Recall@capacity, Brier) → export artifacts and reports.

10.3 Task 2 (aid, multi-label)
derive labels (direct mappings + weak rules) → train 5-fold base learners per head with class weights/focal → stack OOF probs across models and heads → calibrate per head → evaluate (macro-F1, per-head PR, top-2) → export.
optional single-label collapse using priority rule for MIS.

10.4 Feature selection
compute SHAP per fold for strongest tree → retain features consistently in top-k across ≥3/5 folds; confirm with permutation importance on hold-out; lock compact feature set.

---

## 11. Configuration knobs (policy-tunable)

thresholds.yaml
percentiles: income_pc_p10, income_pc_p20, dti_p80
cash_grant_rules: require_at_least_k_triggers (default: 2)
priority_order: [health_support, cash_grant, livelihood_asset, training]
capacity_targets: by district/union or global

params.yaml
cv_folds: 5
class_weights: auto or explicit per head
learning_rates, max_depth, num_leaves, min_child_samples
calibration: isotonic | platt
metrics_primary: AUC_PR (task1), macro_F1 (task2)

---

## 12. Acceptance criteria

Eligibility model achieves target AUC-PR and Recall@capacity on hold-out.
Aid model achieves target macro-F1 and acceptable top-2 accuracy on hold-out.
Calibration meets Brier/ECE thresholds.
Fairness report shows no material disparities or includes mitigation plan.
Reproducible run from clean clone with pinned config produces the same metrics.
Model card, threshold memo, and runbooks completed and signed off.
One pilot batch scored and reviewed with staff; feedback incorporated.

---

## 13. Runbooks

Train
make train-eligibility
make train-aid

Evaluate
make eval-eligibility
make eval-aid

Batch score
python -m src.jobs.batch_score --input data/raw/Participant_Selection_Final.csv --date 2025-10-29 --out artifacts/scored/

Serve (optional)
uvicorn src.serve.api:app --host 0.0.0.0 --port 8000

---

## 14. Risks and mitigations

Imbalance in minority aid heads → class weights, focal loss, threshold tuning, top-2 UX.
Weak-label noise → SME calibration of rules; future replacement with outcome-based labels.
Drift across cohorts/districts → drift alerting and scheduled recalibration.
Over-reliance on a few features → SHAP stability filter; permutation checks; monotonic constraints.
Ethical concerns → human review for edge cases; transparent reasons; appeal path.

---

## 15. Future improvements

Outcome-supervised aid recommender using post-aid impacts (income lift, health expense reduction).
Causal uplift modeling for aid selection.
Geospatial and seasonality features if available.
Active learning loop from staff overrides.
Lightweight mobile UI for field validation and reason review.

---

## 16. Data dictionary (operational subset)

Binary flags stored as 1/0: Has_Chronic_Dissease, Disabled_Yes, has_Savings, Asset_Yes, Migrant, has_under5, has_50_plus, has_18_50_Family_member
Counts: Chronic_Patient_Num, Disabled_Num, Loans_Num, Productive_Asset_Num, Family_Size
Amounts: Total_Income_Annualy, Income_Monthly_per_head, Loans_Outstanding, Savings_Amt, Productive_Asset_Value
Demographics: Gurdian_Age, Marrital_Status, Family_Migrant_Tenure(Years)

All ranges and categorical domains enforced by config/schema.yaml.

---

## 17. Notes on reproducibility

Fix random seeds across libraries.
Log all hashes of data splits and config; save learned percentiles in thresholds.yaml.
Serialize models and preprocessors together; include feature manifest with order and dtypes.
Track runs via MLflow (params, metrics, artifacts, code version).

---

This document specifies the final modeling plan, data transformations, decision logic, and engineering deliverables required to implement, validate, and operate the eligibility and aid-recommendation system at scale with explainability and governance.

import pandas as pd
from joblib import dump
import yaml

from src.data.preprocess import load, preprocess, drop_non_numeric_noise
from src.features.build_features import engineer_features
from src.models.stacking import fit_base_models, fit_meta
from src.eval.metrics import auc_pr, auc_roc, brier, find_best_threshold

PATH = "data/raw/Participant_Selection_Final.csv"

# load + prep
df = load(PATH)
df = preprocess(df)
df = engineer_features(df)

y = (df["Participant_Selected_For_AID"] == "Yes").astype(int)
X = drop_non_numeric_noise(df).select_dtypes(include=["number","bool"]).drop(columns=["Participant_Selected_For_AID"], errors="ignore")

# stacking
with open("config/params.yaml","r") as f:
    cfg = yaml.safe_load(f)
oof, base_models = fit_base_models(X, y, cfg["eligibility"]["base_models"], seed=cfg["seed"], n_splits=cfg["cv_folds"])
meta = fit_meta(oof, y)

# train final meta on full base predictions
# (Refit base models on full data for inference)
final_base = {}
for k, ms in base_models.items():
    # take the last trained as a template and refit on full
    m = ms[-1]
    m.random_state = cfg["seed"]
    m.fit(X, y)
    final_base[k] = m

# build stacked features for full X
import numpy as np
stack_full = np.column_stack([
    final_base["lgbm"].predict_proba(X)[:,1],
    final_base["xgb"].predict_proba(X)[:,1],
    final_base["rf"].predict_proba(X)[:,1],
    final_base["logreg"].predict_proba(X)[:,1],
])
meta.fit(stack_full, y)

# evaluate OOF
print("AUC-PR:", auc_pr(y, oof.mean(axis=1)))
print("AUC-ROC:", auc_roc(y, oof.mean(axis=1)))
thr, val = find_best_threshold(y, oof.mean(axis=1), target="f1")
print("Best F1 threshold:", thr, "score:", val)
print("Brier:", brier(y, oof.mean(axis=1)))

# persist artifacts
dump(list(X.columns), "artifacts/models/eligibility_features.joblib")
dump(final_base, "artifacts/models/eligibility_base_models.joblib")
dump(meta, "artifacts/models/eligibility_meta.joblib")

# save threshold to yaml
with open("config/thresholds.yaml","r") as f:
    th = yaml.safe_load(f)
th["eligibility_threshold"] = float(thr)
with open("config/thresholds.yaml","w") as f:
    yaml.safe_dump(th, f, sort_keys=False)

print("✅ Saved: eligibility models, features, and threshold.")

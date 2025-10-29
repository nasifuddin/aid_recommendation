import pandas as pd
from joblib import dump
from lightgbm import LGBMClassifier
import yaml

from src.data.preprocess import load, preprocess, drop_non_numeric_noise
from src.features.build_features import engineer_features
from src.labels.build_task2_labels import compute_percentiles, build_multilabel, write_thresholds_yaml, collapse_priority

PATH = "data/raw/Participant_Selection_Final.csv"

# load + prep
df = load(PATH)
df = preprocess(df)
df = engineer_features(df)

# percentiles → thresholds.yaml
thr = compute_percentiles(df)
write_thresholds_yaml("config/thresholds.yaml", thr)

# build weak multi-labels
import yaml as _y
with open("config/thresholds.yaml","r") as f:
    T = _y.safe_load(f)
df = build_multilabel(df, thresholds=T, k_triggers=T.get("cash_grant_k_triggers",2))

# collapse to single label using priority
order = T.get("priority_order", ["health_support","cash_grant","livelihood_asset","training"])
df["Aid_Target"] = df.apply(lambda r: collapse_priority(r, order), axis=1)

# train set (optionally restrict to eligible==Yes records if desired)
X = drop_non_numeric_noise(df).select_dtypes(include=["number","bool"]).drop(columns=["Participant_Selected_For_AID"], errors="ignore")
y = df["Aid_Target"]

with open("config/params.yaml","r") as f:
    cfg = yaml.safe_load(f)

aid = LGBMClassifier(objective="multiclass", random_state=cfg["seed"], **cfg["aid"]["lgbm"])
aid.fit(X, y)

dump(aid, "artifacts/models/aid_recommendation.joblib")
dump(list(X.columns), "artifacts/models/aid_features.joblib")
print("✅ Saved: aid_recommendation model + features.")

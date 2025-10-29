import pandas as pd
from joblib import dump
from src.data.preprocess import preprocess
from src.features.build_features import engineer_features
from src.labels.build_task2_labels import build_labels
from lightgbm import LGBMClassifier
import numpy as np

# Load and prep
df = preprocess(pd.read_csv("data/raw/Participant_Selection_Final.csv"))
df = engineer_features(df)

# Percentile thresholds for financial stress
p10 = df["Income_Monthly_per_head"].quantile(0.10)
p80 = (df["Loans_Outstanding"] / (df["Total_Income_Annualy"]+1)).quantile(0.80)

df = build_labels(df, p10, p20=None, p80=p80)

# Simplify to single-label based on priority
def collapse(row):
    if row["label_health_support"]: return "Health Support"
    elif row["label_cash_grant"]: return "Cash Grant"
    elif row["label_livelihood_asset"]: return "Livelihood Asset"
    elif row["label_training"]: return "Training"
    return "Training"

df["Aid_Target"] = df.apply(collapse, axis=1)

y = df["Aid_Target"]
X = df.select_dtypes(include=["number", "bool"]).drop(columns=["Aid_Target"], errors="ignore")

# Train
model = LGBMClassifier(objective="multiclass", num_class=4, random_state=42)
model.fit(X, y)

dump(model, "artifacts/models/aid_recommendation.joblib")
dump(list(X.columns), "artifacts/models/aid_features.joblib")
print("✅ Aid recommendation model saved.")

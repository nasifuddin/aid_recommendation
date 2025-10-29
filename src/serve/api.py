from fastapi import FastAPI
import pandas as pd
from joblib import load
import yaml

from src.data.preprocess import preprocess
from src.features.build_features import engineer_features

app = FastAPI(title="Poverty Graduation ML API")

# Load artifacts
elig_features = load("artifacts/models/eligibility_features.joblib")
elig_base = load("artifacts/models/eligibility_base_models.joblib")  # dict of models
elig_meta = load("artifacts/models/eligibility_meta.joblib")
aid_model = load("artifacts/models/aid_recommendation.joblib")
aid_features = load("artifacts/models/aid_features.joblib")

with open("config/thresholds.yaml","r") as f:
    TH = yaml.safe_load(f)
ELIG_THR = TH.get("eligibility_threshold", 0.5)

EXPECTED_COLS = [
    "Income_Monthly_per_head","Total_Income_Annualy","Loans_Outstanding",
    "Savings_Amt","Productive_Asset_Value","Has_Chronic_Dissease","Disabled_Yes",
    "has_Savings","Asset_Yes","Loans_Taken_Yes","Loans_Running_Yes","Migrant",
    "Family_Size","Chronic_Patient_Num","Disabled_Num","Productive_Asset_Num",
    "has_under5","has_50_plus","has_18_50_Family_member","Gurdian_Age"
]

def align(df, cols):
    for c in cols:
        if c not in df.columns:
            df[c] = 0
    return df[cols]

def predict_elig_proba(dfX):
    # stacked features from base models
    import numpy as np
    stack = np.column_stack([
        elig_base["lgbm"].predict_proba(dfX)[:,1],
        elig_base["xgb"].predict_proba(dfX)[:,1],
        elig_base["rf"].predict_proba(dfX)[:,1],
        elig_base["logreg"].predict_proba(dfX)[:,1],
    ])
    return elig_meta.predict_proba(stack)[:,1]

@app.post("/score")
def score(payload: dict):
    for col in EXPECTED_COLS:
        payload.setdefault(col, 0)
    df = pd.DataFrame([payload])
    df = engineer_features(preprocess(df))
    df = align(df, elig_features)
    p = float(predict_elig_proba(df)[0])
    return {"p_eligible": p, "threshold": ELIG_THR, "decision": int(p >= ELIG_THR)}

@app.post("/recommend")
def recommend(payload: dict):
    for col in EXPECTED_COLS:
        payload.setdefault(col, 0)
    df = pd.DataFrame([payload])
    df = engineer_features(preprocess(df))
    df = align(df, aid_features)
    probs = aid_model.predict_proba(df)[0]
    classes = aid_model.classes_
    order = probs.argsort()[::-1]
    return {
        "eligible_threshold_used": ELIG_THR,
        "Aid_Probabilities": {str(classes[i]): float(probs[i]) for i in range(len(classes))},
        "Top_1": str(classes[order[0]]),
        "Top_2": str(classes[order[1]]) if len(classes) > 1 else str(classes[order[0]])
    }

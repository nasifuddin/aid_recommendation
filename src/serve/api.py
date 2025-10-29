from fastapi import FastAPI
import pandas as pd
from joblib import load
from src.data.preprocess import preprocess
from src.features.build_features import engineer_features

app = FastAPI(title="Poverty Graduation ML API")

# Load models + feature lists
eligibility_model = load("artifacts/models/eligibility.joblib")
eligibility_features = load("artifacts/models/eligibility_features.joblib")
aid_model = load("artifacts/models/aid_recommendation.joblib")
aid_features = load("artifacts/models/aid_features.joblib")

EXPECTED_COLS = [
    "Income_Monthly_per_head","Total_Income_Annualy","Loans_Outstanding",
    "Savings_Amt","Productive_Asset_Value","Has_Chronic_Dissease","Disabled_Yes",
    "has_Savings","Asset_Yes","Loans_Taken_Yes","Loans_Running_Yes","Migrant",
    "Family_Size","Chronic_Patient_Num","Disabled_Num","Productive_Asset_Num",
    "has_under5","has_50_plus","has_18_50_Family_member","Gurdian_Age"
]

# ---------- TASK 1 ----------
@app.post("/score")
def score(payload: dict):
    for col in EXPECTED_COLS:
        payload.setdefault(col, 0)

    df = pd.DataFrame([payload])
    df = engineer_features(preprocess(df))

    # Align with training columns
    for col in eligibility_features:
        if col not in df.columns:
            df[col] = 0
    df = df[eligibility_features]

    p = eligibility_model.predict_proba(df)[:, 1][0]
    return {"p_eligible": float(p)}

# ---------- TASK 2 ----------
@app.post("/recommend")
def recommend(payload: dict):
    for col in EXPECTED_COLS:
        payload.setdefault(col, 0)

    df = pd.DataFrame([payload])
    df = engineer_features(preprocess(df))

    # Align to training feature list
    for col in aid_features:
        if col not in df.columns:
            df[col] = 0
    df = df[aid_features]

    preds = aid_model.predict_proba(df)[0]
    classes = aid_model.classes_

    # Get top recommendations
    top1 = classes[preds.argmax()]
    top2 = classes[preds.argsort()[-2]] if len(classes) > 1 else top1

    return {
        "Aid_Probabilities": dict(zip(classes, map(float, preds))),
        "Top_1": top1,
        "Top_2": top2
    }

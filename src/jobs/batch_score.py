import pandas as pd
from joblib import load
from src.data.preprocess import preprocess
from src.features.build_features import engineer_features

def batch_score(input_path, model_path, output_path):
    df = preprocess(pd.read_csv(input_path))
    df = engineer_features(df)
    model = load(model_path)
    df["p_eligible"] = model.predict_proba(df)[:,1]
    df.to_csv(output_path, index=False)

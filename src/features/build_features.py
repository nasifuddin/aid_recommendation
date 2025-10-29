import numpy as np
import pandas as pd

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    eps = 1.0
    df["debt_to_income"] = df["Loans_Outstanding"] / (df["Total_Income_Annualy"] + eps)
    df["savings_to_income"] = df["Savings_Amt"] / (df["Total_Income_Annualy"] + eps)
    df["asset_value_pc"] = df["Productive_Asset_Value"] / (df["Family_Size"] + 1.0)
    df["dependency_ratio"] = (df["has_under5"] + df["has_50_plus"]) / (df["has_18_50_Family_member"] + 1.0)
    df["health_burden"] = (
        df["Chronic_Patient_Num"].fillna(0)
        + df["Disabled_Num"].fillna(0)
        + df["Has_Chronic_Dissease"]
        + df["Disabled_Yes"]
    )
    # winsorize heavy tails (99th)
    for col in ["Total_Income_Annualy","Income_Monthly_per_head","Loans_Outstanding",
                "Savings_Amt","Productive_Asset_Value"]:
        q99 = df[col].quantile(0.99)
        df[col] = np.clip(df[col], None, q99)
    return df

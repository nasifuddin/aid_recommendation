import pandas as pd
import yaml

def compute_percentiles(df: pd.DataFrame) -> dict:
    p10 = float(df["Income_Monthly_per_head"].quantile(0.10))
    dti = df["Loans_Outstanding"] / (df["Total_Income_Annualy"] + 1.0)
    p80_dti = float(dti.replace([float("inf"), float("-inf")], None).quantile(0.80))
    return {"income_pc_p10": p10, "dti_p80": p80_dti}

def write_thresholds_yaml(path: str, update: dict):
    with open(path, "r") as f:
        base = yaml.safe_load(f)
    base.update(update)
    with open(path, "w") as f:
        yaml.safe_dump(base, f, sort_keys=False)

def build_multilabel(df: pd.DataFrame, thresholds: dict, k_triggers: int = 2) -> pd.DataFrame:
    # direct mappings
    df["label_livelihood_asset"] = ((df.get("Aid_Type_Recomended") == "Enterprise Development")
                                    | (df.get("Aid_Type_Recomended") == "Both")).astype(int)
    df["label_training"] = ((df.get("Aid_Type_Recomended") == "Skill Development")
                            | (df.get("Aid_Type_Recomended") == "Both")).astype(int)
    # health
    df["label_health_support"] = (
        (df["Has_Chronic_Dissease"] == 1) |
        (df["Chronic_Patient_Num"] > 0) |
        (df["Disabled_Yes"] == 1) |
        (df["Disabled_Num"] > 0)
    ).astype(int)
    # cash grant
    triggers = 0
    triggers += (df["Income_Monthly_per_head"] <= thresholds["income_pc_p10"]).astype(int)
    dti = df["Loans_Outstanding"] / (df["Total_Income_Annualy"] + 1.0)
    triggers += (dti >= thresholds["dti_p80"]).astype(int)
    triggers += ((df["has_Savings"] == 0) & (df["Savings_Amt"] == 0)).astype(int)
    triggers += ((df["Asset_Yes"] == 0) & (df["Productive_Asset_Value"] == 0)).astype(int)
    df["label_cash_grant"] = (triggers >= k_triggers).astype(int)
    return df

def collapse_priority(row, order):
    for k in order:
        if k == "health_support" and row["label_health_support"] == 1: return "Health Support"
        if k == "cash_grant" and row["label_cash_grant"] == 1: return "Cash Grant"
        if k == "livelihood_asset" and row["label_livelihood_asset"] == 1: return "Livelihood Asset"
        if k == "training" and row["label_training"] == 1: return "Training"
    return "Training"

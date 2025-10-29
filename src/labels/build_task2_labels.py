import numpy as np

def build_labels(df, p10, p20, p80, k_triggers=2):
    # base heads
    df["label_livelihood_asset"] = (df["Aid_Type_Recomended"]=="Enterprise Development") | (df["Aid_Type_Recomended"]=="Both")
    df["label_training"] = (df["Aid_Type_Recomended"]=="Skill Development") | (df["Aid_Type_Recomended"]=="Both")
    df["label_health_support"] = (
        (df["Has_Chronic_Dissease"]==1) |
        (df["Chronic_Patient_Num"]>0) |
        (df["Disabled_Yes"]==1) |
        (df["Disabled_Num"]>0)
    )
    conds = [
        df["Income_Monthly_per_head"] <= p10,
        (df["Loans_Outstanding"] / (df["Total_Income_Annualy"] + 1)) >= p80,
        (df["has_Savings"]==0) & (df["Savings_Amt"]==0),
        (df["Asset_Yes"]==0) & (df["Productive_Asset_Value"]==0),
    ]
    df["label_cash_grant"] = (sum(conds) >= k_triggers)
    return df

def engineer_features(df):
    df["debt_to_income"] = df["Loans_Outstanding"] / (df["Total_Income_Annualy"] + 1)
    df["savings_to_income"] = df["Savings_Amt"] / (df["Total_Income_Annualy"] + 1)
    df["asset_value_pc"] = df["Productive_Asset_Value"] / (df["Family_Size"] + 1)
    df["dependency_ratio"] = (df["has_under5"] + df["has_50_plus"]) / (df["has_18_50_Family_member"] + 1)
    df["health_burden"] = (
        df["Chronic_Patient_Num"].fillna(0)
        + df["Disabled_Num"].fillna(0)
        + df["Has_Chronic_Dissease"]
        + df["Disabled_Yes"]
    )
    return df

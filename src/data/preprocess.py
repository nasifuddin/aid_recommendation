import pandas as pd
import numpy as np


def load_dataset(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def to01(series):
    return series.map({"Yes": 1, "No": 0}).fillna(0).astype(int)


def preprocess(df):
    df["has_Savings"] = to01(df["has_Savings"])
    df["Asset_Yes"] = to01(df["Asset_Yes"])
    df["Has_Chronic_Dissease"] = to01(df["Has_Chronic_Dissease"])
    df["Disabled_Yes"] = to01(df["Disabled_Yes"])
    df["Loans_Taken_Yes"] = to01(df["Loans_Taken_Yes"])
    df["Loans_Running_Yes"] = to01(df["Loans_Running_Yes"])
    df["Migrant"] = to01(df["Migrant"])
    for c in [
        "Total_Income_Annualy",
        "Income_Monthly_per_head",
        "Loans_Outstanding",
        "Savings_Amt",
        "Productive_Asset_Value",
    ]:
        df[c] = df[c].astype(float)
    return df

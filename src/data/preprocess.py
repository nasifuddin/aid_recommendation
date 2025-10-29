import pandas as pd

YN = {"Yes": 1, "No": 0}

BIN_COLS = [
    "has_Savings","Asset_Yes","Has_Chronic_Dissease","Disabled_Yes",
    "Loans_Taken_Yes","Loans_Running_Yes","Migrant",
    "has_under5","has_50_plus","has_18_50_Family_member"
]

NUM_COLS = [
    "Total_Income_Annualy","Income_Monthly_per_head","Loans_Outstanding",
    "Savings_Amt","Productive_Asset_Value","Chronic_Patient_Num",
    "Disabled_Num","Loans_Num","Productive_Asset_Num","Family_Size","Gurdian_Age"
]

DROP_COLS = ["Participant_ID","Participant_Birthdate","Marrital_Status",
             "Aid_Type_Recomended"]  # drop target-ish from features for Task 1

def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    for c in BIN_COLS:
        if c in df.columns:
            df[c] = df[c].map(YN).fillna(0).astype(int)
        else:
            df[c] = 0
    for c in NUM_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        else:
            df[c] = 0.0
    return df

def drop_non_numeric_noise(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")

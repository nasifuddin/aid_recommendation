import pandas as pd
from joblib import dump
from src.data.preprocess import preprocess
from src.features.build_features import engineer_features
from src.models.train_eligibility import train_eligibility

# Load your dataset
df = preprocess(pd.read_csv("data/raw/Participant_Selection_Final.csv"))
df = engineer_features(df)

# Define target and features
# Define target and features
y = (df["Participant_Selected_For_AID"] == "Yes").astype(int)

# Drop obvious non-numeric / ID / date fields
drop_cols = [
    "Participant_Selected_For_AID",
    "Aid_Type_Recomended",
    "Participant_ID",
    "Participant_Birthdate",
    "Marrital_Status",
]
X = df.drop(columns=[c for c in drop_cols if c in df.columns])

# Keep only numeric dtypes
X = X.select_dtypes(include=["number", "bool"])

# Train
model = train_eligibility(X, y)

# Save model
dump(model, "artifacts/models/eligibility.joblib")

# Save feature columns
dump(list(X.columns), "artifacts/models/eligibility_features.joblib")
print("✅ Model and feature list saved.")

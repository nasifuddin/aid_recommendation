from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
import numpy as np

def train_eligibility(X, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    preds, scores = [], []
    for tr, va in skf.split(X, y):
        m = LGBMClassifier(class_weight="balanced", random_state=42)
        m.fit(X.iloc[tr], y.iloc[tr])
        p = m.predict_proba(X.iloc[va])[:,1]
        preds.extend(p)
        scores.append(average_precision_score(y.iloc[va], p))
    print("Mean AUC-PR:", np.mean(scores))
    m.fit(X, y)
    return m

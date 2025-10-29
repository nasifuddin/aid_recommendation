import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

def fit_base_models(X, y, params, seed=42, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros((len(X), 4))  # lgbm, xgb, rf, logreg
    models = {"lgbm": [], "xgb": [], "rf": [], "logreg": []}

    for tr, va in skf.split(X, y):
        Xtr, Xva, ytr, yva = X.iloc[tr], X.iloc[va], y.iloc[tr], y.iloc[va]

        lgbm = LGBMClassifier(class_weight="balanced", random_state=seed, **params["lgbm"])
        lgbm.fit(Xtr, ytr)
        oof[va,0] = lgbm.predict_proba(Xva)[:,1]
        models["lgbm"].append(lgbm)

        xgb = XGBClassifier(eval_metric="logloss", random_state=seed, **params["xgb"])
        scale_pos_weight = (ytr==0).sum() / max((ytr==1).sum(), 1)
        xgb.set_params(scale_pos_weight=scale_pos_weight)
        xgb.fit(Xtr, ytr)
        oof[va,1] = xgb.predict_proba(Xva)[:,1]
        models["xgb"].append(xgb)

        rf = RandomForestClassifier(class_weight="balanced", random_state=seed, **params["rf"])
        rf.fit(Xtr, ytr)
        oof[va,2] = rf.predict_proba(Xva)[:,1]
        models["rf"].append(rf)

        lr = LogisticRegression(penalty="l1", solver="liblinear", class_weight="balanced",
                                C=params["logreg_l1"]["C"], random_state=seed)
        lr.fit(Xtr, ytr)
        oof[va,3] = lr.predict_proba(Xva)[:,1]
        models["logreg"].append(lr)

    return oof, models

def fit_meta(oof, y):
    meta = LogisticRegression(max_iter=1000)
    meta.fit(oof, y)
    return meta

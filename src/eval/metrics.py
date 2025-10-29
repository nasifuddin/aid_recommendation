import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, brier_score_loss

def auc_pr(y_true, y_prob): return average_precision_score(y_true, y_prob)
def auc_roc(y_true, y_prob): return roc_auc_score(y_true, y_prob)
def f1_at(y_true, y_prob, thr):
    y_pred = (y_prob >= thr).astype(int)
    return f1_score(y_true, y_pred)

def find_best_threshold(y_true, y_prob, grid=None, target="f1"):
    if grid is None: grid = np.linspace(0.1, 0.9, 17)
    best_thr, best_val = 0.5, -1
    for t in grid:
        if target == "f1":
            val = f1_at(y_true, y_prob, t)
        else:
            val = f1_at(y_true, y_prob, t)
        if val > best_val:
            best_val, best_thr = val, t
    return float(best_thr), float(best_val)

def brier(y_true, y_prob): return brier_score_loss(y_true, y_prob)

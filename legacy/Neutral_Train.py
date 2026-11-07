"""
02_train_neutral.py

Trains LogisticRegression for NEUTRAL vs OTHER using features from 01_extract_features_neutral.py

Saves:
- models/neutral_lr.joblib
- models/neutral_threshold.txt
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve

FEAT_DIR = "fer_features_neutral"
MODEL_DIR = "models"

# Neutral is often subtle; 0.90 precision can be VERY strict.
# You can change this, or later override threshold in webcam with --theta.
PRECISION_TARGET = 0.70

def load(split_name: str) -> pd.DataFrame:
    path = os.path.join(FEAT_DIR, f"features_{split_name}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}. Run 01_extract_features_neutral.py first.")
    df = pd.read_csv(path)
    return df.dropna(subset=["mw_ratio", "mh_ratio", "label"]).reset_index(drop=True)

def pick_theta(y: np.ndarray, p: np.ndarray, p_target: float = 0.80) -> float:
    """
    FIXED: precision_recall_curve returns:
      prec, rec length N+1
      thr length N
    so align prec[:-1], rec[:-1] with thr.
    """
    prec, rec, thr = precision_recall_curve(y, p)
    if len(thr) == 0:
        return 0.5

    prec_t = prec[:-1]
    rec_t  = rec[:-1]

    mask = prec_t >= p_target
    if mask.any():
        best_i = int(np.argmax(rec_t[mask]))
        theta = thr[mask][best_i]
    else:
        f1s = (2 * prec_t * rec_t) / (prec_t + rec_t + 1e-12)
        theta = thr[int(np.argmax(f1s))]

    return float(theta)

def main():
    tr = load("train")
    va = load("validation")

    if tr.empty or va.empty:
        raise RuntimeError("Train/validation is empty. Check your dataset and rerun feature extraction.")

    if tr["label"].nunique() < 2 or va["label"].nunique() < 2:
        raise RuntimeError("Need both classes (neutral and other) in train+val to train properly.")

    Xtr, ytr = tr[["mw_ratio", "mh_ratio"]].to_numpy(), tr["label"].to_numpy()
    Xva, yva = va[["mw_ratio", "mh_ratio"]].to_numpy(), va["label"].to_numpy()

    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xtr, ytr)

    p_va = clf.predict_proba(Xva)[:, 1]
    theta = pick_theta(yva, p_va, p_target=PRECISION_TARGET)

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "neutral_lr.joblib")
    thr_path   = os.path.join(MODEL_DIR, "neutral_threshold.txt")

    joblib.dump(clf, model_path)
    with open(thr_path, "w") as f:
        f.write(str(theta))

    print("[OK] Saved:", model_path)
    print("[OK] Saved:", thr_path)
    print(f"[OK] Threshold θ = {theta:.4f} (precision target={PRECISION_TARGET:.2f})")

if __name__ == "__main__":
    main()

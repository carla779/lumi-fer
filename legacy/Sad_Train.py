import os
import argparse
import time
import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    precision_recall_curve,
    confusion_matrix,
    classification_report,
    average_precision_score,
)

def normalize_folder_label(name: str) -> str:
    s = (name or "").strip().lower()
    if s == "fearful":
        return "fear"
    return s

def infer_src_label_from_path(p: str) -> str:
    if not isinstance(p, str) or not p:
        return "unknown"
    try:
        return normalize_folder_label(os.path.basename(os.path.dirname(p)))
    except Exception:
        return "unknown"

def detect_feat_cols(df: pd.DataFrame):
    ignore = {"path", "label", "src_label", "orig_label", "emotion", "class", "folder"}
    feat_cols = []
    for c in df.columns:
        if c in ignore:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            feat_cols.append(c)
    return feat_cols

def pick_theta_from_pr(y, p, precision_target=0.55, select="f1", min_pred_pos=50):
    prec, rec, thr = precision_recall_curve(y, p)
    if len(thr) == 0:
        return 0.5, {"max_precision": float(np.max(prec) if len(prec) else 0.0)}

    prec_t = prec[:-1]
    rec_t  = rec[:-1]

    pred_pos = np.array([(p >= t).sum() for t in thr], dtype=np.int64)
    valid = pred_pos >= int(min_pred_pos)
    f1 = (2.0 * prec_t * rec_t) / (prec_t + rec_t + 1e-12)

    max_prec = float(prec_t[valid].max()) if valid.any() else float(prec_t.max())

    if select == "f1":
        if valid.any():
            i = int(np.argmax(f1[valid]))
            return float(thr[valid][i]), {"max_precision": max_prec}
        i = int(np.argmax(f1))
        return float(thr[i]), {"max_precision": max_prec}

    if select == "target":
        ok = valid & (prec_t >= float(precision_target))
        if ok.any():
            i = int(np.argmax(rec_t[ok]))
            return float(thr[ok][i]), {"max_precision": max_prec}
        # fallback to max precision (still with >=min_pred_pos)
        if valid.any():
            best_prec = prec_t[valid].max()
            cand = valid & (prec_t >= best_prec - 1e-12)
            i = int(np.argmax(rec_t[cand]))
            return float(thr[cand][i]), {"max_precision": max_prec}
        # last resort
        best_prec = prec_t.max()
        cand = (prec_t >= best_prec - 1e-12)
        i = int(np.argmax(rec_t[cand]))
        return float(thr[cand][i]), {"max_precision": max_prec}

    if select == "max_precision":
        if valid.any():
            best_prec = prec_t[valid].max()
            cand = valid & (prec_t >= best_prec - 1e-12)
            i = int(np.argmax(rec_t[cand]))
            return float(thr[cand][i]), {"max_precision": max_prec}
        best_prec = prec_t.max()
        cand = (prec_t >= best_prec - 1e-12)
        i = int(np.argmax(rec_t[cand]))
        return float(thr[cand][i]), {"max_precision": max_prec}

    raise ValueError("select must be one of: f1, target, max_precision")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat_dir", required=True)
    ap.add_argument("--model_out", default=os.path.join("models", "sad_lr_negfilter.joblib"))
    ap.add_argument("--theta_out", default=os.path.join("models", "sad_threshold_negfilter.txt"))

    ap.add_argument("--neg_classes", nargs="*", default=["neutral", "happy", "fear"])
    ap.add_argument("--poly_degree", type=int, default=2)
    ap.add_argument("--C_grid", nargs="*", type=float, default=[0.01, 0.1, 1.0, 10.0, 100.0])

    ap.add_argument("--select", choices=["f1", "target", "max_precision"], default="f1")
    ap.add_argument("--precision_target", type=float, default=0.55)
    ap.add_argument("--min_pred_pos", type=int, default=50)

    ap.add_argument("--neg_weight", type=float, default=1.0, help=">1 penalizes false positives more")
    args = ap.parse_args()

    tr = pd.read_csv(os.path.join(args.feat_dir, "features_train.csv"))
    va = pd.read_csv(os.path.join(args.feat_dir, "features_validation.csv"))

    if "src_label" not in tr.columns:
        tr["src_label"] = tr["path"].apply(infer_src_label_from_path) if "path" in tr.columns else "unknown"
    if "src_label" not in va.columns:
        va["src_label"] = va["path"].apply(infer_src_label_from_path) if "path" in va.columns else "unknown"

    ytr = tr["label"].astype(int).to_numpy()
    yva = va["label"].astype(int).to_numpy()

    neg_set = set(normalize_folder_label(x) for x in (args.neg_classes or []))

    tr_keep = (ytr == 1) | ((ytr == 0) & (tr["src_label"].isin(neg_set)))
    va_keep = (yva == 1) | ((yva == 0) & (va["src_label"].isin(neg_set)))
    tr = tr.loc[tr_keep].reset_index(drop=True)
    va = va.loc[va_keep].reset_index(drop=True)

    ytr = tr["label"].astype(int).to_numpy()
    yva = va["label"].astype(int).to_numpy()

    print(f"[DATA] neg_classes={sorted(list(neg_set))}")
    print(f"[DATA] train rows={len(tr)}  val rows={len(va)}")
    print(f"[DATA] train sad={int((ytr==1).sum())} other={int((ytr==0).sum())}")
    print(f"[DATA] val   sad={int((yva==1).sum())} other={int((yva==0).sum())}")

    feat_cols = detect_feat_cols(tr)
    print("[TRAIN] feat_cols:", feat_cols)

    tr = tr.dropna(subset=feat_cols + ["label"]).reset_index(drop=True)
    va = va.dropna(subset=feat_cols + ["label"]).reset_index(drop=True)
    ytr = tr["label"].astype(int).to_numpy()
    yva = va["label"].astype(int).to_numpy()

    Xtr = tr[feat_cols].to_numpy(dtype=np.float32)
    Xva = va[feat_cols].to_numpy(dtype=np.float32)

    class_weight = "balanced"
    if abs(args.neg_weight - 1.0) > 1e-9:
        class_weight = {0: float(args.neg_weight), 1: 1.0}

    steps = [("scaler", StandardScaler())]
    if args.poly_degree and args.poly_degree > 1:
        steps.append(("poly", PolynomialFeatures(degree=args.poly_degree, include_bias=False)))
    steps.append(("clf", LogisticRegression(
        max_iter=12000, solver="lbfgs", class_weight=class_weight
    )))
    pipe = Pipeline(steps)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=13)
    gs = GridSearchCV(
        pipe,
        param_grid={"clf__C": list(args.C_grid)},
        scoring="average_precision",
        cv=cv,
        n_jobs=-1
    )
    gs.fit(Xtr, ytr)
    best = gs.best_estimator_
    print("[TRAIN] best params:", gs.best_params_)
    print(f"[TRAIN] cv_best_avg_precision={gs.best_score_:.4f}")

    p_va = best.predict_proba(Xva)[:, 1]
    ap_val = average_precision_score(yva, p_va)

    theta, pr_stats = pick_theta_from_pr(
        yva, p_va,
        precision_target=args.precision_target,
        select=args.select,
        min_pred_pos=args.min_pred_pos
    )

    y_pred = (p_va >= theta).astype(int)
    tn, fp, fn, tp = confusion_matrix(yva, y_pred, labels=[0, 1]).ravel()
    prec = tp / (tp + fp + 1e-12)
    rec  = tp / (tp + fn + 1e-12)
    f1   = (2*prec*rec) / (prec + rec + 1e-12)

    print(f"[PR] val average_precision={ap_val:.4f}")
    print(f"[PR] val max_precision~{pr_stats['max_precision']:.4f} (min_pred_pos={args.min_pred_pos})")
    print(f"[VAL] theta={theta:.4f} (select={args.select}, precision_target={args.precision_target:.2f})")
    print(f"[VAL] precision={prec:.3f} recall={rec:.3f} f1={f1:.3f}")
    print("[VAL] confusion [[TN FP],[FN TP]]:")
    print(np.array([[tn, fp], [fn, tp]]))
    print("[VAL] report:")
    print(classification_report(yva, y_pred, target_names=["other(0)", "sad(1)"], digits=3))

    fp_mask = (yva == 0) & (y_pred == 1)
    if fp_mask.any():
        top = va.loc[fp_mask, "src_label"].value_counts().head(10)
        print("[VAL] Top false-positive source classes:")
        for k, v in top.items():
            print(f"   {k:>10s}: {int(v)}")

    os.makedirs(os.path.dirname(args.model_out) or ".", exist_ok=True)
    bundle = {
        "model": best,
        "feat_cols": feat_cols,
        "theta": float(theta),
        "neg_classes": sorted(list(neg_set)),
        "poly_degree": int(args.poly_degree),
        "class_weight": class_weight,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "val_average_precision": float(ap_val),
        "val_precision": float(prec),
        "val_recall": float(rec),
        "val_f1": float(f1),
    }
    joblib.dump(bundle, args.model_out)
    with open(args.theta_out, "w", encoding="utf-8") as f:
        f.write(str(theta))

    print("[OK] Saved model ->", args.model_out)
    print("[OK] Saved theta ->", args.theta_out)

if __name__ == "__main__":
    main()
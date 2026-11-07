import cv2
import time
import mediapipe as mp
import os

# --------- Config ----------
CAM_INDEX = 0
SET_RESOLUTION = (1280, 720)   # None to skip
DRAW_BOX = True
DRAW_MESH = True
RECORD = False
OUTPUT_MP4 = "mediapipe_landmarks.mp4"
OUTPUT_FPS = 30
# ---------------------------

mp_fd   = mp.solutions.face_detection
mp_mesh = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils

# Drawing specs
mesh_spec  = mp_draw.DrawingSpec(thickness=1, circle_radius=1, color=(0,255,0))
mesh_conn  = mp_mesh.FACEMESH_CONTOURS

# Create solutions
face_det = mp_fd.FaceDetection(model_selection=0, min_detection_confidence=0.5)
face_mesh = mp_mesh.FaceMesh(static_image_mode=False,
                             max_num_faces=1,
                             refine_landmarks=True,
                             min_detection_confidence=0.5,
                             min_tracking_confidence=0.5)

cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

if SET_RESOLUTION:
    w, h = SET_RESOLUTION
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

writer = None
prev = time.time(); fps = 0.0

print("Hotkeys: q/ESC quit | b toggle box | m toggle mesh | r record on/off")

while True:
    ok, frame = cap.read()
    if not ok:
        print("Frame grab failed.")
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Face detection (for box)
    if DRAW_BOX:
        det = face_det.process(rgb)
        if det.detections:
            for d in det.detections:
                bb = d.location_data.relative_bounding_box
                x1 = int(bb.xmin * w); y1 = int(bb.ymin * h)
                x2 = int((bb.xmin + bb.width)  * w)
                y2 = int((bb.ymin + bb.height) * h)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)

    # Face mesh (landmarks)
    if DRAW_MESH:
        res = face_mesh.process(rgb)
        if res.multi_face_landmarks:
            for fl in res.multi_face_landmarks:
                mp_draw.draw_landmarks(frame, fl, mesh_conn, mesh_spec, mesh_spec)

    # FPS
    now = time.time(); fps = 0.9*fps + 0.1*(1.0/(now - prev)); prev = now
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)

    # Recording (lazy create writer to match current size)
    if RECORD and writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(OUTPUT_MP4, fourcc, OUTPUT_FPS, (frame.shape[1], frame.shape[0]))
        print(f"[REC] Writing to {OUTPUT_MP4}")
    if writer is not None:
        writer.write(frame)

    cv2.putText(frame, "b:box  m:mesh  r:rec  q:quit", (10, frame.shape[0]-12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220,220,220), 1, cv2.LINE_AA)

    cv2.imshow("MediaPipe Face Box + Mesh", frame)
    k = cv2.waitKey(1) & 0xFF
    if k in (27, ord('q')):
        break
    elif k == ord('b'):
        DRAW_BOX = not DRAW_BOX
    elif k == ord('m'):
        DRAW_MESH = not DRAW_MESH
    elif k == ord('r'):
        if writer is None:
            RECORD = True
        else:
            RECORD = False
            writer.release(); writer = None
            print("[REC] Stopped.")

cap.release()
if writer is not None:
    writer.release()
cv2.destroyAllWindows()

# ======================== FER-2013 PREP (HARDENED) ==========================
# fer2013_download_and_prep.py  (hardened)
import os, zipfile, subprocess, sys, glob, csv, cv2, numpy as np, re
import mediapipe as mp

DATA_ROOT = "data/fer2013_img"
DATASET_ID_IMG = "astraszab/facial-expression-dataset-image-folders-fer2013"  # train/validation/test folders

def download_with_kagglehub(ds_id, out_dir):
    try:
        import kagglehub
        path = kagglehub.dataset_download(ds_id)
        if os.path.isfile(path) and path.endswith((".zip",".7z",".tar",".tar.gz")):
            os.makedirs(out_dir, exist_ok=True)
            with zipfile.ZipFile(path, 'r') as zf:
                zf.extractall(out_dir)
            return out_dir
        return path
    except Exception as e:
        print(f"[kagglehub] fallback to Kaggle CLI: {e}")
        return None

def download_with_cli(ds_id, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    subprocess.run(["kaggle","datasets","download","-d",ds_id,"-p",out_dir,"--unzip"], check=True)
    return out_dir

def ensure_dataset():
    path = download_with_kagglehub(DATASET_ID_IMG, DATA_ROOT)
    if path and os.path.exists(path): return path
    return download_with_cli(DATASET_ID_IMG, DATA_ROOT)

# --- MediaPipe models ---
mp_mesh_prep = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
mp_fd_prep   = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
LIPS = set(i for e in mp.solutions.face_mesh.FACEMESH_LIPS for i in e)

# Robust "happy" mapping: handles mixed names & numeric '3' (FER-2013)
HAPPY_TOKENS = {"happy", "happiness", "smile", "smiling", "joy", "joyful", "3"}
def is_happy_folder(name: str) -> bool:
    n = name.strip().lower()
    if n in HAPPY_TOKENS: return True
    if n.isdigit() and n == "3": return True
    if ("happy" in n) or ("smil" in n) or ("joy" in n): return True
    return False

def bbox_xy_from_landmarks(lm, idxs, w, h):
    xs, ys = [], []
    for i in idxs:
        xs.append(lm[i].x*w); ys.append(lm[i].y*h)
    return min(xs), min(ys), max(xs), max(ys)

def bbox_abs_from_detection(det, w, h):
    r = det.location_data.relative_bounding_box
    x0 = max(0, int(r.xmin * w)); y0 = max(0, int(r.ymin * h))
    x1 = min(w, int((r.xmin + r.width) * w)); y1 = min(h, int((r.ymin + r.height) * h))
    return x0, y0, x1, y1

def brightness_gray(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return float(g.mean())

def mouth_roi_fallback(img, face_box):
    """Lower-face heuristic when FaceMesh fails: bottom 35% of the face box."""
    fx0, fy0, fx1, fy1 = face_box
    fw, fh = max(1, fx1 - fx0), max(1, fy1 - fy0)
    y0 = fy0 + int(0.65 * fh); y1 = fy0 + int(1.00 * fh)
    x0, x1 = fx0, fx1
    y0 = max(0, min(img.shape[0]-1, y0)); y1 = max(0, min(img.shape[0]-1, y1))
    roi = img[y0:y1, x0:x1]
    if roi.size == 0:
        return 0.0, 0.0
    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(g, 50, 150)
    ys, xs = np.where(edges > 0)
    if len(xs) < 5 or len(ys) < 5:
        mw_ratio = 0.35; mh_ratio = 0.18
    else:
        mw = (xs.max() - xs.min()) / fw
        mh = (ys.max() - ys.min()) / fh
        mw_ratio = float(np.clip(mw, 0.0, 1.0))
        mh_ratio = float(np.clip(mh, 0.0, 1.0))
    return mw_ratio, mh_ratio

def process_split(split_dir, out_csv):
    rows = []
    classes = sorted([d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))])
    print(f"[DEBUG] Classes in {split_dir}:", classes)

    for cls in classes:
        cls_dir = os.path.join(split_dir, cls)
        label = 1 if is_happy_folder(cls) else 0
        for p in glob.glob(os.path.join(cls_dir,"**","*.jpg"), recursive=True) + \
                 glob.glob(os.path.join(cls_dir,"**","*.png"), recursive=True):
            img = cv2.imread(p)
            if img is None: continue
            # Upscale aggressively to help detectors on 48x48 inputs
            h, w = img.shape[:2]
            scale = max(1, int(640 / max(h, w)))
            if scale > 1:
                img = cv2.resize(img, (w*scale, h*scale), interpolation=cv2.INTER_CUBIC)
                h, w = img.shape[:2]

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Try FaceMesh first
            res_mesh = mp_mesh_prep.process(rgb)
            if res_mesh.multi_face_landmarks:
                lm = res_mesh.multi_face_landmarks[0].landmark
                fx0, fy0, fx1, fy1 = bbox_xy_from_landmarks(lm, range(len(lm)), w, h)
                lx0, ly0, lx1, ly1 = bbox_xy_from_landmarks(lm, LIPS, w, h)
                mw_ratio = (lx1 - lx0) / max(1e-6, (fx1 - fx0))
                mh_ratio = (ly1 - ly0) / max(1e-6, (fy1 - fy0))
            else:
                # Fallback: FaceDetection + mouth ROI heuristic
                res_fd = mp_fd_prep.process(rgb)
                if res_fd.detections:
                    fx0, fy0, fx1, fy1 = bbox_abs_from_detection(res_fd.detections[0], w, h)
                    mw_ratio, mh_ratio = mouth_roi_fallback(img, (fx0, fy0, fx1, fy1))
                else:
                    continue

            bright = brightness_gray(img)
            rows.append([p, label, mw_ratio, mh_ratio, bright])

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["path","label","mw_ratio","mh_ratio","brightness"])
        wr.writerows(rows)

    import pandas as pd
    df = pd.DataFrame(rows, columns=["path","label","mw_ratio","mh_ratio","brightness"])
    print(f"[OK] {out_csv}: N={len(df)}  class_counts={df['label'].value_counts().to_dict()}")

def main():
    root = ensure_dataset()
    # find train/validation/test
    candidates = [root] + [os.path.join(root, d) for d in os.listdir(root)]
    split_map = {}
    for c in candidates:
        if not os.path.isdir(c): continue
        subs = set(os.listdir(c))
        if "train" in subs and ("validation" in subs or "val" in subs) and "test" in subs:
            split_map["train"] = os.path.join(c,"train")
            split_map["validation"] = os.path.join(c,"validation" if "validation" in subs else "val")
            split_map["test"] = os.path.join(c,"test")
            break
    if not split_map:
        print("Could not locate train/validation/test. Check the downloaded structure.")
        sys.exit(1)

    process_split(split_map["train"],      "features_train.csv")
    process_split(split_map["validation"], "features_validation.csv")
    process_split(split_map["test"],       "features_test.csv")
    print("\nNext: python fer2013_train_eval.py")

if __name__ == "__main__":
    main()
import pandas as pd, numpy as np, argparse, joblib, os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="data/features.csv")
ap.add_argument("--precision_target", type=float, default=0.90)
ap.add_argument("--model", default="models/happy_lr.joblib")
ap.add_argument("--theta", default="models/threshold.txt")
args = ap.parse_args()
os.makedirs("models", exist_ok=True)

df = pd.read_csv(args.csv).dropna(subset=["label","mw_ratio","mh_ratio","brightness"]).reset_index(drop=True)
df["label"] = df["label"].astype(int)

# ---- split: subject-wise if we have >1 subject; else random stratified 80/20 ----
subjects = df["subject"].unique().tolist()
if len(subjects) > 1:
    test_subj = subjects[-1]
    train = df[df.subject != test_subj].reset_index(drop=True)
    test  = df[df.subject == test_subj].reset_index(drop=True)
    # small val set from train
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=13)
    (itr, iva) = next(sss.split(train[["mw_ratio","mh_ratio"]], train["label"]))
    val = train.iloc[iva].reset_index(drop=True)
    train = train.iloc[itr].reset_index(drop=True)
else:
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=13)
    (itr, ite) = next(sss.split(df[["mw_ratio","mh_ratio"]], df["label"]))
    test  = df.iloc[ite].reset_index(drop=True)
    train = df.iloc[itr].reset_index(drop=True)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=7)
    (itr2, iva2) = next(sss2.split(train[["mw_ratio","mh_ratio"]], train["label"]))
    val = train.iloc[iva2].reset_index(drop=True)
    train = train.iloc[itr2].reset_index(drop=True)

def comp(tbl, name):
    n = len(tbl); pos = int(tbl["label"].sum())
    print(f"{name:10s} | N={n:5d} | E1={pos:5d} | Other={n-pos:5d} | %E1={(100*pos/max(1,n)):.1f}%")

print("\nTable F2 — Dataset composition (happy vs other)")
comp(train, "Train"); comp(val, "Val"); comp(test, "Test")

# ---- model ----
Xtr, ytr = train[["mw_ratio","mh_ratio"]].to_numpy(), train["label"].to_numpy()
Xva, yva = val[["mw_ratio","mh_ratio"]].to_numpy(), val["label"].to_numpy()
Xte, yte = test[["mw_ratio","mh_ratio"]].to_numpy(), test["label"].to_numpy()

clf = LogisticRegression(max_iter=1000).fit(Xtr, ytr)

# pick threshold to meet precision target on validation; fallback = best F1
probs_va = clf.predict_proba(Xva)[:,1]
prec, rec, thr = precision_recall_curve(yva, probs_va)
theta = 0.5
mask = prec >= args.precision_target
if mask.any():
    theta = thr[mask][np.argmax(rec[mask])]
elif len(thr):
    f1s = [f1_score(yva, (probs_va >= t).astype(int)) for t in thr]
    theta = thr[int(np.argmax(f1s))]

# test predictions
probs_te = clf.predict_proba(Xte)[:,1]
pred_te  = (probs_te >= theta).astype(int)

def metrics(y, p, th):
    ap = average_precision_score(y, p)
    pred = (p >= th).astype(int)
    f1 = f1_score(y, pred)
    tp = int(((pred==1)&(y==1)).sum()); fp = int(((pred==1)&(y==0)).sum()); fn = int(((pred==0)&(y==1)).sum())
    pr = tp / max(1, tp+fp); rc = tp / max(1, tp+fn)
    return ap, f1, pr, rc

ap_over, f1_over, pr_over, rc_over = metrics(yte, probs_te, theta)
print(f"\nTable F3 — Test overall at θ={theta:.3f}")
print(f"{'AP':>6s} | {'F1':>6s} | {'Prec':>6s} | {'Rec':>6s}")
print(f"{ap_over:.3f} | {f1_over:.3f} | {pr_over:.3f} | {rc_over:.3f}")

# brightness bins as simple "conditions"
q = test["brightness"].quantile([0.33, 0.66]).to_numpy()
bins = [
    ("Low light",    test["brightness"] <= q[0]),
    ("Medium light", (test["brightness"] > q[0]) & (test["brightness"] <= q[1])),
    ("High light",   test["brightness"] > q[1]),
]
print(f"\nTable F3 — Test by brightness bin at θ={theta:.3f}")
print(f"{'Condition':15s} | {'AP':>6s} | {'F1':>6s} | {'Prec':>6s} | {'Rec':>6s} | Notes")
for name, m in bins:
    sub = test[m].reset_index(drop=True)
    if sub.empty or sub["label"].nunique() < 2:
        print(f"{name:15s} |   n/a |   n/a |   n/a |   n/a | n={len(sub)} (insufficient class variety)")
        continue
    ap_b, f1_b, pr_b, rc_b = metrics(sub["label"].to_numpy(),
                                     clf.predict_proba(sub[["mw_ratio","mh_ratio"]].to_numpy())[:,1],
                                     theta)
    print(f"{name:15s} | {ap_b:.3f} | {f1_b:.3f} | {pr_b:.3f} | {rc_b:.3f} | n={len(sub)}")

# save model + threshold for runtime
joblib.dump(clf, args.model)
with open(args.theta, "w") as f: f.write(str(theta))
print(f"\nSaved model → {args.model}  |  threshold → {args.theta}")

# ======================== TRAIN / EVAL (DF-ALIGNED, FIXED) ==================
# fer2013_train_eval.py
import os, pandas as pd, numpy as np, joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score

PRECISION_TARGET = 0.90

def load(split):
    df = pd.read_csv(f"features_{split}.csv")
    # Keep arrays aligned; drop any NaNs that could desync arrays later
    return df.dropna(subset=["mw_ratio","mh_ratio","label","brightness"]).reset_index(drop=True)

def print_table_f2():
    for split in ["train","validation","test"]:
        df = load(split); n = len(df); pos = int(df["label"].sum())
        print(f"{split.capitalize():10s} | N={n:6d} | E1={pos:6d} | Other={n-pos:6d} | %E1={(100*pos/max(1,n)):.1f}%")

def ensure_two_classes(y, name):
    u = np.unique(y)
    if len(u) < 2:
        raise ValueError(f"{name} has only one class {u}.")
    return True

def merged_resplit(df_tr, df_va, test_size=0.2, seed=13):
    pool = pd.concat([df_tr, df_va], ignore_index=True)
    y = pool["label"].to_numpy()
    if np.unique(y).size < 2:
        raise ValueError("Merged train+val still has one class. Check feature extraction / class mapping.")
    X = pool[["mw_ratio","mh_ratio"]].to_numpy()
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    idx_tr, idx_va = next(sss.split(X, y))
    return pool.iloc[idx_tr].reset_index(drop=True), pool.iloc[idx_va].reset_index(drop=True)

def pick_threshold(y, prob, p_target=0.9):
    prec, rec, thr = precision_recall_curve(y, prob)
    ap = average_precision_score(y, prob)
    theta = 0.5
    mask = prec >= p_target
    if mask.any():
        theta = thr[mask][np.argmax(rec[mask])]
    elif len(thr):
        f1s = [f1_score(y, (prob >= t).astype(int)) for t in thr]
        theta = thr[int(np.argmax(f1s))]
    return float(theta), ap

def metrics_at_theta(y, prob, theta):
    pred = (prob >= theta).astype(int)
    ap = average_precision_score(y, prob)
    f1 = f1_score(y, pred)
    tp = int(((y==1) & (pred==1)).sum()); fp = int(((y==0) & (pred==1)).sum()); fn = int(((y==1) & (pred==0)).sum())
    precision = tp / max(1, tp+fp); recall = tp / max(1, tp+fn)
    return dict(AP=ap, F1=f1, Precision=precision, Recall=recall)

def print_table_f3_brightness_df(df_test: pd.DataFrame, probs: np.ndarray, theta: float, title: str):
    """Compute per-brightness-bin metrics directly from the same DF to keep shapes aligned."""
    df = df_test.reset_index(drop=True).copy()

    # Hard guard: trim to the same length if anything got out of sync
    n = min(len(df), len(probs))
    if len(df) != len(probs):
        print(f"[warn] length mismatch test_df={len(df)} probs={len(probs)} → trimming to {n}")
    df = df.iloc[:n].copy()
    df["prob"] = np.asarray(probs[:n])

    # Brightness bins (low / medium / high). Handle degenerate cases where all brightness equal.
    if df["brightness"].nunique() < 3:
        q1, q2 = df["brightness"].min(), df["brightness"].max()
    else:
        q1, q2 = np.quantile(df["brightness"].values, [0.33, 0.66])

    conds = [
        ("Low light",    df["brightness"] <= q1),
        ("Medium light", (df["brightness"] >  q1) & (df["brightness"] <= q2)),
        ("High light",   df["brightness"] >  q2),
    ]

    print(f"\nTable F3 — {title} at θ={theta:.3f}")
    print(f"{'Condition':15s} | {'AP':>6s} | {'F1':>6s} | {'Prec':>6s} | {'Rec':>6s} | Notes")
    for name, mask in conds:
        sub = df.loc[mask]
        if sub.empty or sub["label"].nunique() < 2:
            print(f"{name:15s} |   n/a |   n/a |   n/a |   n/a | n={len(sub)} (single class or empty)")
            continue
        m = metrics_at_theta(sub["label"].to_numpy(), sub["prob"].to_numpy(), theta)
        print(f"{name:15s} | {m['AP']:.3f} | {m['F1']:.3f} | {m['Precision']:.3f} | {m['Recall']:.3f} | n={len(sub)}")

def main():
    tr = load("train"); va = load("validation"); te = load("test")

    print("\nTable F2 — Dataset composition (happy vs other)")
    print_table_f2()

    # Ensure two classes; otherwise re-split pooled train+val
    try:
        ensure_two_classes(tr["label"].to_numpy(), "Train")
        ensure_two_classes(va["label"].to_numpy(), "Validation")
        tr_fit, va_fit = tr, va
    except ValueError as e:
        print(f"[warn] {e}  → re‑splitting from merged train+val...")
        tr_fit, va_fit = merged_resplit(tr, va)

    # Train LR
    Xtr = tr_fit[["mw_ratio","mh_ratio"]].to_numpy(); ytr = tr_fit["label"].to_numpy()
    Xva = va_fit[["mw_ratio","mh_ratio"]].to_numpy(); yva = va_fit["label"].to_numpy()
    clf = LogisticRegression(max_iter=1000).fit(Xtr, ytr)

    # Pick θ on validation (high precision target; fallback = best F1)
    probs_va = clf.predict_proba(Xva)[:,1]
    theta, _ = pick_threshold(yva, probs_va, p_target=PRECISION_TARGET)

    # Test metrics (overall)
    Xte = te[["mw_ratio","mh_ratio"]].to_numpy()
    yte = te["label"].to_numpy()
    probs_te = clf.predict_proba(Xte)[:,1]

    # Hard guard if any tiny desync sneaks in
    n = min(len(yte), len(probs_te))
    if len(yte) != len(probs_te):
        print(f"[warn] y_test={len(yte)} vs probs_test={len(probs_te)} → trimming to {n}")
    yte = yte[:n]; probs_te = probs_te[:n]

    overall = metrics_at_theta(yte, probs_te, theta)

    os.makedirs("models", exist_ok=True)
    joblib.dump(clf, "models/happy_lr.joblib")
    with open("models/threshold.txt","w") as f: f.write(str(theta))

    print(f"\nTable F3 — Test overall at θ={theta:.3f}")
    print(f"{'AP':>6s} | {'F1':>6s} | {'Prec':>6s} | {'Rec':>6s}")
    print(f"{overall['AP']:.3f} | {overall['F1']:.3f} | {overall['Precision']:.3f} | {overall['Recall']:.3f}")

    # Per-brightness bins from the same DF (no boolean-index mismatch)
    print_table_f3_brightness_df(te, probs_te, theta, title="Test (brightness bins)")

    print("\nSaved model → models/happy_lr.joblib; threshold → models/threshold.txt")

if __name__ == "__main__":
    main()


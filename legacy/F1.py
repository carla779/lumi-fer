import os, argparse, numpy as np, pandas as pd, cv2, joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score
import mediapipe as mp

ap = argparse.ArgumentParser()
ap.add_argument("--feat_dir", default="fer_features_local",
                help="folder with features_train/validation/test.csv")
ap.add_argument("--precision_target", type=float, default=0.90,
                help="select θ on validation to meet this precision (fallback = best F1)")
ap.add_argument("--model", default="models/happy_lr.joblib")
ap.add_argument("--theta", default="models/threshold.txt")
ap.add_argument("--min_side", type=int, default=256, help="upscale shortest side before FaceMesh")
args = ap.parse_args()

def load_split(name):
    p = os.path.join(args.feat_dir, f"features_{name}.csv")
    if not os.path.exists(p):
        raise SystemExit(f"Missing {p} — run your prep first.")
    df = pd.read_csv(p).dropna(subset=["mw_ratio","mh_ratio","label","brightness"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    return df

train = load_split("train")
val   = load_split("validation")
test  = load_split("test")

# ----- model / θ -----
def select_theta(y, prob, p_target=0.9):
    prec, rec, thr = precision_recall_curve(y, prob)
    theta = 0.5
    mask = prec >= p_target
    if mask.any():
        theta = float(thr[mask][np.argmax(rec[mask])])
    elif len(thr):
        f1s = [f1_score(y, (prob >= t).astype(int)) for t in thr]
        theta = float(thr[int(np.argmax(f1s))])
    return theta

if os.path.exists(args.model) and os.path.exists(args.theta):
    clf = joblib.load(args.model)
    with open(args.theta) as f: theta = float(f.read().strip())
else:
    Xtr, ytr = train[["mw_ratio","mh_ratio"]].to_numpy(), train["label"].to_numpy()
    Xva, yva = val[["mw_ratio","mh_ratio"]].to_numpy(), val["label"].to_numpy()
    clf = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
    theta = select_theta(yva, clf.predict_proba(Xva)[:,1], p_target=args.precision_target)

# ----- predict on test -----
test = test.copy()
test["prob"] = clf.predict_proba(test[["mw_ratio","mh_ratio"]].to_numpy())[:,1]

# ----- add condition tags by re-reading images and computing simple cues -----
mp_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
LIPS = set(i for e in mp.solutions.face_mesh.FACEMESH_LIPS for i in e)

def upscale(img, min_side):
    H, W = img.shape[:2]
    s = max(1, int(min_side / max(1, min(H, W))))
    return cv2.resize(img, (W*s, H*s), interpolation=cv2.INTER_CUBIC) if s>1 else img

def bbox_xy(lm, idxs, w, h):
    xs, ys = [], []
    for i in idxs: xs.append(lm[i].x*w); ys.append(lm[i].y*h)
    return min(xs), min(ys), max(xs), max(ys)

def mouth_edge_density(img, face_box, lips_box):
    # Canny edges inside mouth ROI / ROI area
    x0, y0, x1, y1 = map(int, lips_box)
    x0 = max(0, min(img.shape[1]-1, x0)); x1 = max(0, min(img.shape[1]-1, x1))
    y0 = max(0, min(img.shape[0]-1, y0)); y1 = max(0, min(img.shape[0]-1, y1))
    if x1 <= x0 or y1 <= y0: return 0.0
    roi = img[y0:y1, x0:x1]
    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim==3 else roi
    edges = cv2.Canny(g, 50, 150)
    return float((edges>0).sum()) / float(edges.size)

def yaw_score_from_eyes(lm, fx0, fx1, w):
    # Use outer eye corners: 33 (right eye outer) and 263 (left eye outer), normalized by face width
    try:
        x_r = lm[33].x * w
        x_l = lm[263].x * w
    except Exception:
        return 0.0
    face_w = max(1e-6, fx1 - fx0)
    # Positive if face turned to the right (asymmetry in eye positions relative to face edges)
    left_margin  = (x_l - fx0) / face_w
    right_margin = (fx1 - x_r) / face_w
    return float(left_margin - right_margin)  # magnitude ~ yaw; 0 ~ frontal

yaw_vals, occ_vals = [], []
for i, row in test.iterrows():
    p = row["path"]
    img = cv2.imread(p)
    if img is None:
        yaw_vals.append(0.0); occ_vals.append(0.0); continue
    img = upscale(img, args.min_side)
    H, W = img.shape[:2]
    res = mp_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not res.multi_face_landmarks:
        yaw_vals.append(0.0); occ_vals.append(0.0); continue
    lm = res.multi_face_landmarks[0].landmark
    fx0, fy0, fx1, fy1 = bbox_xy(lm, range(len(lm)), W, H)
    lx0, ly0, lx1, ly1 = bbox_xy(lm, LIPS, W, H)
    yaw_vals.append(abs(yaw_score_from_eyes(lm, fx0, fx1, W)))
    occ_vals.append(mouth_edge_density(img, (fx0, fy0, fx1, fy1), (lx0, ly0, lx1, ly1)))

test["yaw_score"] = np.array(yaw_vals, dtype=float)
test["occ_score"] = np.array(occ_vals, dtype=float)

# ----- derive condition masks -----
# Low light: bottom third of brightness
q33, q66 = np.quantile(test["brightness"].to_numpy(), [0.33, 0.66])
low_light = test["brightness"] <= q33

# ~30° yaw: top third of yaw_score (exclude low-light samples so bins don't overlap)
yaw_thr = np.quantile(test["yaw_score"].to_numpy(), 0.67)
yaw_30  = (~low_light) & (test["yaw_score"] >= yaw_thr)

# Partial occlusion: lowest third of mouth-edge density (exclude low-light and yaw bins first)
occ_thr = np.quantile(test["occ_score"].to_numpy(), 0.33)
occluded = (~low_light) & (~yaw_30) & (test["occ_score"] <= occ_thr)

# Normal light, frontal: everything else
normal_frontal = ~(low_light | yaw_30 | occluded)

bins = [
    ("Normal light, frontal", normal_frontal),
    ("Normal light, ~30° yaw", yaw_30),
    ("Low light, frontal",     low_light),
    ("Partial occlusion (hand)", occluded),
]

def metrics(y, p, th):
    pred = (p >= th).astype(int)
    ap = average_precision_score(y, p)
    f1 = f1_score(y, pred)
    tp = int(((y==1)&(pred==1)).sum()); fp = int(((y==0)&(pred==1)).sum()); fn = int(((y==1)&(pred==0)).sum())
    precision = tp / max(1, tp+fp); recall = tp / max(1, tp+fn)
    return ap, f1, precision, recall

# ----- overall -----
ap_all, f1_all, pr_all, rc_all = metrics(test["label"].to_numpy(), test["prob"].to_numpy(), theta)
print(f"\nTable F3 — Test overall at θ={theta:.3f}")
print(f"{'AP':>6s} | {'F1':>6s} | {'Prec':>6s} | {'Rec':>6s}")
print(f"{ap_all:.3f} | {f1_all:.3f} | {pr_all:.3f} | {rc_all:.3f}")

# ----- per condition -----
print(f"\nTable F3 — Test by condition at θ={theta:.3f}")
print(f"{'Condition':25s} | {'AP':>6s} | {'F1':>6s} | {'Prec':>6s} | {'Rec':>6s} | n")
for name, mask in bins:
    sub = test.loc[mask].reset_index(drop=True)
    if sub.empty or sub["label"].nunique() < 2:
        print(f"{name:25s} |   n/a |   n/a |   n/a |   n/a | n={len(sub)}")
        continue
    ap_b, f1_b, pr_b, rc_b = metrics(sub["label"].to_numpy(), sub["prob"].to_numpy(), theta)
    print(f"{name:25s} | {ap_b:.3f} | {f1_b:.3f} | {pr_b:.3f} | {rc_b:.3f} | n={len(sub)}")

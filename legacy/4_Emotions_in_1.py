import os
# Reduce MediaPipe/TF log noise
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import sys
import glob
import time
import argparse
from collections import deque

import cv2
import numpy as np
import pandas as pd
import joblib
import mediapipe as mp

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score


# -----------------------------
# DEFAULTS
# -----------------------------
DEFAULT_DATASET_ROOT = r"C:\Users\carla\PyCharmMiscProject\fer-2013"
DEFAULT_EMOTIONS = ["happy", "sad", "neutral", "fear"]   # "fearful" handled automatically
DEFAULT_MODEL_PATH = os.path.join("models", "emotions_4_lr.joblib")

# Upgraded feature list (Mouth + Eyes + Eyebrows + Mouth corners)
FEAT_COLS = [
    "mw_ratio",
    "mh_ratio",
    "mouth_corner_lift",
    "mouth_corner_asym",
    "eye_open_l",
    "eye_open_r",
    "eye_aspect_l",
    "eye_aspect_r",
    "brow_eye_dist_l",
    "brow_eye_dist_r",

    # --- NEW brow features ---
    "inner_brow_eye_dist_l",
    "inner_brow_eye_dist_r",
    "brow_slope_l",
    "brow_slope_r",
    "brow_inner_dist",
]


def indices_from_edges(edges):
    s = set()
    for a, b in edges:
        s.add(a); s.add(b)
    return s


# Build region index sets from MediaPipe constants
LIPS_IDX = indices_from_edges(mp.solutions.face_mesh.FACEMESH_LIPS)
LEFT_EYE_IDX = indices_from_edges(mp.solutions.face_mesh.FACEMESH_LEFT_EYE)
RIGHT_EYE_IDX = indices_from_edges(mp.solutions.face_mesh.FACEMESH_RIGHT_EYE)
LEFT_BROW_IDX = indices_from_edges(mp.solutions.face_mesh.FACEMESH_LEFT_EYEBROW)
RIGHT_BROW_IDX = indices_from_edges(mp.solutions.face_mesh.FACEMESH_RIGHT_EYEBROW)


def try_open_camera(index: int):
    backends = [cv2.CAP_ANY]
    if os.name == "nt":
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]

    for backend in backends:
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            continue
        ok, frame = cap.read()
        if ok and frame is not None and frame.size > 0:
            return cap
        cap.release()
    return None


def bbox_xy_from_landmarks(lm, idxs, w, h):
    xs, ys = [], []
    for i in idxs:
        xs.append(lm[i].x * w)
        ys.append(lm[i].y * h)
    return min(xs), min(ys), max(xs), max(ys)


def bbox_abs_from_detection(det, w, h):
    r = det.location_data.relative_bounding_box
    x0 = max(0, int(r.xmin * w))
    y0 = max(0, int(r.ymin * h))
    x1 = min(w, int((r.xmin + r.width) * w))
    y1 = min(h, int((r.ymin + r.height) * h))
    return x0, y0, x1, y1


def crop_with_margin(img, box, margin=0.20):
    x0, y0, x1, y1 = box
    h, w = img.shape[:2]
    bw = x1 - x0
    bh = y1 - y0
    mx = int(bw * margin)
    my = int(bh * margin)
    xx0 = max(0, x0 - mx)
    yy0 = max(0, y0 - my)
    xx1 = min(w, x1 + mx)
    yy1 = min(h, y1 + my)
    crop = img[yy0:yy1, xx0:xx1]
    return crop


def run_facemesh(img_bgr, mp_mesh, mp_fd=None):

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    res = mp_mesh.process(rgb)
    if res.multi_face_landmarks:
        lm = res.multi_face_landmarks[0].landmark
        h, w = img_bgr.shape[:2]
        return lm, w, h

    if mp_fd is None:
        return None, None, None

    res_fd = mp_fd.process(rgb)
    if res_fd and res_fd.detections:
        h, w = img_bgr.shape[:2]
        x0, y0, x1, y1 = bbox_abs_from_detection(res_fd.detections[0], w, h)
        crop = crop_with_margin(img_bgr, (x0, y0, x1, y1), margin=0.25)
        if crop is None or crop.size == 0:
            return None, None, None

        rgb2 = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        res2 = mp_mesh.process(rgb2)
        if res2.multi_face_landmarks:
            lm2 = res2.multi_face_landmarks[0].landmark
            hh, ww = crop.shape[:2]
            return lm2, ww, hh

    return None, None, None


def normalize_label(folder_name: str) -> str:
    s = folder_name.strip().lower()
    if s == "fearful":
        return "fear"
    return s


def mean_xy(lm, idxs, w, h):
    xs, ys = [], []
    for i in idxs:
        xs.append(lm[i].x * w)
        ys.append(lm[i].y * h)
    return float(np.mean(xs)), float(np.mean(ys))


def extract_feature_vector(lm, w, h):
    """
    Compute upgraded features from landmarks (scale-invariant ratios).

    Changes vs the previous version:
      - Mouth width/height use fixed, anatomically-stable landmark indices:
          * mouth corners: 61 and 291
          * upper/lower lip center: 13 and 14
      - Eye openness uses the standard Eye Aspect Ratio (EAR) geometry (6 landmarks / eye),
        which is much less jittery than a raw eye bounding box.

    Output keys remain the same (FEAT_COLS compatible):
      mw_ratio, mh_ratio, mouth_corner_lift, mouth_corner_asym,
      eye_open_l, eye_open_r, eye_aspect_l, eye_aspect_r,
      brow_eye_dist_l, brow_eye_dist_r
    """

    # -----------------------------
    # Helpers
    # -----------------------------
    def xy(i: int):
        return float(lm[i].x * w), float(lm[i].y * h)

    def dist(a, b) -> float:
        ax, ay = a
        bx, by = b
        return float(np.hypot(ax - bx, ay - by))

    def ear_from_indices(idxs):
        # idxs = [p1, p2, p3, p4, p5, p6]
        p1 = xy(idxs[0]); p2 = xy(idxs[1]); p3 = xy(idxs[2])
        p4 = xy(idxs[3]); p5 = xy(idxs[4]); p6 = xy(idxs[5])

        # EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
        v1 = dist(p2, p6)
        v2 = dist(p3, p5)
        h_ = max(1e-6, dist(p1, p4))
        ear = (v1 + v2) / (2.0 * h_)

        # Also return a pixel-space "vertical opening" proxy (useful for eye_open_*)
        v_open = 0.5 * (v1 + v2)
        eye_center_y = float(np.mean([p1[1], p2[1], p3[1], p4[1], p5[1], p6[1]]))
        return ear, v_open, eye_center_y

    # -----------------------------
    # Face bbox (for scale normalization)
    # -----------------------------
    all_idxs = range(len(lm))
    fx0, fy0, fx1, fy1 = bbox_xy_from_landmarks(lm, all_idxs, w, h)
    face_w = max(1e-6, (fx1 - fx0))
    face_h = max(1e-6, (fy1 - fy0))

    # -----------------------------
    # Mouth geometry (fixed indices)
    # -----------------------------
    # Stable landmarks used widely with FaceMesh:
    #   corners: 61, 291; lip centers: 13 (upper), 14 (lower)
    MOUTH_L, MOUTH_R = 61, 291
    UPPER_LIP, LOWER_LIP = 13, 14

    p_ml = xy(MOUTH_L)
    p_mr = xy(MOUTH_R)
    p_ul = xy(UPPER_LIP)
    p_ll = xy(LOWER_LIP)

    mouth_w = max(1e-6, dist(p_ml, p_mr))
    mouth_h = max(1e-6, dist(p_ul, p_ll))

    mw_ratio = mouth_w / face_w
    mh_ratio = mouth_h / face_h

    mouth_center_y = 0.5 * (p_ul[1] + p_ll[1])
    corner_y_avg = 0.5 * (p_ml[1] + p_mr[1])

    # y grows downward; corners up => smaller y => positive lift
    mouth_corner_lift = (mouth_center_y - corner_y_avg) / face_h
    mouth_corner_asym = (p_mr[1] - p_ml[1]) / face_h

    # -----------------------------
    # Eyes (EAR-based openness)
    # -----------------------------
    # Common FaceMesh EAR landmark sets (6 points each)
    RIGHT_EAR = [33, 160, 158, 133, 153, 144]
    LEFT_EAR  = [362, 385, 387, 263, 373, 380]

    ear_r, vopen_r, eye_y_r = ear_from_indices(RIGHT_EAR)
    ear_l, vopen_l, eye_y_l = ear_from_indices(LEFT_EAR)

    # eye_open_* stays a "vertical opening vs face height"
    eye_open_l = vopen_l / face_h
    eye_open_r = vopen_r / face_h

    # eye_aspect_* becomes EAR (dimensionless openness ratio)
    eye_aspect_l = ear_l
    eye_aspect_r = ear_r

    # -----------------------------
    # Eyebrow vs eye distance
    # -----------------------------
    # Keep your existing brow index sets; eye Y uses the EAR eye center (less jittery).
    _, brow_y_l = mean_xy(lm, LEFT_BROW_IDX, w, h)
    _, brow_y_r = mean_xy(lm, RIGHT_BROW_IDX, w, h)

    brow_eye_dist_l = (eye_y_l - brow_y_l) / face_h
    brow_eye_dist_r = (eye_y_r - brow_y_r) / face_h

    feats = {
        "mw_ratio": float(np.clip(mw_ratio, 0.0, 2.0)),
        "mh_ratio": float(np.clip(mh_ratio, 0.0, 2.0)),
        "mouth_corner_lift": float(np.clip(mouth_corner_lift, -1.0, 1.0)),
        "mouth_corner_asym": float(np.clip(mouth_corner_asym, -1.0, 1.0)),
        "eye_open_l": float(np.clip(eye_open_l, 0.0, 1.0)),
        "eye_open_r": float(np.clip(eye_open_r, 0.0, 1.0)),
        "eye_aspect_l": float(np.clip(eye_aspect_l, 0.0, 1.0)),
        "eye_aspect_r": float(np.clip(eye_aspect_r, 0.0, 1.0)),
        "brow_eye_dist_l": float(np.clip(brow_eye_dist_l, -1.0, 1.0)),
        "brow_eye_dist_r": float(np.clip(brow_eye_dist_r, -1.0, 1.0)),
    }
    return feats

def extract_features_from_split(split_dir: str, emotions_set: set, mp_mesh, mp_fd, verbose=False) -> pd.DataFrame:
    """
    Extract upgraded features for images in split_dir/<class>/**.
    Keeps only labels in emotions_set.
    """
    if not os.path.isdir(split_dir):
        return pd.DataFrame(columns=["path", "label"] + FEAT_COLS)

    class_folders = sorted(
        [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]
    )

    rows = []
    for cls in class_folders:
        label = normalize_label(cls)
        if label not in emotions_set:
            continue

        cls_dir = os.path.join(split_dir, cls)

        files = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            files += glob.glob(os.path.join(cls_dir, "**", ext), recursive=True)

        if verbose:
            print(f"[INFO] {split_dir} / {cls} files={len(files)}")

        for p in files:
            img = cv2.imread(p)
            if img is None:
                continue

            # Upscale to help FaceMesh on small images
            ih, iw = img.shape[:2]
            scale = max(1, int(640 / max(ih, iw)))
            if scale > 1:
                img = cv2.resize(img, (iw * scale, ih * scale), interpolation=cv2.INTER_CUBIC)

            lm, ww, hh = run_facemesh(img, mp_mesh, mp_fd=mp_fd)
            if lm is None:
                continue

            feats = extract_feature_vector(lm, ww, hh)
            rows.append([p, label] + [feats[c] for c in FEAT_COLS])

    df = pd.DataFrame(rows, columns=["path", "label"] + FEAT_COLS)
    return df


def draw_bar(frame, label, value, x, y, w, h, highlight=False):
    value = float(np.clip(value, 0.0, 1.0))
    cv2.rectangle(frame, (x, y), (x + w, y + h), (40, 40, 40), thickness=-1)
    fill_w = int(w * value)
    fill_color = (0, 255, 0) if highlight else (180, 180, 180)
    if fill_w > 0:
        cv2.rectangle(frame, (x, y), (x + fill_w, y + h), fill_color, thickness=-1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (235, 235, 235), thickness=2)
    cv2.putText(frame, f"{label}: {value:.2f}", (x, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (235, 235, 235), 2)


def train_mode(args):
    dataset_root = args.dataset_root
    train_dir = os.path.join(dataset_root, "train")
    test_dir  = os.path.join(dataset_root, "test")
    val_dir   = os.path.join(dataset_root, "validation")

    emotions = [normalize_label(e) for e in args.emotions]
    emotions_set = set(emotions)

    print("[TRAIN] emotions:", emotions)
    print("[TRAIN] dataset_root:", dataset_root)
    os.makedirs(args.cache_dir, exist_ok=True)

    train_csv = os.path.join(args.cache_dir, "features_train.csv")
    val_csv   = os.path.join(args.cache_dir, "features_validation.csv")
    test_csv  = os.path.join(args.cache_dir, "features_test.csv")

    def cache_valid(path):
        if not os.path.exists(path):
            return False
        try:
            df = pd.read_csv(path, nrows=5)
            return all(c in df.columns for c in (["label"] + FEAT_COLS))
        except Exception:
            return False

    use_cache = (not args.recompute) and cache_valid(train_csv) and cache_valid(val_csv)

    if use_cache:
        print("[TRAIN] Loading cached features from:", args.cache_dir)
        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)
        test_df = pd.read_csv(test_csv) if os.path.exists(test_csv) else pd.DataFrame()
    else:
        print("[TRAIN] Extracting upgraded features (mouth+eyes+brows)...")
        with mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1, refine_landmarks=True
        ) as mp_mesh, mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        ) as mp_fd:

            train_df = extract_features_from_split(train_dir, emotions_set, mp_mesh, mp_fd, verbose=args.verbose)
            test_df  = extract_features_from_split(test_dir, emotions_set, mp_mesh, mp_fd, verbose=args.verbose)

            if os.path.isdir(val_dir):
                val_df = extract_features_from_split(val_dir, emotions_set, mp_mesh, mp_fd, verbose=args.verbose)
            else:
                if train_df.empty:
                    raise RuntimeError("Train split produced 0 samples. Check dataset paths.")
                y = train_df["label"].to_numpy()
                if len(set(y)) < 2:
                    raise RuntimeError("Train has only one class. Need multiple emotion folders.")

                sss = StratifiedShuffleSplit(n_splits=1, test_size=args.val_split, random_state=13)
                idx_tr, idx_va = next(sss.split(train_df[FEAT_COLS].to_numpy(), y))
                val_df = train_df.iloc[idx_va].reset_index(drop=True)
                train_df = train_df.iloc[idx_tr].reset_index(drop=True)

        train_df.to_csv(train_csv, index=False)
        val_df.to_csv(val_csv, index=False)
        if not test_df.empty:
            test_df.to_csv(test_csv, index=False)
        print("[TRAIN] Saved cached features to:", args.cache_dir)

    # Sanity checks
    for name, df in [("train", train_df), ("val", val_df)]:
        if df.empty:
            raise RuntimeError(f"{name} is empty after extraction. Check your dataset folders.")
        missing = set(emotions) - set(df["label"].unique().tolist())
        if missing:
            print(f"[WARN] {name} missing classes: {sorted(missing)} (model may be weak for those)")

    Xtr = train_df[FEAT_COLS].to_numpy()
    ytr = train_df["label"].to_numpy()
    Xva = val_df[FEAT_COLS].to_numpy()
    yva = val_df["label"].to_numpy()

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=8000,
            solver="lbfgs",
            class_weight="balanced"
        ))
    ])

    param_grid = {
        "clf__C": [0.01, 0.1, 1.0, 10.0, 100.0],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=13)
    gs = GridSearchCV(pipe, param_grid=param_grid, scoring="f1_macro", cv=cv, n_jobs=-1)
    gs.fit(Xtr, ytr)

    best = gs.best_estimator_
    pred_va = best.predict(Xva)
    f1m = f1_score(yva, pred_va, average="macro")

    print("[TRAIN] best params:", gs.best_params_)
    print(f"[TRAIN] validation f1_macro={f1m:.3f}")

    os.makedirs(os.path.dirname(args.out_model) or ".", exist_ok=True)
    bundle = {
        "model": best,
        "classes": sorted(list(set(ytr) | set(yva))),
        "feat_cols": FEAT_COLS,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    joblib.dump(bundle, args.out_model)
    print("[TRAIN] Saved model ->", args.out_model)


def webcam_mode(args):
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Missing model: {args.model}. Run: python emotions_4in1.py train --recompute")

    bundle = joblib.load(args.model)
    model = bundle["model"]
    classes = bundle["classes"]
    feat_cols = bundle.get("feat_cols", FEAT_COLS)

    cap = try_open_camera(args.cam)
    if cap is None:
        raise RuntimeError("Cannot open camera. Try --cam 1 and close other apps using the camera.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    fd = None
    if args.fallback:
        fd = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

    # smoothing buffers per class prob
    hist = {c: deque(maxlen=max(1, args.smooth)) for c in classes}

    print("Webcam started. Press 'q' to quit.")
    print("Model classes:", classes)
    print(f"min_conf (OTHER cutoff): {args.min_conf:.2f}")
    print(f"smooth window: {args.smooth}")

    desired = ["happy", "sad", "neutral", "fear"]
    order = [c for c in desired if c in classes] + [c for c in sorted(classes) if c not in desired]

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        h, w = frame.shape[:2]

        lm, ww, hh = run_facemesh(frame, mesh, mp_fd=fd if args.fallback else None)
        probs_smoothed = {c: 0.0 for c in classes}
        label_text = "NO FACE"
        top_label = None
        top_prob = 0.0

        if lm is not None:
            feats = extract_feature_vector(lm, ww, hh)
            x = np.array([[feats[c] for c in feat_cols]], dtype=np.float32)

            p = model.predict_proba(x)[0]
            model_classes = model.named_steps["clf"].classes_

            for cls_name, prob in zip(model_classes, p):
                if cls_name in hist:
                    hist[cls_name].append(float(prob))

            for c in classes:
                probs_smoothed[c] = float(np.mean(hist[c])) if len(hist[c]) else 0.0

            top_label = max(probs_smoothed, key=lambda k: probs_smoothed[k])
            top_prob = probs_smoothed[top_label]

            if top_prob < args.min_conf:
                label_text = "OTHER"
            else:
                label_text = top_label.upper()

        # label colors
        if label_text == "NO FACE":
            label_color = (200, 200, 200)
        elif label_text == "OTHER":
            label_color = (0, 255, 255)
        else:
            label_color = (0, 255, 0)

        cv2.putText(frame, label_text, (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, label_color, 4)

        # Bars
        x0, y0 = 20, 120
        bar_w, bar_h, gap = 360, 18, 14

        for i, c in enumerate(order):
            yy = y0 + i * (bar_h + gap)
            highlight = (c == top_label) and (label_text not in ["NO FACE", "OTHER"])
            draw_bar(frame, c.upper(), probs_smoothed.get(c, 0.0), x0, yy, bar_w, bar_h, highlight=highlight)

        # min_conf vertical marker
        thr_x = x0 + int(bar_w * float(np.clip(args.min_conf, 0.0, 1.0)))
        cv2.line(frame, (thr_x, y0 - 8), (thr_x, y0 + len(order) * (bar_h + gap) - gap + 8),
                 (0, 255, 255), 2)
        cv2.putText(frame, f"min_conf={args.min_conf:.2f}", (thr_x - 40, y0 - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        cv2.putText(frame, "Press 'q' to quit", (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (235, 235, 235), 2)

        cv2.imshow("4-Emotion Detector (Upgraded Features)", frame)

        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    mesh.close()
    if fd is not None:
        fd.close()


def main():
    # python 4_Emotions_in_1.py  run the webcam automatically.
    if len(sys.argv) == 1:
        sys.argv.append("webcam")
    elif sys.argv[1] in ("-h", "--help"):
        pass  # keep normal help behavior
    elif sys.argv[1].startswith("-"):
        # Allows: python 4_Emotions_in_1.py --cam 1  (treat as webcam args)
        sys.argv.insert(1, "webcam")
    elif sys.argv[1] not in ("train", "webcam"):
        # Any unknown first word -> treat as webcam mode
        sys.argv.insert(1, "webcam")

    ap = argparse.ArgumentParser(
        description="Train + webcam for 4-emotion detector (UPGRADED: eyes+brows+mouth corners)"
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    # ---- train ----
    tr = sub.add_parser("train", help="Extract upgraded features + train multiclass model")
    tr.add_argument("--dataset_root", default=DEFAULT_DATASET_ROOT)
    tr.add_argument(
        "--emotions",
        nargs="+",
        default=DEFAULT_EMOTIONS,
        help="Emotion folder names to include (default: happy sad neutral fear). 'fearful' is mapped to fear.",
    )
    tr.add_argument("--out_model", default=DEFAULT_MODEL_PATH)
    tr.add_argument(
        "--cache_dir",
        default="fer_features_4emotions_cache_upgraded",
        help="Where to save/load cached features CSVs",
    )
    tr.add_argument(
        "--recompute",
        action="store_true",
        help="Force recompute features even if cached CSVs exist",
    )
    tr.add_argument(
        "--val_split",
        type=float,
        default=0.20,
        help="If no validation folder exists, create validation split from train",
    )
    tr.add_argument("--verbose", action="store_true", help="Print per-folder extraction info")

    # ---- webcam ----
    wb = sub.add_parser("webcam", help="Run webcam inference and show all 4 emotions")
    wb.add_argument("--model", default=DEFAULT_MODEL_PATH)
    wb.add_argument("--cam", type=int, default=0)
    wb.add_argument("--smooth", type=int, default=7, help="Smoothing window (frames)")
    wb.add_argument(
        "--min_conf",
        type=float,
        default=0.50,
        help="If top probability < min_conf, show OTHER",
    )
    wb.add_argument(
        "--fallback",
        action="store_true",
        help="Enable FaceDetection crop fallback if FaceMesh fails (slower, more robust)",
    )

    args = ap.parse_args()

    if args.cmd == "train":
        train_mode(args)
    elif args.cmd == "webcam":
        webcam_mode(args)
    else:
        raise RuntimeError("Unknown command")


if __name__ == "__main__":
    main()

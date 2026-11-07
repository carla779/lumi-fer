import os
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import glob
import argparse
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from sklearn.model_selection import StratifiedShuffleSplit


def indices_from_edges(edges):
    s = set()
    for a, b in edges:
        s.add(a); s.add(b)
    return s


LEFT_BROW_IDX  = indices_from_edges(mp.solutions.face_mesh.FACEMESH_LEFT_EYEBROW)
RIGHT_BROW_IDX = indices_from_edges(mp.solutions.face_mesh.FACEMESH_RIGHT_EYEBROW)


FEAT_COLS = [
    "mw_ratio", "mh_ratio",
    "mouth_corner_lift", "mouth_corner_asym",
    "eye_open_l", "eye_open_r",
    "eye_aspect_l", "eye_aspect_r",
    "brow_eye_dist_l", "brow_eye_dist_r",

    # NEW brow features
    "inner_brow_eye_dist_l", "inner_brow_eye_dist_r",
    "brow_slope_l", "brow_slope_r",
    "brow_inner_dist",
]


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


def crop_with_margin(img, box, margin=0.25):
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
    return img[yy0:yy1, xx0:xx1]


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


def mean_xy(lm, idxs, w, h):
    xs, ys = [], []
    for i in idxs:
        xs.append(lm[i].x * w)
        ys.append(lm[i].y * h)
    return float(np.mean(xs)), float(np.mean(ys))


def brow_inner_outer_points(lm, brow_idxs, w, h, face_cx):
    pts = [(float(lm[i].x * w), float(lm[i].y * h)) for i in brow_idxs]
    # inner = closest to face center in x; outer = farthest
    inner = min(pts, key=lambda p: abs(p[0] - face_cx))
    outer = max(pts, key=lambda p: abs(p[0] - face_cx))
    return inner, outer


def extract_feature_vector(lm, w, h):
    def xy(i: int):
        return float(lm[i].x * w), float(lm[i].y * h)

    def dist(a, b) -> float:
        ax, ay = a
        bx, by = b
        return float(np.hypot(ax - bx, ay - by))

    def ear_from_indices(idxs):
        p1 = xy(idxs[0]); p2 = xy(idxs[1]); p3 = xy(idxs[2])
        p4 = xy(idxs[3]); p5 = xy(idxs[4]); p6 = xy(idxs[5])
        v1 = dist(p2, p6)
        v2 = dist(p3, p5)
        h_ = max(1e-6, dist(p1, p4))
        ear = (v1 + v2) / (2.0 * h_)
        v_open = 0.5 * (v1 + v2)
        eye_center_y = float(np.mean([p1[1], p2[1], p3[1], p4[1], p5[1], p6[1]]))
        return ear, v_open, eye_center_y

    # face bbox for normalization
    fx0, fy0, fx1, fy1 = bbox_xy_from_landmarks(lm, range(len(lm)), w, h)
    face_w = max(1e-6, (fx1 - fx0))
    face_h = max(1e-6, (fy1 - fy0))
    face_cx = 0.5 * (fx0 + fx1)

    # mouth fixed points
    MOUTH_L, MOUTH_R = 61, 291
    UPPER_LIP, LOWER_LIP = 13, 14
    p_ml = xy(MOUTH_L); p_mr = xy(MOUTH_R)
    p_ul = xy(UPPER_LIP); p_ll = xy(LOWER_LIP)

    mouth_w = max(1e-6, dist(p_ml, p_mr))
    mouth_h = max(1e-6, dist(p_ul, p_ll))

    mw_ratio = mouth_w / face_w
    mh_ratio = mouth_h / face_h

    mouth_center_y = 0.5 * (p_ul[1] + p_ll[1])
    corner_y_avg = 0.5 * (p_ml[1] + p_mr[1])
    mouth_corner_lift = (mouth_center_y - corner_y_avg) / face_h
    mouth_corner_asym = (p_mr[1] - p_ml[1]) / face_h

    # eyes EAR
    RIGHT_EAR = [33, 160, 158, 133, 153, 144]
    LEFT_EAR  = [362, 385, 387, 263, 373, 380]
    ear_r, vopen_r, eye_y_r = ear_from_indices(RIGHT_EAR)
    ear_l, vopen_l, eye_y_l = ear_from_indices(LEFT_EAR)

    eye_open_l = vopen_l / face_h
    eye_open_r = vopen_r / face_h
    eye_aspect_l = ear_l
    eye_aspect_r = ear_r

    # existing brow mean distance
    _, brow_y_l = mean_xy(lm, LEFT_BROW_IDX, w, h)
    _, brow_y_r = mean_xy(lm, RIGHT_BROW_IDX, w, h)
    brow_eye_dist_l = (eye_y_l - brow_y_l) / face_h
    brow_eye_dist_r = (eye_y_r - brow_y_r) / face_h

    # NEW brow features
    inner_l, outer_l = brow_inner_outer_points(lm, LEFT_BROW_IDX, w, h, face_cx)
    inner_r, outer_r = brow_inner_outer_points(lm, RIGHT_BROW_IDX, w, h, face_cx)

    inner_brow_eye_dist_l = (eye_y_l - inner_l[1]) / face_h
    inner_brow_eye_dist_r = (eye_y_r - inner_r[1]) / face_h

    brow_slope_l = (outer_l[1] - inner_l[1]) / face_h
    brow_slope_r = (outer_r[1] - inner_r[1]) / face_h

    brow_inner_dist = float(np.hypot(inner_l[0] - inner_r[0], inner_l[1] - inner_r[1])) / face_w

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

        "inner_brow_eye_dist_l": float(np.clip(inner_brow_eye_dist_l, -1.0, 1.0)),
        "inner_brow_eye_dist_r": float(np.clip(inner_brow_eye_dist_r, -1.0, 1.0)),
        "brow_slope_l": float(np.clip(brow_slope_l, -1.0, 1.0)),
        "brow_slope_r": float(np.clip(brow_slope_r, -1.0, 1.0)),
        "brow_inner_dist": float(np.clip(brow_inner_dist, 0.0, 2.0)),
    }
    return feats


def iter_images(split_dir):
    files = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        files += glob.glob(os.path.join(split_dir, "**", ext), recursive=True)
    return files


def extract_split(split_dir, mp_mesh, mp_fd):
    if not os.path.isdir(split_dir):
        return pd.DataFrame(columns=["path", "label"] + FEAT_COLS)

    rows = []
    class_folders = sorted([d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))])

    for cls in class_folders:
        cls_dir = os.path.join(split_dir, cls)
        cls_norm = cls.strip().lower()
        label = 1 if cls_norm == "sad" else 0

        for p in iter_images(cls_dir):
            img = cv2.imread(p)
            if img is None:
                continue

            # upscale FER images for FaceMesh stability
            ih, iw = img.shape[:2]
            scale = max(1, int(640 / max(ih, iw)))
            if scale > 1:
                img = cv2.resize(img, (iw * scale, ih * scale), interpolation=cv2.INTER_CUBIC)

            lm, ww, hh = run_facemesh(img, mp_mesh, mp_fd=mp_fd)
            if lm is None:
                continue

            feats = extract_feature_vector(lm, ww, hh)
            rows.append([p, label] + [feats[c] for c in FEAT_COLS])

    return pd.DataFrame(rows, columns=["path", "label"] + FEAT_COLS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True,
                    help=r'Path like C:\Users\carla\PyCharmMiscProject\fer-2013')
    ap.add_argument("--out_dir", default="fer_features_sad_brow_v1")
    ap.add_argument("--val_split", type=float, default=0.20,
                    help="Used only if dataset_root/validation does not exist")
    args = ap.parse_args()

    train_dir = os.path.join(args.dataset_root, "train")
    test_dir  = os.path.join(args.dataset_root, "test")
    val_dir   = os.path.join(args.dataset_root, "validation")

    os.makedirs(args.out_dir, exist_ok=True)

    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1, refine_landmarks=True
    ) as mp_mesh, mp.solutions.face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.5
    ) as mp_fd:

        train_df = extract_split(train_dir, mp_mesh, mp_fd)
        test_df  = extract_split(test_dir, mp_mesh, mp_fd)

        if os.path.isdir(val_dir):
            val_df = extract_split(val_dir, mp_mesh, mp_fd)
        else:
            # create stratified split from train_df
            if train_df.empty or train_df["label"].nunique() < 2:
                raise RuntimeError("Train split produced 0 samples or only one class.")
            X = train_df[FEAT_COLS].to_numpy()
            y = train_df["label"].to_numpy()
            sss = StratifiedShuffleSplit(n_splits=1, test_size=args.val_split, random_state=13)
            idx_tr, idx_va = next(sss.split(X, y))
            val_df = train_df.iloc[idx_va].reset_index(drop=True)
            train_df = train_df.iloc[idx_tr].reset_index(drop=True)

    train_df.to_csv(os.path.join(args.out_dir, "features_train.csv"), index=False)
    val_df.to_csv(os.path.join(args.out_dir, "features_validation.csv"), index=False)
    test_df.to_csv(os.path.join(args.out_dir, "features_test.csv"), index=False)

    print("[OK] Saved feature CSVs to:", args.out_dir)
    print("[OK] train:", len(train_df), "val:", len(val_df), "test:", len(test_df))
    print("[OK] columns:", ["path", "label"] + FEAT_COLS)

if __name__ == "__main__":
    main()
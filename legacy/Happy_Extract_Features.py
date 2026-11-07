"""
01_extract_features.py

Extracts simple features for FER-2013 style folders:
FER_DATASET_ROOT/
  train/<class>/*.jpg
  test/<class>/*.jpg
  validation/<class>/*.jpg   (optional)

Outputs CSVs to OUT_DIR:
- features_train.csv
- features_validation.csv
- features_test.csv
"""

import os
# (optional) reduce mediapipe/absl logging noise
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import glob
import sys
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from sklearn.model_selection import StratifiedShuffleSplit

# --------------------------------------------------------------------
# EDIT THIS to your FER-2013 root folder:
# Example:
# FER_DATASET_ROOT = r"C:\Users\carla\PyCharmMiscProject\fer-2013"
# --------------------------------------------------------------------
FER_DATASET_ROOT = r"C:\Users\carla\PyCharmMiscProject\fer-2013"

TRAIN_DIR = os.path.join(FER_DATASET_ROOT, "train")
TEST_DIR = os.path.join(FER_DATASET_ROOT, "test")
VAL_DIR = os.path.join(FER_DATASET_ROOT, "validation")  # optional

OUT_DIR = "fer_features_local"
os.makedirs(OUT_DIR, exist_ok=True)

POS_CLASS_NAME = "happy"  # label=1; all other folders label=0

LIPS = set(i for e in mp.solutions.face_mesh.FACEMESH_LIPS for i in e)


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


def brightness_gray(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return float(g.mean())


def mouth_roi_fallback(img, face_box):
    """Lower-face heuristic when FaceMesh fails: bottom ~35% of the face box."""
    fx0, fy0, fx1, fy1 = face_box
    fw, fh = max(1, fx1 - fx0), max(1, fy1 - fy0)

    y0 = fy0 + int(0.65 * fh)
    y1 = fy0 + int(1.00 * fh)
    x0, x1 = fx0, fx1

    y0 = max(0, min(img.shape[0] - 1, y0))
    y1 = max(0, min(img.shape[0] - 1, y1))

    roi = img[y0:y1, x0:x1]
    if roi.size == 0:
        return 0.0, 0.0

    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(g, 50, 150)
    ys, xs = np.where(edges > 0)

    if len(xs) < 5 or len(ys) < 5:
        return 0.35, 0.18

    mw = (xs.max() - xs.min()) / fw
    mh = (ys.max() - ys.min()) / fh
    return float(np.clip(mw, 0.0, 1.0)), float(np.clip(mh, 0.0, 1.0))


def extract_split(split_dir, mp_mesh, mp_fd):
    """Return DataFrame with columns: path, label, mw_ratio, mh_ratio, brightness."""
    if not os.path.isdir(split_dir):
        return pd.DataFrame(columns=["path", "label", "mw_ratio", "mh_ratio", "brightness"])

    class_folders = sorted(
        [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]
    )
    rows = []

    for cls in class_folders:
        cls_dir = os.path.join(split_dir, cls)
        label = 1 if cls.lower() == POS_CLASS_NAME else 0

        files = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            files += glob.glob(os.path.join(cls_dir, "**", ext), recursive=True)

        for p in files:
            img = cv2.imread(p)
            if img is None:
                continue

            # Upscale aggressively to help landmarking on small FER images
            h, w = img.shape[:2]
            scale = max(1, int(640 / max(h, w)))
            if scale > 1:
                img = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
                h, w = img.shape[:2]

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # FaceMesh
            res = mp_mesh.process(rgb)
            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0].landmark
                fx0, fy0, fx1, fy1 = bbox_xy_from_landmarks(lm, range(len(lm)), w, h)
                lx0, ly0, lx1, ly1 = bbox_xy_from_landmarks(lm, LIPS, w, h)
                mw_ratio = (lx1 - lx0) / max(1e-6, (fx1 - fx0))
                mh_ratio = (ly1 - ly0) / max(1e-6, (fy1 - fy0))
            else:
                # Fallback: FaceDetection + mouth ROI heuristic
                res_fd = mp_fd.process(rgb)
                if res_fd and res_fd.detections:
                    fx0, fy0, fx1, fy1 = bbox_abs_from_detection(res_fd.detections[0], w, h)
                    mw_ratio, mh_ratio = mouth_roi_fallback(img, (fx0, fy0, fx1, fy1))
                else:
                    continue

            rows.append([p, label, mw_ratio, mh_ratio, brightness_gray(img)])

    return pd.DataFrame(rows, columns=["path", "label", "mw_ratio", "mh_ratio", "brightness"])


def main():
    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1, refine_landmarks=True
    ) as mp_mesh, mp.solutions.face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.5
    ) as mp_fd:

        train_df = extract_split(TRAIN_DIR, mp_mesh, mp_fd)
        test_df = extract_split(TEST_DIR, mp_mesh, mp_fd)

        if os.path.isdir(VAL_DIR):
            val_df = extract_split(VAL_DIR, mp_mesh, mp_fd)
        else:
            if len(train_df) == 0 or train_df["label"].nunique() < 2:
                print("[ERROR] Train split has insufficient data or only one class.")
                sys.exit(1)

            X = train_df[["mw_ratio", "mh_ratio"]].to_numpy()
            y = train_df["label"].to_numpy()

            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=13)
            idx_tr, idx_va = next(sss.split(X, y))
            val_df = train_df.iloc[idx_va].reset_index(drop=True)
            train_df = train_df.iloc[idx_tr].reset_index(drop=True)

        train_df.to_csv(os.path.join(OUT_DIR, "features_train.csv"), index=False)
        val_df.to_csv(os.path.join(OUT_DIR, "features_validation.csv"), index=False)
        test_df.to_csv(os.path.join(OUT_DIR, "features_test.csv"), index=False)

        print("[OK] Saved feature CSVs to:", OUT_DIR)


if __name__ == "__main__":
    main()


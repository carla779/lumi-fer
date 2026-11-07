import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import mediapipe as mp


# Mediapipe FaceMesh indices (commonly used)
L_EYE = [33, 160, 158, 133, 153, 144]     # p1,p2,p3,p4,p5,p6 for EAR
R_EYE = [263, 387, 385, 362, 380, 373]
MOUTH_L = 61
MOUTH_R = 291
LIP_UP = 13
LIP_LO = 14
L_EYE_OUT = 33
R_EYE_OUT = 263
# brows (common)
L_BROW = [70, 63, 105, 66, 107]
R_BROW = [336, 296, 334, 293, 300]
L_EYE_TOP = 159
R_EYE_TOP = 386


def dist(a, b):
    return float(np.linalg.norm(np.array(a) - np.array(b)))


def ear(pts6):
    # EAR = (||p2-p6|| + ||p3-p5||) / (2*||p1-p4||)
    p1, p2, p3, p4, p5, p6 = pts6
    denom = 2.0 * dist(p1, p4) + 1e-6
    return (dist(p2, p6) + dist(p3, p5)) / denom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--splits", nargs="*", default=["train", "val"], help="Which splits to process (default train val)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    splits = set([s.lower() for s in args.splits])

    fm = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        refine_landmarks=True,
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # Output columns
    au_cols = [
        "au_eye_ear_l", "au_eye_ear_r",
        "au_eye_open_l", "au_eye_open_r",
        "au_mouth_open", "au_mouth_width",
        "au_brow_raise_l", "au_brow_raise_r",
        "au_brow_inner_dist",
    ]

    # init with NaNs
    for c in au_cols:
        if c not in df.columns:
            df[c] = np.nan
    if "au_ok" not in df.columns:
        df["au_ok"] = 0

    n = len(df) if args.limit <= 0 else min(len(df), args.limit)

    for i in range(n):
        sp = str(df.at[i, "split"]).lower()
        if sp not in splits:
            continue

        img_path = str(df.at[i, "image"])
        bgr = cv2.imread(img_path)
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = fm.process(rgb)
        if not res.multi_face_landmarks:
            continue

        lm = res.multi_face_landmarks[0].landmark

        def xy(idx):
            return (lm[idx].x * w, lm[idx].y * h)

        # scale by inter-ocular distance (more stable)
        iod = dist(xy(L_EYE_OUT), xy(R_EYE_OUT)) + 1e-6

        # eye
        ear_l = ear([xy(j) for j in L_EYE])
        ear_r = ear([xy(j) for j in R_EYE])
        eye_open_l = dist(xy(L_EYE_TOP), xy(145)) / iod
        eye_open_r = dist(xy(R_EYE_TOP), xy(374)) / iod

        # mouth
        mouth_open = dist(xy(LIP_UP), xy(LIP_LO)) / iod
        mouth_width = dist(xy(MOUTH_L), xy(MOUTH_R)) / iod

        # brows
        brow_raise_l = dist(xy(105), xy(L_EYE_TOP)) / iod
        brow_raise_r = dist(xy(334), xy(R_EYE_TOP)) / iod
        brow_inner_dist = dist(xy(70), xy(336)) / iod

        df.at[i, "au_eye_ear_l"] = ear_l
        df.at[i, "au_eye_ear_r"] = ear_r
        df.at[i, "au_eye_open_l"] = eye_open_l
        df.at[i, "au_eye_open_r"] = eye_open_r
        df.at[i, "au_mouth_open"] = mouth_open
        df.at[i, "au_mouth_width"] = mouth_width
        df.at[i, "au_brow_raise_l"] = brow_raise_l
        df.at[i, "au_brow_raise_r"] = brow_raise_r
        df.at[i, "au_brow_inner_dist"] = brow_inner_dist
        df.at[i, "au_ok"] = 1

        if (i + 1) % 1000 == 0:
            ok_cnt = int(df["au_ok"].sum())
            print(f"[{i+1}/{n}] au_ok={ok_cnt}")

    fm.close()
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print("[OK] wrote:", args.out_csv)
    print("[OK] au_ok total:", int(df["au_ok"].sum()))


if __name__ == "__main__":
    main()
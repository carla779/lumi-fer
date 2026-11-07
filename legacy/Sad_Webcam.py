import os
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse
from collections import deque

import cv2
import numpy as np
import joblib
import mediapipe as mp


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


def indices_from_edges(edges):
    s = set()
    for a, b in edges:
        s.add(a); s.add(b)
    return s


LEFT_BROW_IDX  = indices_from_edges(mp.solutions.face_mesh.FACEMESH_LEFT_EYEBROW)
RIGHT_BROW_IDX = indices_from_edges(mp.solutions.face_mesh.FACEMESH_RIGHT_EYEBROW)


def bbox_xy_from_landmarks(lm, idxs, w, h):
    xs, ys = [], []
    for i in idxs:
        xs.append(lm[i].x * w)
        ys.append(lm[i].y * h)
    return min(xs), min(ys), max(xs), max(ys)


def mean_xy(lm, idxs, w, h):
    xs, ys = [], []
    for i in idxs:
        xs.append(lm[i].x * w)
        ys.append(lm[i].y * h)
    return float(np.mean(xs)), float(np.mean(ys))


def brow_inner_outer_points(lm, brow_idxs, w, h, face_cx):
    pts = [(float(lm[i].x * w), float(lm[i].y * h)) for i in brow_idxs]
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

    # Face bbox (normalize everything)
    fx0, fy0, fx1, fy1 = bbox_xy_from_landmarks(lm, range(len(lm)), w, h)
    face_w = max(1e-6, (fx1 - fx0))
    face_h = max(1e-6, (fy1 - fy0))
    face_cx = 0.5 * (fx0 + fx1)

    # Mouth fixed indices
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

    # Eyes EAR
    RIGHT_EAR = [33, 160, 158, 133, 153, 144]
    LEFT_EAR  = [362, 385, 387, 263, 373, 380]
    ear_r, vopen_r, eye_y_r = ear_from_indices(RIGHT_EAR)
    ear_l, vopen_l, eye_y_l = ear_from_indices(LEFT_EAR)

    eye_open_l = vopen_l / face_h
    eye_open_r = vopen_r / face_h
    eye_aspect_l = ear_l
    eye_aspect_r = ear_r

    # Brow mean distance
    _, brow_y_l = mean_xy(lm, LEFT_BROW_IDX, w, h)
    _, brow_y_r = mean_xy(lm, RIGHT_BROW_IDX, w, h)
    brow_eye_dist_l = (eye_y_l - brow_y_l) / face_h
    brow_eye_dist_r = (eye_y_r - brow_y_r) / face_h

    # Extra brow features
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--model", default=r"models\sad_lr_negfilter.joblib")
    ap.add_argument("--theta", type=float, default=None, help="Override theta")
    ap.add_argument("--smooth", type=int, default=7)
    ap.add_argument("--k_frames", type=int, default=3, help="Require K consecutive frames above theta")
    args = ap.parse_args()

    bundle = joblib.load(args.model)
    model = bundle["model"]
    feat_cols = bundle["feat_cols"]
    theta = float(bundle.get("theta", 0.5))
    if args.theta is not None:
        theta = float(args.theta)

    cap = try_open_camera(args.cam)
    if cap is None:
        raise RuntimeError("Cannot open camera. Try --cam 1 and close other apps using camera.")

    mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    hist = deque(maxlen=max(1, args.smooth))
    streak = 0

    print("Press 'q' to quit.")
    print(f"theta={theta:.3f}, smooth={args.smooth}, k_frames={args.k_frames}")

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = mesh.process(rgb)

        label = "NO FACE"
        prob_s = 0.0

        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            feats = extract_feature_vector(lm, w, h)
            x = np.array([[feats[c] for c in feat_cols]], dtype=np.float32)
            p = float(model.predict_proba(x)[0][1])

            hist.append(p)
            prob_s = float(np.mean(hist))

            # K-frame confirmation to cut down false positives
            if prob_s >= theta:
                streak += 1
            else:
                streak = max(0, streak - 1)

            label = "SAD" if streak >= args.k_frames else "NOT SAD"

        cv2.putText(frame, f"{label}  p={prob_s:.2f}  theta={theta:.2f}",
                    (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                    (0, 255, 0) if label == "SAD" else (0, 0, 255) if label == "NOT SAD" else (200, 200, 200),
                    3)

        cv2.imshow("Sad (negfilter + brow)", frame)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    mesh.close()


if __name__ == "__main__":
    main()
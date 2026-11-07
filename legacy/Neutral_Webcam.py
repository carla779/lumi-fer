"""
03_webcam_neutral_realtime.py

Real-time NEUTRAL vs NOT NEUTRAL using webcam + MediaPipe FaceMesh
Loads:
- models/neutral_lr.joblib
- models/neutral_threshold.txt

Press 'q' to quit.
"""

import os
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse
import cv2
import numpy as np
import joblib
import mediapipe as mp

DEFAULT_MODEL = "models/neutral_lr.joblib"
DEFAULT_THRESH = "models/neutral_threshold.txt"

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

def draw_confidence_bar(frame, conf, theta, x=20, y=130, w=340, h=22, fill_color=(0,255,0)):
    conf = float(np.clip(conf, 0.0, 1.0))
    theta = float(np.clip(theta, 0.0, 1.0))

    # background
    cv2.rectangle(frame, (x, y), (x+w, y+h), (40, 40, 40), thickness=-1)

    # fill
    fill_w = int(w * conf)
    if fill_w > 0:
        cv2.rectangle(frame, (x, y), (x+fill_w, y+h), fill_color, thickness=-1)

    # border
    cv2.rectangle(frame, (x, y), (x+w, y+h), (235, 235, 235), thickness=2)

    # threshold line
    thr_x = x + int(w * theta)
    cv2.line(frame, (thr_x, y-4), (thr_x, y+h+4), (0, 255, 255), thickness=2)

    cv2.putText(frame, f"confidence={conf:.2f}  theta={theta:.2f}",
                (x, y+h+22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (235,235,235), 2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0, help="Webcam index (0,1,2...)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Path to neutral model")
    ap.add_argument("--theta_file", default=DEFAULT_THRESH, help="Path to neutral threshold")
    ap.add_argument("--theta", type=float, default=None, help="Override threshold manually (e.g. 0.5)")
    ap.add_argument("--smooth", type=int, default=5, help="Smoothing window (frames)")
    args = ap.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Missing model: {args.model}. Run 02_train_neutral.py first.")
    clf = joblib.load(args.model)

    if args.theta is not None:
        theta = float(args.theta)
    else:
        if not os.path.exists(args.theta_file):
            raise FileNotFoundError(f"Missing threshold: {args.theta_file}. Run 02_train_neutral.py first.")
        with open(args.theta_file, "r") as f:
            theta = float(f.read().strip())

    mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    LIPS = set(i for e in mp.solutions.face_mesh.FACEMESH_LIPS for i in e)

    def bbox_xy(lm, idxs, w, h):
        xs, ys = [], []
        for i in idxs:
            xs.append(lm[i].x * w)
            ys.append(lm[i].y * h)
        return min(xs), min(ys), max(xs), max(ys)

    cap = try_open_camera(args.cam)
    if cap is None:
        raise RuntimeError("Cannot open camera. Try --cam 1 and close other apps using camera.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    probs_hist = []

    print("Webcam started. Press 'q' to quit.")
    print(f"Using NEUTRAL threshold θ = {theta:.3f}  (try --theta 0.5 if too strict)")

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = mesh.process(rgb)

        label_text = "NO FACE"
        smooth = None
        is_neutral = None

        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            fx0, fy0, fx1, fy1 = bbox_xy(lm, range(len(lm)), w, h)
            lx0, ly0, lx1, ly1 = bbox_xy(lm, LIPS, w, h)

            mw = (lx1 - lx0) / max(1e-6, (fx1 - fx0))
            mh = (ly1 - ly0) / max(1e-6, (fy1 - fy0))

            prob = float(clf.predict_proba(np.array([[mw, mh]]))[:, 1][0])

            probs_hist.append(prob)
            if len(probs_hist) > max(1, args.smooth):
                probs_hist.pop(0)
            smooth = float(np.mean(probs_hist))

            is_neutral = smooth >= theta
            label_text = "NEUTRAL" if is_neutral else "NOT NEUTRAL"

        # colors
        if is_neutral is None:
            color = (200, 200, 200)
            fill_color = (160, 160, 160)
        elif is_neutral:
            color = (0, 255, 0)
            fill_color = (0, 255, 0)
        else:
            color = (0, 0, 255)
            fill_color = (0, 0, 255)

        cv2.putText(frame, label_text, (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, color, 4)

        cv2.putText(frame, "Neutral confidence", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (235,235,235), 2)

        conf_for_bar = smooth if smooth is not None else 0.0
        draw_confidence_bar(frame, conf_for_bar, theta, x=20, y=130, w=340, h=22, fill_color=fill_color)

        cv2.putText(frame, "Press 'q' to quit", (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (235,235,235), 2)

        cv2.imshow("Neutral detector (real-time)", frame)

        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    mesh.close()

if __name__ == "__main__":
    main()

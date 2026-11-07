import os
# Reduce MediaPipe / TF logging noise (won't remove everything, but helps)
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse
import cv2
import numpy as np
import joblib
import mediapipe as mp


MODEL_PATH = "models/happy_lr.joblib"
THRESH_PATH = "models/threshold.txt"
CAM_INDEX = 0
SMOOTH_N = 5  # smoothing window


def try_open_camera(index: int):
    """
    On Windows, OpenCV sometimes reads frames better with CAP_DSHOW or CAP_MSMF.
    Try a few backends and return the first working capture.
    """
    backends = []
    if os.name == "nt":
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    else:
        backends = [cv2.CAP_ANY]

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


def draw_confidence_bar(frame, conf, theta, x=20, y=130, w=340, h=22, fill_color=(0, 255, 0)):
    """
    Draws a horizontal confidence bar with:
    - background
    - filled portion for conf in [0,1]
    - vertical line for theta in [0,1]
    """
    conf = float(np.clip(conf, 0.0, 1.0))
    theta = float(np.clip(theta, 0.0, 1.0))

    # Background (dark)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (40, 40, 40), thickness=-1)

    # Filled portion
    fill_w = int(w * conf)
    if fill_w > 0:
        cv2.rectangle(frame, (x, y), (x + fill_w, y + h), fill_color, thickness=-1)

    # Border
    cv2.rectangle(frame, (x, y), (x + w, y + h), (235, 235, 235), thickness=2)

    # Threshold line
    thr_x = x + int(w * theta)
    cv2.line(frame, (thr_x, y - 4), (thr_x, y + h + 4), (0, 255, 255), thickness=2)

    # Text on/near bar
    txt = f"confidence={conf:.2f}   theta={theta:.2f}"
    cv2.putText(frame, txt, (x, y + h + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (235, 235, 235), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=CAM_INDEX, help="Webcam index (try 0,1,2...)")
    ap.add_argument("--model", default=MODEL_PATH, help="Path to trained model")
    ap.add_argument("--theta_file", default=THRESH_PATH, help="Path to saved threshold")
    ap.add_argument("--theta", type=float, default=None, help="Override threshold (e.g. 0.5)")
    ap.add_argument("--smooth", type=int, default=SMOOTH_N, help="Smoothing window (frames)")
    args = ap.parse_args()

    # ---- Load model + threshold ----
    if not os.path.exists(args.model):
        raise FileNotFoundError(
            f"Missing model: {args.model}\n"
            f"Train first to create models/happy_lr.joblib"
        )
    clf = joblib.load(args.model)

    if args.theta is not None:
        theta = float(args.theta)
    else:
        if not os.path.exists(args.theta_file):
            raise FileNotFoundError(
                f"Missing threshold file: {args.theta_file}\n"
                f"Train first to create models/threshold.txt"
            )
        with open(args.theta_file, "r") as f:
            theta = float(f.read().strip())

    # ---- MediaPipe FaceMesh ----
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

    # ---- Open camera ----
    cap = try_open_camera(args.cam)
    if cap is None:
        raise RuntimeError(
            f"Could not open webcam at index {args.cam}.\n"
            f"Try: --cam 1 or --cam 2, and close Zoom/Teams/Camera app if open."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    probs_hist = []
    fail_reads = 0

    print("Webcam started. Press 'q' to quit.")
    print(f"Using threshold θ = {theta:.3f}  (Tip: try --theta 0.5 if it's too strict)")

    while True:
        ok, frame = cap.read()
        if not ok or frame is None or frame.size == 0:
            fail_reads += 1
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "Camera read failed...", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.putText(blank, "Close other camera apps, try --cam 1", (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Happy detector (real-time)", blank)

            if (cv2.waitKey(30) & 0xFF) == ord("q"):
                break

            if fail_reads > 60:
                print("[ERROR] Webcam opened but frames cannot be read.")
                print("Close other apps using the camera and try a different --cam index.")
                break
            continue

        fail_reads = 0

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = mesh.process(rgb)

        label_text = "NO FACE"
        smooth = None
        is_happy = None

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
            is_happy = smooth >= theta
            label_text = "HAPPY" if is_happy else "NOT HAPPY"

        # Overlay color
        if is_happy is None:
            color = (200, 200, 200)  # gray
            fill_color = (160, 160, 160)
        elif is_happy:
            color = (0, 255, 0)      # green
            fill_color = (0, 255, 0)
        else:
            color = (0, 0, 255)      # red
            fill_color = (0, 0, 255)

        # Big label
        cv2.putText(frame, label_text, (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, color, 4)

        # Confidence bar (always show; if no face, conf=0)
        conf_for_bar = smooth if smooth is not None else 0.0
        cv2.putText(frame, "Happy confidence", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (235, 235, 235), 2)
        draw_confidence_bar(frame, conf_for_bar, theta, x=20, y=130, w=340, h=22, fill_color=fill_color)

        cv2.putText(frame, "Press 'q' to quit", (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (235, 235, 235), 2)

        cv2.imshow("Happy detector (real-time)", frame)

        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    mesh.close()


if __name__ == "__main__":
    main()

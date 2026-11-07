import os
import json
import glob
from collections import deque

from eye_tracking_facemesh import EyeTracker

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models
import mediapipe as mp


# -----------------------------
# Model builder (match training arch)
# -----------------------------
def make_model(arch: str, num_classes: int):
    arch = arch.lower()
    if arch == "resnet18":
        m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if arch == "mobilenet_v3_small":
        m = models.mobilenet_v3_small(weights=None)
        in_f = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_f, num_classes)
        return m
    if arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None)
        in_f = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_f, num_classes)
        return m
    raise ValueError(f"Unsupported arch: {arch}")


# -----------------------------
# Utilities
# -----------------------------
def clamp_bbox(x0, y0, x1, y1, w, h):
    x0 = max(0, min(w - 1, x0))
    y0 = max(0, min(h - 1, y0))
    x1 = max(0, min(w, x1))
    y1 = max(0, min(h, y1))
    if x1 <= x0:
        x1 = min(w, x0 + 1)
    if y1 <= y0:
        y1 = min(h, y0 + 1)
    return x0, y0, x1, y1


def crop_with_margin(frame, box, margin=0.30):
    x0, y0, x1, y1 = box
    h, w = frame.shape[:2]
    bw, bh = x1 - x0, y1 - y0
    mx, my = int(bw * margin), int(bh * margin)
    xx0 = max(0, x0 - mx)
    yy0 = max(0, y0 - my)
    xx1 = min(w, x1 + mx)
    yy1 = min(h, y1 + my)
    return frame[yy0:yy1, xx0:xx1]


def load_theta_from_file(path: str) -> dict:
    """
    Expects output from calibrate_raf7_thetas_top1.py:
    {
      "margin": 0.10 or 0.12,
      "thresholds_top1": {
        "angry": {"theta": ...},
        ...
      }
    }
    """
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    margin = float(d["margin"])
    th = {k: float(v["theta"]) for k, v in d["thresholds_top1"].items()}
    return {"margin": margin, "theta": th}


def expand_bbox_for_forehead(x0, y0, x1, y1, left_eye_xy, right_eye_xy,
                             target_eye_y=0.38, max_extra_top=0.90):
    """
    Expand bbox upward so the eye line sits around target_eye_y of the box height.
    This adds forehead space automatically for different face shapes.
    """
    bw = x1 - x0
    bh = y1 - y0
    if bw <= 0 or bh <= 0:
        return x0, y0, x1, y1

    eye_y = 0.5 * (left_eye_xy[1] + right_eye_xy[1])
    num = eye_y - y0

    denom = (1.0 - target_eye_y)
    if denom <= 1e-6:
        return x0, y0, x1, y1

    t = (target_eye_y * bh - num) / denom
    if t <= 0:
        return x0, y0, x1, y1

    t = min(t, max_extra_top * bh)
    y0_new = int(y0 - t)
    return x0, y0_new, x1, y1


def find_latest_run(model_dir: str) -> str:
    metas = glob.glob(os.path.join(model_dir, "*_meta.json"))
    if not metas:
        raise FileNotFoundError(f"No *_meta.json found in {model_dir}")
    latest = max(metas, key=os.path.getmtime)
    run_name = os.path.basename(latest).replace("_meta.json", "")
    return run_name


def pick_threshold_file(model_dir: str, run_name: str) -> str:
    """
    Pick the newest thresholds_top1 file for this run.
    Prefers files like *_thresholds_top1_*.json (p80_m10 etc).
    """
    patt = os.path.join(model_dir, f"{run_name}*thresholds_top1*.json")
    matches = glob.glob(patt)
    if not matches:
        raise FileNotFoundError(f"No thresholds_top1 json found for run={run_name} in {model_dir}")
    return max(matches, key=os.path.getmtime)


# -----------------------------
# Main
# -----------------------------
def main():
    # ----------------- Paths (Point 2) -----------------
    MODEL_DIR = "models_cnn_raf7"

    # If you want to force a specific run name, set it here:
    # RUN_NAME = "raf7_clean_kfold"
    RUN_NAME = None

    if RUN_NAME is None:
        RUN_NAME = find_latest_run(MODEL_DIR)

    META_PATH = os.path.join(MODEL_DIR, f"{RUN_NAME}_meta.json")
    CKPT_PATH = os.path.join(MODEL_DIR, f"{RUN_NAME}_best.pt")
    THR_PATH = pick_threshold_file(MODEL_DIR, RUN_NAME)

    if not os.path.isfile(CKPT_PATH):
        raise FileNotFoundError(f"Missing checkpoint: {CKPT_PATH}")

    # ----------------- Runtime knobs (Points 4 + 5) -----------------
    # Smoothing reduced (Point 4)
    SMOOTH = 7

    # Crop tuning (keep forehead, but not too much background)
    FACE_MARGIN = 0.30

    # K frames reduced for angry/disgust so they can actually appear (Point 4)
    K_DEFAULT = 3
    K_BY_CLASS = {
        "disgust": 1,
        "angry": 2,
        "fear": 2,
        "happy": 2,
        "neutral": 2,
        "surprise": 2,
        "sad": 4,  # keep sad stricter (your priority)
    }

    # Hysteresis so 1 uncertain frame doesn't instantly become OTHER
    OTHER_CLEAR_FRAMES = 3
    other_streak = 0

    # Camera rotation if needed later
    ROTATE = None  # "cw" or "ccw" or None

    # ----------------- Load meta -----------------
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    classes = meta["classes"]
    arch = meta["arch"]
    img_size = int(meta["img_size"])

    # ----------------- Load thresholds -----------------
    thr = load_theta_from_file(THR_PATH)
    calib_margin = float(thr["margin"])
    theta = dict(thr["theta"])

    # ----------------- Load model -----------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = make_model(arch, num_classes=len(classes)).to(device)

    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    # ----------------- EyeTracker -----------------
    eye = EyeTracker(
        smoothing=7,
        update_every=3,   # slightly lighter CPU load than 2
        mirror_view=False,
        x_thresh=(0.42, 0.58),
        y_thresh=(0.42, 0.58),
    )

    # ----------------- Webcam -----------------
    WINDOW_NAME = f"RAF7 CNN ({RUN_NAME})"
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam. Try closing other apps or use camera index 1.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720)

    fd = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

    prob_hist = {c: deque(maxlen=SMOOTH) for c in classes}

    last_candidate = "OTHER"
    streak = 0
    stable = "OTHER"

    print("Press 'q' to quit.")
    print("RUN_NAME:", RUN_NAME)
    print("META:", META_PATH)
    print("CKPT:", CKPT_PATH)
    print("THR :", THR_PATH)
    print("classes:", classes)
    print("arch:", arch, "img_size:", img_size)
    print("calibration margin:", calib_margin)
    print("SMOOTH:", SMOOTH, "K_BY_CLASS:", K_BY_CLASS)

    # ----------------- Requirements (Point 3) -----------------
    def requirements(emo: str):
        theta_req = theta.get(emo, 0.99)
        margin_req = calib_margin

        # Keep SAD strict (avoid false sad)
        if emo == "sad":
            theta_req = max(theta_req, 0.55)
            margin_req = max(margin_req, calib_margin)  # keep calibrated margin

        # Reduce false SURPRISE, but don't make it too strict (avoid "ALL OTHER")
        if emo == "surprise":
            theta_req = max(theta_req, 0.70)
            margin_req = max(margin_req, calib_margin)

        # Make HAPPY/NEUTRAL easier (they often have small margins in webcam)
        if emo in {"happy", "neutral"}:
            margin_req = min(margin_req, 0.05)

        # Neutral theta can be too high for webcam (optional)
        if emo == "neutral":
            theta_req = min(theta_req, 0.60)

        # IMPORTANT: Angry/Disgust loosening so they can appear on webcam
        # (This is what you asked for.)
        if emo == "angry":
            theta_req = min(theta_req, 0.55)
            margin_req = min(margin_req, 0.06)

        if emo == "disgust":
            theta_req = min(theta_req, 0.60)
            margin_req = min(margin_req, 0.05)

        return theta_req, margin_req

    # ----------------- Loop -----------------
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        if ROTATE == "cw":
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif ROTATE == "ccw":
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = fd.process(rgb)

        candidate = "OTHER"
        show_probs = {c: 0.0 for c in classes}
        raw_top = "NONE"
        raw_second = "NONE"
        raw_margin = 0.0

        if res.detections:
            det = res.detections[0]
            r = det.location_data.relative_bounding_box
            x0 = int(r.xmin * w)
            y0 = int(r.ymin * h)
            x1 = int((r.xmin + r.width) * w)
            y1 = int((r.ymin + r.height) * h)
            x0, y0, x1, y1 = clamp_bbox(x0, y0, x1, y1, w, h)

            # Forehead expansion using eye keypoints
            kps = det.location_data.relative_keypoints
            re = (kps[0].x * w, kps[0].y * h)  # right eye
            le = (kps[1].x * w, kps[1].y * h)  # left eye

            x0, y0, x1, y1 = expand_bbox_for_forehead(
                x0, y0, x1, y1,
                left_eye_xy=le, right_eye_xy=re,
                target_eye_y=0.38,  # tweak 0.36-0.42 if needed
                max_extra_top=0.90
            )
            x0, y0, x1, y1 = clamp_bbox(x0, y0, x1, y1, w, h)

            face = crop_with_margin(frame, (x0, y0, x1, y1), margin=FACE_MARGIN)
            if face.size > 0:
                pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
                x = tf(pil).unsqueeze(0).to(device)

                with torch.no_grad():
                    logits = model(x)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

                # smoothing
                for i, c in enumerate(classes):
                    prob_hist[c].append(float(probs[i]))
                for c in classes:
                    show_probs[c] = float(np.mean(prob_hist[c])) if prob_hist[c] else 0.0

                sorted_cs = sorted(classes, key=lambda c: show_probs[c], reverse=True)
                raw_top, raw_second = sorted_cs[0], sorted_cs[1]
                raw_margin = show_probs[raw_top] - show_probs[raw_second]

                # Try top1, then 2nd, then 3rd (helps when top1 gets blocked)
                candidate = "OTHER"
                for i in range(min(3, len(sorted_cs) - 1)):
                    emo = sorted_cs[i]
                    p = show_probs[emo]
                    p_next = show_probs[sorted_cs[i + 1]]
                    m = p - p_next

                    theta_req, margin_req = requirements(emo)

                    if (p >= theta_req) and (m >= margin_req):
                        candidate = emo
                        break

                cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 255), 2)
        else:
            # no face detected -> clear history slowly
            pass

        # ---------- stability + hysteresis ----------
        if candidate == last_candidate:
            streak += 1
        else:
            last_candidate = candidate
            streak = 1

        if candidate == "OTHER":
            other_streak += 1
            if other_streak >= OTHER_CLEAR_FRAMES:
                stable = "OTHER"
        else:
            other_streak = 0
            k_req = K_BY_CLASS.get(candidate, K_DEFAULT)
            if streak >= k_req:
                stable = candidate

        # ---------- gaze ----------
        gaze = eye.process(frame)
        if gaze is not None:
            eye.draw(frame, gaze, org=(20, 140))

        # ---------- display ----------
        cv2.putText(frame, stable.upper(), (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6,
                    (0, 255, 0) if stable != "OTHER" else (0, 0, 255), 4)

        # Debug line (top2)
        cv2.putText(frame,
                    f"RAW top={raw_top}:{show_probs.get(raw_top,0):.2f}  2nd={raw_second}:{show_probs.get(raw_second,0):.2f}  raw_margin={raw_margin:.2f}  cand={candidate}",
                    (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 2)

        # (Point 5) Show class probs for the ones you care about
        def gp(name):
            return float(show_probs.get(name, 0.0))

        cv2.putText(frame,
                    f"A:{gp('angry'):.2f}  D:{gp('disgust'):.2f}  F:{gp('fear'):.2f}  N:{gp('neutral'):.2f}  H:{gp('happy'):.2f}  S:{gp('sad'):.2f}",
                    (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)

        # Also show the calibration margin used
        cv2.putText(frame,
                    f"calib_margin={calib_margin:.2f}  smooth={SMOOTH}  other_clear={OTHER_CLEAR_FRAMES}",
                    (20, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        cv2.imshow(WINDOW_NAME, frame)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    fd.close()
    eye.close()


if __name__ == "__main__":
    main()
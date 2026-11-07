"""
FER ensemble webcam inference (averages predictions from multiple models)
=========================================================================
Loads two or more trained FERModel checkpoints and averages their softmax
outputs for more robust emotion predictions.

Usage:
  python cnn_webcam_ensemble.py
  python cnn_webcam_ensemble.py --run_names fer7_v9_5class_mixup fer7_v10_swin_5class
  python cnn_webcam_ensemble.py --smooth 5 --cam 1
"""

import os
import json
import glob
import argparse
from collections import deque

from eye_tracking_facemesh import EyeTracker

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models
import mediapipe as mp


# ── Backbone factory (must match training) ────────────────────────────────────

def make_backbone(arch: str):
    arch = arch.lower()

    if arch == "resnet18":
        m = models.resnet18(weights=None)
        feat_dim = m.fc.in_features
        m.fc = nn.Identity()
        return m, feat_dim

    if arch == "resnet50":
        m = models.resnet50(weights=None)
        feat_dim = m.fc.in_features
        m.fc = nn.Identity()
        return m, feat_dim

    if arch == "mobilenet_v3_small":
        m = models.mobilenet_v3_small(weights=None)
        feat_dim = m.classifier[-1].in_features
        m.classifier[-1] = nn.Identity()
        return m, feat_dim

    if arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None)
        feat_dim = m.classifier[-1].in_features
        m.classifier[-1] = nn.Identity()
        return m, feat_dim

    if arch == "efficientnet_b2":
        m = models.efficientnet_b2(weights=None)
        feat_dim = m.classifier[-1].in_features
        m.classifier[-1] = nn.Identity()
        return m, feat_dim

    if arch == "swin_tiny":
        import timm
        m = timm.create_model("swin_tiny_patch4_window7_224",
                              pretrained=False, num_classes=0)
        feat_dim = m.num_features
        return m, feat_dim

    raise ValueError(f"Unsupported arch: {arch}")


class FERModel(nn.Module):
    def __init__(self, arch: str, num_classes: int = 7,
                 au_dim: int = 9, use_au: bool = True,
                 dropout: float = 0.2):
        super().__init__()
        self.backbone, feat_dim = make_backbone(arch)
        self.dropout = nn.Dropout(p=dropout)
        self.cls_head = nn.Linear(feat_dim, num_classes)
        self.use_au = use_au
        if use_au:
            self.au_head = nn.Sequential(
                nn.Linear(feat_dim, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.1),
                nn.Linear(256, au_dim),
            )

    def forward(self, x):
        feats = self.backbone(x)
        feats_d = self.dropout(feats)
        logits = self.cls_head(feats_d)
        au_pred = self.au_head(feats) if self.use_au else None
        return logits, au_pred


# ── Utilities ─────────────────────────────────────────────────────────────────

def clamp_bbox(x0, y0, x1, y1, w, h):
    x0 = max(0, min(w - 1, x0))
    y0 = max(0, min(h - 1, y0))
    x1 = max(1, min(w, x1))
    y1 = max(1, min(h, y1))
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


def expand_bbox_for_forehead(x0, y0, x1, y1, left_eye_xy, right_eye_xy,
                             target_eye_y=0.38, max_extra_top=0.90):
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
    return x0, int(y0 - t), x1, y1


# ── Colours for each emotion ─────────────────────────────────────────────────

EMOTION_COLOURS = {
    "angry":    (0, 0, 255),      # red
    "disgust":  (0, 140, 0),      # dark green
    "fear":     (200, 100, 255),   # purple
    "happy":    (0, 220, 255),     # yellow
    "sad":      (255, 150, 0),     # blue
    "surprise": (0, 255, 255),     # cyan
    "neutral":  (200, 200, 200),   # grey
}

OTHER_COLOUR = (0, 0, 200)


# ── Model loading helper ─────────────────────────────────────────────────────

def load_model(model_dir, run_name, device):
    meta_path = os.path.join(model_dir, f"{run_name}_meta.json")
    ckpt_path = os.path.join(model_dir, f"{run_name}_best.pt")

    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Missing meta: {meta_path}")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    classes  = meta["classes"]
    arch     = meta["arch"]
    img_size = int(meta["img_size"])
    use_au   = bool(meta.get("use_au", False))
    au_cols  = meta.get("au_cols", [])
    dropout  = float(meta.get("dropout", 0.2))

    model = FERModel(
        arch=arch,
        num_classes=len(classes),
        au_dim=len(au_cols) if au_cols else 9,
        use_au=use_au,
        dropout=dropout,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    print(f"[MODEL] {run_name}  arch={arch}  img_size={img_size}  "
          f"epoch={ckpt.get('epoch','?')}  val_f1={ckpt.get('val_f1',0):.3f}")

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    return model, tf, classes


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="models_cnn_multi")
    ap.add_argument("--run_names", nargs="+",
                    default=["fer7_v9_5class_mixup", "fer7_v10_swin_5class"],
                    help="Run names to ensemble")
    ap.add_argument("--cam",       type=int, default=0, help="Camera index")
    ap.add_argument("--smooth",    type=int, default=7,
                    help="Temporal smoothing window (frames)")
    ap.add_argument("--face_margin", type=float, default=0.30)
    ap.add_argument("--min_conf",  type=float, default=0.40,
                    help="Min softmax confidence to display an emotion")
    ap.add_argument("--min_margin", type=float, default=0.08,
                    help="Min gap between top-1 and top-2 probabilities")
    ap.add_argument("--stability_k", type=int, default=3,
                    help="Consecutive frames needed to change displayed emotion")
    ap.add_argument("--rotate",    default=None, choices=["cw", "ccw"],
                    help="Rotate camera frame 90 degrees")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Load all models ───────────────────────────────────────────────────────
    model_list = []
    tf_list = []
    classes = None

    for rn in args.run_names:
        model, tf, cls = load_model(args.model_dir, rn, device)
        model_list.append(model)
        tf_list.append(tf)
        if classes is None:
            classes = cls
        else:
            assert cls == classes, (
                f"Class mismatch: {rn} has {cls} but expected {classes}. "
                "All ensemble models must share the same class list."
            )

    print(f"[ENSEMBLE] {len(model_list)} models loaded, classes={classes}")

    # ── Eye tracker ───────────────────────────────────────────────────────────
    eye = EyeTracker(
        smoothing=7,
        update_every=3,
        mirror_view=False,
        x_thresh=(0.42, 0.58),
        y_thresh=(0.42, 0.58),
    )

    # ── Face detector (Tasks API) ────────────────────────────────────────────
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _fd_model = os.path.join(_script_dir, "blaze_face_short_range.tflite")
    if not os.path.isfile(_fd_model):
        raise FileNotFoundError(
            f"FaceDetector model not found at: {_fd_model}\n"
            f"Download from: https://storage.googleapis.com/mediapipe-models/"
            f"face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
        )
    fd_options = mp.tasks.vision.FaceDetectorOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=_fd_model),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        min_detection_confidence=0.5,
    )
    fd = mp.tasks.vision.FaceDetector.create_from_options(fd_options)
    fd_timestamp_ms = 0

    # ── Webcam ────────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open camera {args.cam}. "
            "Try --cam 1 or close other apps using the camera."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    WINDOW = f"FER {len(classes)}-class ensemble ({len(model_list)} models)"
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 1280, 720)

    # ── Smoothing / stability state ───────────────────────────────────────────
    prob_hist = {c: deque(maxlen=args.smooth) for c in classes}

    last_candidate = "OTHER"
    streak = 0
    stable = "OTHER"
    other_streak = 0
    OTHER_CLEAR = 3

    # Per-class stability frames (higher = harder to switch to that emotion)
    K_BY_CLASS = {
        "angry": 2, "happy": 2, "neutral": 2,
        "surprise": 2, "sad": 4,
    }

    print(f"[CAM]   index={args.cam}  smooth={args.smooth}  "
          f"min_conf={args.min_conf}  min_margin={args.min_margin}")
    print("Press 'q' to quit.")

    # ── Loop ──────────────────────────────────────────────────────────────────
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        if args.rotate == "cw":
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif args.rotate == "ccw":
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        fd_timestamp_ms += 33
        res = fd.detect_for_video(mp_image, fd_timestamp_ms)

        candidate = "OTHER"
        show_probs = {c: 0.0 for c in classes}

        if res.detections:
            det = res.detections[0]
            bb = det.bounding_box
            x0 = bb.origin_x
            y0 = bb.origin_y
            x1 = bb.origin_x + bb.width
            y1 = bb.origin_y + bb.height
            x0, y0, x1, y1 = clamp_bbox(x0, y0, x1, y1, w, h)

            # forehead expansion via eye keypoints (normalized 0..1)
            kps = det.keypoints
            right_eye = (kps[0].x * w, kps[0].y * h)
            left_eye  = (kps[1].x * w, kps[1].y * h)

            x0, y0, x1, y1 = expand_bbox_for_forehead(
                x0, y0, x1, y1,
                left_eye_xy=left_eye, right_eye_xy=right_eye,
                target_eye_y=0.38, max_extra_top=0.90,
            )
            x0, y0, x1, y1 = clamp_bbox(x0, y0, x1, y1, w, h)

            face = crop_with_margin(frame, (x0, y0, x1, y1), margin=args.face_margin)
            if face.size > 0:
                pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))

                # Run each model and average probabilities
                all_probs = []
                with torch.no_grad():
                    for model, tf in zip(model_list, tf_list):
                        inp = tf(pil).unsqueeze(0).to(device)
                        logits, _au = model(inp)
                        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                        all_probs.append(probs)

                avg_probs = np.mean(all_probs, axis=0)

                # temporal smoothing
                for ci, c in enumerate(classes):
                    prob_hist[c].append(float(avg_probs[ci]))
                for c in classes:
                    show_probs[c] = float(np.mean(prob_hist[c])) if prob_hist[c] else 0.0

                # pick candidate: top-1 must exceed conf and margin thresholds
                sorted_cs = sorted(classes, key=lambda c: show_probs[c], reverse=True)
                for rank in range(min(3, len(sorted_cs) - 1)):
                    emo = sorted_cs[rank]
                    p = show_probs[emo]
                    p_next = show_probs[sorted_cs[rank + 1]]
                    margin = p - p_next

                    if p >= args.min_conf and margin >= args.min_margin:
                        candidate = emo
                        break

                # draw face box
                cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 255), 2)

        # ── stability + hysteresis ────────────────────────────────────────────
        if candidate == last_candidate:
            streak += 1
        else:
            last_candidate = candidate
            streak = 1

        if candidate == "OTHER":
            other_streak += 1
            if other_streak >= OTHER_CLEAR:
                stable = "OTHER"
        else:
            other_streak = 0
            k_req = K_BY_CLASS.get(candidate, args.stability_k)
            if streak >= k_req:
                stable = candidate

        # ── gaze ──────────────────────────────────────────────────────────────
        gaze = eye.process(frame)
        if gaze is not None:
            eye.draw(frame, gaze, org=(20, 160))

        # ── draw HUD ──────────────────────────────────────────────────────────
        colour = EMOTION_COLOURS.get(stable, OTHER_COLOUR)
        cv2.putText(frame, stable.upper(), (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, colour, 4)

        # probability bars
        bar_x = 20
        bar_y = 200
        bar_h = 18
        bar_max_w = 200
        for ci, c in enumerate(classes):
            p = show_probs[c]
            bw = int(p * bar_max_w)
            y_top = bar_y + ci * (bar_h + 6)
            bcol = EMOTION_COLOURS.get(c, (180, 180, 180))
            cv2.rectangle(frame, (bar_x, y_top), (bar_x + bw, y_top + bar_h), bcol, -1)
            cv2.rectangle(frame, (bar_x, y_top), (bar_x + bar_max_w, y_top + bar_h), (100, 100, 100), 1)
            cv2.putText(frame, f"{c[:3].upper()} {p:.2f}",
                        (bar_x + bar_max_w + 8, y_top + bar_h - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # debug line
        sorted_display = sorted(classes, key=lambda c: show_probs[c], reverse=True)
        top1 = sorted_display[0]
        top2 = sorted_display[1]
        cv2.putText(frame,
                    f"top1={top1}:{show_probs[top1]:.2f}  "
                    f"top2={top2}:{show_probs[top2]:.2f}  "
                    f"margin={show_probs[top1]-show_probs[top2]:.2f}  "
                    f"cand={candidate}",
                    (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                    (235, 235, 235), 1)

        cv2.imshow(WINDOW, frame)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    fd.close()
    eye.close()


if __name__ == "__main__":
    main()

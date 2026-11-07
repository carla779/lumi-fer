"""
FER Experiment Logger
=====================
Runs the FER model on webcam and logs per-frame predictions to CSV.
Designed to run alongside PsychoPy during the LUMI validation experiment.

Outputs:
  results/<participant_id>_fer_log.csv
  Columns: unix_ts, frame_idx, predicted_emotion, confidence,
           p_angry, p_happy, p_sad, p_surprise, p_neutral

Usage:
  python cnn_webcam_experiment.py --participant P001
  python cnn_webcam_experiment.py --participant P001 --run_name fer7_v9_5class_mixup
  python cnn_webcam_experiment.py --participant P001 --ensemble
"""

import os
import csv
import json
import time
import argparse
from collections import deque

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
    def __init__(self, arch, num_classes=7, au_dim=9, use_au=True, dropout=0.2):
        super().__init__()
        self.backbone, feat_dim = make_backbone(arch)
        self.dropout = nn.Dropout(p=dropout)
        self.cls_head = nn.Linear(feat_dim, num_classes)
        self.use_au = use_au
        if use_au:
            self.au_head = nn.Sequential(
                nn.Linear(feat_dim, 256), nn.ReLU(inplace=True),
                nn.Dropout(p=0.1), nn.Linear(256, au_dim),
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
    if x1 <= x0: x1 = min(w, x0 + 1)
    if y1 <= y0: y1 = min(h, y0 + 1)
    return x0, y0, x1, y1


def crop_with_margin(frame, box, margin=0.30):
    x0, y0, x1, y1 = box
    h, w = frame.shape[:2]
    bw, bh = x1 - x0, y1 - y0
    mx, my = int(bw * margin), int(bh * margin)
    return frame[max(0, y0-my):min(h, y1+my), max(0, x0-mx):min(w, x1+mx)]


def expand_bbox_for_forehead(x0, y0, x1, y1, left_eye_xy, right_eye_xy,
                             target_eye_y=0.38, max_extra_top=0.90):
    bw, bh = x1 - x0, y1 - y0
    if bw <= 0 or bh <= 0:
        return x0, y0, x1, y1
    eye_y = 0.5 * (left_eye_xy[1] + right_eye_xy[1])
    denom = 1.0 - target_eye_y
    if denom <= 1e-6:
        return x0, y0, x1, y1
    t = (target_eye_y * bh - (eye_y - y0)) / denom
    if t <= 0:
        return x0, y0, x1, y1
    t = min(t, max_extra_top * bh)
    return x0, int(y0 - t), x1, y1


def load_model(model_dir, run_name, device):
    meta_path = os.path.join(model_dir, f"{run_name}_meta.json")
    ckpt_path = os.path.join(model_dir, f"{run_name}_best.pt")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    classes = meta["classes"]
    arch = meta["arch"]
    img_size = int(meta["img_size"])
    use_au = bool(meta.get("use_au", False))
    au_cols = meta.get("au_cols", [])
    dropout = float(meta.get("dropout", 0.2))

    model = FERModel(arch=arch, num_classes=len(classes),
                     au_dim=len(au_cols) if au_cols else 9,
                     use_au=use_au, dropout=dropout).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"[MODEL] {run_name}  arch={arch}  epoch={ckpt.get('epoch','?')}  "
          f"val_f1={ckpt.get('val_f1',0):.3f}")

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])
    return model, tf, classes


EMOTION_COLOURS = {
    "angry": (0, 0, 255), "happy": (0, 220, 255), "sad": (255, 150, 0),
    "surprise": (0, 255, 255), "neutral": (200, 200, 200),
}
OTHER_COLOUR = (0, 0, 200)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--participant", required=True, help="Participant ID (e.g. P001)")
    ap.add_argument("--model_dir", default="models_cnn_multi")
    ap.add_argument("--run_name", default="fer7_v9_5class_mixup")
    ap.add_argument("--ensemble", action="store_true",
                    help="Ensemble v9 + v10 (requires both models)")
    ap.add_argument("--ensemble_runs", nargs="+",
                    default=["fer7_v9_5class_mixup", "fer7_v10_swin_5class"])
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--smooth", type=int, default=7)
    ap.add_argument("--face_margin", type=float, default=0.30)
    ap.add_argument("--min_conf", type=float, default=0.40)
    ap.add_argument("--min_margin", type=float, default=0.08)
    ap.add_argument("--stability_k", type=int, default=3)
    ap.add_argument("--rotate", default=None, choices=["cw", "ccw"])
    ap.add_argument("--output_dir", default="results")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Load model(s) ─────────────────────────────────────────────────────────
    if args.ensemble:
        model_list, tf_list = [], []
        classes = None
        for rn in args.ensemble_runs:
            model, tf, cls = load_model(args.model_dir, rn, device)
            model_list.append(model)
            tf_list.append(tf)
            if classes is None:
                classes = cls
            else:
                assert cls == classes, f"Class mismatch in ensemble"
        print(f"[ENSEMBLE] {len(model_list)} models")
    else:
        model, tf, classes = load_model(args.model_dir, args.run_name, device)
        model_list = [model]
        tf_list = [tf]

    # ── Prepare output CSV ────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, f"{args.participant}_fer_log.csv")

    header = ["unix_ts", "frame_idx", "predicted_emotion", "confidence",
              "smoothed_emotion", "smoothed_confidence", "face_detected"]
    header += [f"p_{c}" for c in classes]
    header += [f"smooth_p_{c}" for c in classes]

    log_file = open(log_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(log_file)
    writer.writerow(header)

    print(f"[LOG] Writing to {log_path}")
    print(f"[LOG] Columns: {header}")

    # ── Face detector ─────────────────────────────────────────────────────────
    fd = mp.solutions.face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.5)

    # ── Webcam ────────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {args.cam}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    WINDOW = f"FER Experiment — {args.participant}"
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 1280, 720)

    # ── Smoothing state ───────────────────────────────────────────────────────
    prob_hist = {c: deque(maxlen=args.smooth) for c in classes}
    last_candidate = "OTHER"
    streak = 0
    stable = "OTHER"
    other_streak = 0
    OTHER_CLEAR = 3
    K_BY_CLASS = {"angry": 2, "happy": 2, "neutral": 2, "surprise": 2, "sad": 4}

    frame_idx = 0
    print(f"[CAM] index={args.cam}  participant={args.participant}")
    print("Press 'q' to quit and save log.")

    # ── Loop ──────────────────────────────────────────────────────────────────
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        if args.rotate == "cw":
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif args.rotate == "ccw":
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        unix_ts = time.time()
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = fd.process(rgb)

        # defaults for this frame
        raw_probs = np.zeros(len(classes), dtype=np.float32)
        face_detected = 0
        candidate = "OTHER"
        show_probs = {c: 0.0 for c in classes}

        if res.detections:
            det = res.detections[0]
            bb = det.location_data.relative_bounding_box
            x0 = int(bb.xmin * w)
            y0 = int(bb.ymin * h)
            x1 = int((bb.xmin + bb.width) * w)
            y1 = int((bb.ymin + bb.height) * h)
            x0, y0, x1, y1 = clamp_bbox(x0, y0, x1, y1, w, h)

            kps = det.location_data.relative_keypoints
            right_eye = (kps[0].x * w, kps[0].y * h)
            left_eye = (kps[1].x * w, kps[1].y * h)
            x0, y0, x1, y1 = expand_bbox_for_forehead(
                x0, y0, x1, y1, left_eye, right_eye)
            x0, y0, x1, y1 = clamp_bbox(x0, y0, x1, y1, w, h)

            face = crop_with_margin(frame, (x0, y0, x1, y1), margin=args.face_margin)
            if face.size > 0:
                face_detected = 1
                pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))

                all_probs = []
                with torch.no_grad():
                    for mdl, tf in zip(model_list, tf_list):
                        inp = tf(pil).unsqueeze(0).to(device)
                        logits, _ = mdl(inp)
                        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                        all_probs.append(probs)
                raw_probs = np.mean(all_probs, axis=0)

                for ci, c in enumerate(classes):
                    prob_hist[c].append(float(raw_probs[ci]))
                for c in classes:
                    show_probs[c] = float(np.mean(prob_hist[c])) if prob_hist[c] else 0.0

                sorted_cs = sorted(classes, key=lambda c: show_probs[c], reverse=True)
                for rank in range(min(3, len(sorted_cs) - 1)):
                    emo = sorted_cs[rank]
                    p = show_probs[emo]
                    p_next = show_probs[sorted_cs[rank + 1]]
                    if p >= args.min_conf and (p - p_next) >= args.min_margin:
                        candidate = emo
                        break

                cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 255), 2)

        # ── stability ─────────────────────────────────────────────────────────
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
            if streak >= K_BY_CLASS.get(candidate, args.stability_k):
                stable = candidate

        # ── Write CSV row ─────────────────────────────────────────────────────
        raw_top = classes[int(np.argmax(raw_probs))] if face_detected else "none"
        raw_conf = float(np.max(raw_probs)) if face_detected else 0.0
        smooth_conf = show_probs.get(stable, 0.0)

        row = [f"{unix_ts:.4f}", frame_idx, raw_top, f"{raw_conf:.4f}",
               stable, f"{smooth_conf:.4f}", face_detected]
        row += [f"{raw_probs[ci]:.4f}" for ci in range(len(classes))]
        row += [f"{show_probs.get(c, 0.0):.4f}" for c in classes]
        writer.writerow(row)

        # ── HUD ───────────────────────────────────────────────────────────────
        colour = EMOTION_COLOURS.get(stable, OTHER_COLOUR)
        cv2.putText(frame, stable.upper(), (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, colour, 4)

        # recording indicator
        cv2.circle(frame, (w - 40, 40), 12, (0, 0, 255), -1)
        cv2.putText(frame, "REC", (w - 90, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, f"Frame {frame_idx}", (w - 180, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # probability bars
        bar_x, bar_y, bar_h, bar_max_w = 20, 200, 18, 200
        for ci, c in enumerate(classes):
            p = show_probs[c]
            y_top = bar_y + ci * (bar_h + 6)
            bcol = EMOTION_COLOURS.get(c, (180, 180, 180))
            cv2.rectangle(frame, (bar_x, y_top),
                          (bar_x + int(p * bar_max_w), y_top + bar_h), bcol, -1)
            cv2.rectangle(frame, (bar_x, y_top),
                          (bar_x + bar_max_w, y_top + bar_h), (100, 100, 100), 1)
            cv2.putText(frame, f"{c[:3].upper()} {p:.2f}",
                        (bar_x + bar_max_w + 8, y_top + bar_h - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        cv2.imshow(WINDOW, frame)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

        frame_idx += 1

    # ── Cleanup ───────────────────────────────────────────────────────────────
    log_file.close()
    cap.release()
    cv2.destroyAllWindows()
    fd.close()

    print(f"\n[DONE] {frame_idx} frames logged to {log_path}")
    print(f"[SYNC] Use onset_unix/offset_unix from PsychoPy trial_results.csv "
          f"to slice this log per trial.")


if __name__ == "__main__":
    main()

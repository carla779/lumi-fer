#!/usr/bin/env python3
"""
02_run_experiment.py  —  LUMI FER Validation Study

Pure OpenCV + pygame implementation (NO PsychoPy).
  - Video frames:  OpenCV VideoCapture  (any codec, no glitches)
  - Audio:         pygame.mixer         (synced with video)
  - Display:       OpenCV fullscreen    (no coordinate-system flips)
  - Text:          PIL / Pillow         (nice anti-aliased fonts)
  - Dialogs:       tkinter              (built-in, zero extra deps)
  - Webcam + FER:  OpenCV + MediaPipe + PyTorch  (unchanged)

Usage:
  python 02_run_experiment.py
"""

import os
import sys
import csv
import time
import random
import datetime
import tempfile
import subprocess
import threading
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
import json

# ── Text rendering ────────────────────────────────────────────────────────
from PIL import Image as PILImage, ImageDraw, ImageFont

# ── Audio ─────────────────────────────────────────────────────────────────
HAS_AUDIO = False
try:
    import pygame
    pygame.mixer.pre_init(frequency=44100, size=-16, channels=2, buffer=1024)
    pygame.mixer.init()
    HAS_AUDIO = True
except Exception:
    print("WARNING: pygame audio unavailable.  pip install pygame")

# ── FER dependencies (optional) ──────────────────────────────────────────
HAS_TORCH = False
try:
    import torch
    import torch.nn as nn
    from torchvision import transforms, models
    import mediapipe as mp
    HAS_TORCH = True
except ImportError:
    print("WARNING: torch / mediapipe not installed — FER disabled.")

# ── Demographics dialog ──────────────────────────────────────────────────
import tkinter as tk
from tkinter import ttk


# ════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════
STIMULI_DIR   = Path(r"C:\Users\carla\PyCharmMiscProject\stimuli_local")
DATA_DIR      = Path("data")
RAW_VIDEO_DIR = DATA_DIR / "raw_video"
FER_LOG_DIR   = DATA_DIR / "fer_logs"
RESULTS_DIR   = Path("results")
STIMULUS_CSV  = Path("stimulus_list.csv")

WASHOUT_DURATION = 3.0        # seconds between trials
BEEP_FREQ        = 440        # Hz
BEEP_DURATION    = 0.2        # seconds

# ── Webcam + FER ──────────────────────────────────────────────────────────
CAMERA_INDEX       = 0      # UVC camera
CAMERA_WIDTH       = 1280
CAMERA_HEIGHT      = 720
ENABLE_FER         = True
ENABLE_VIDEO_SAVE  = False   # disabled — .avi files fill disk fast; FER CSV has all the data
ENABLE_CAM_PREVIEW = True
CAM_PREVIEW_W      = 240      # pixels
CAM_PREVIEW_H      = 180
FER_INTERVAL       = 0.5      # seconds between classifications
DUAL_PARTICIPANT   = False    # run 2 participants side-by-side (left/right)

# ── ffmpeg ────────────────────────────────────────────────────────────────
FFMPEG_BIN = (
    r"C:\Users\carla\AppData\Local\Microsoft\WinGet\Packages"
    r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
    r"\ffmpeg-8.1-full_build\bin\ffmpeg.exe"
)

EMOTIONS          = ["happy", "sad", "angry", "surprised", "neutral"]
FITZPATRICK_TYPES = ["I", "II", "III", "IV", "V", "VI"]

# ── Screen ────────────────────────────────────────────────────────────────
SCREEN_W = 1536
SCREEN_H = 1024
BG_BGR   = (64, 17, 17)       # navy-blue  (PsychoPy's [-0.87,-0.87,-0.5])

IMAGE_DISPLAY_DURATION = 10.0  # seconds to show each image stimulus

WIN_NAME = "LUMI"              # OpenCV window name


# ════════════════════════════════════════════════════════════════════════════
# DISPLAY HELPERS
# ════════════════════════════════════════════════════════════════════════════

def _get_font(size=40):
    for name in ["arial.ttf", "Arial.ttf", "segoeui.ttf", "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def make_canvas():
    """Blank navy-blue canvas at screen resolution."""
    return np.full((SCREEN_H, SCREEN_W, 3), BG_BGR, dtype=np.uint8)


def render_text(canvas, text, font_size=40, color=(255, 255, 255), y_offset=0):
    """Draw centred multi-line text onto canvas (PIL)."""
    pil = PILImage.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    font = _get_font(font_size)
    bb = draw.multiline_textbbox((0, 0), text, font=font, align="center")
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    x = (SCREEN_W - tw) // 2
    y = (SCREEN_H - th) // 2 + y_offset
    draw.multiline_text((x, y), text, font=font, fill=color, align="center")
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def paste_preview(canvas, webcam_fer):
    """Paste live camera preview + FER label in the bottom-right corner."""
    if webcam_fer is None or webcam_fer.latest_frame is None:
        return
    frame = cv2.flip(webcam_fer.latest_frame, 1)        # mirror only (horizontal)
    preview = cv2.resize(frame, (CAM_PREVIEW_W, CAM_PREVIEW_H))
    margin_r, margin_b = 20, 50
    y1 = SCREEN_H - CAM_PREVIEW_H - margin_b
    x1 = SCREEN_W - CAM_PREVIEW_W - margin_r
    canvas[y1:y1 + CAM_PREVIEW_H, x1:x1 + CAM_PREVIEW_W] = preview
    cv2.rectangle(canvas, (x1 - 1, y1 - 1),
                  (x1 + CAM_PREVIEW_W + 1, y1 + CAM_PREVIEW_H + 1),
                  (255, 255, 255), 1)
    # Slot 0 = LEFT participant → green, Slot 1 = RIGHT participant → pink
    slot_colors = [(0, 255, 0), (180, 105, 255)]   # BGR: green, pink
    slot_labels = ["L", "R"]
    y_text = y1 + CAM_PREVIEW_H + 22
    for slot in range(len(webcam_fer.participant_ids)):
        if webcam_fer.trial_fer_results[slot]:
            last = webcam_fer.trial_fer_results[slot][-1]
            prefix = f"{slot_labels[slot]}: " if webcam_fer.dual else ""
            lbl = f"{prefix}{last['emotion'].upper()} ({last['confidence']:.0%})"
            cv2.putText(canvas, lbl, (x1, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, slot_colors[slot], 2,
                        cv2.LINE_AA)
            y_text += 22


def show(canvas):
    cv2.imshow(WIN_NAME, canvas)


def check_key(timeout_ms=1):
    """Return pressed key code, or -1."""
    return cv2.waitKey(timeout_ms) & 0xFF


def generate_beep():
    """Synthesise a short beep as a pygame Sound."""
    if not HAS_AUDIO:
        return None
    sr = 44100
    n = int(sr * BEEP_DURATION)
    t = np.linspace(0, BEEP_DURATION, n, endpoint=False)
    wave = (np.sin(2 * np.pi * BEEP_FREQ * t) * 16000).astype(np.int16)
    stereo = np.column_stack([wave, wave])
    return pygame.sndarray.make_sound(stereo)


# ════════════════════════════════════════════════════════════════════════════
# FER MODEL
# ════════════════════════════════════════════════════════════════════════════

_MODEL_TO_EXPERIMENT = {"surprise": "surprised"}

if HAS_TORCH:
    def _make_backbone(arch: str):
        arch = arch.lower()
        if arch == "efficientnet_b0":
            m = models.efficientnet_b0(weights=None)
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

    class _FERNet(nn.Module):
        def __init__(self, arch, num_classes, au_dim=9, use_au=True, dropout=0.2):
            super().__init__()
            self.backbone, feat_dim = _make_backbone(arch)
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
            return self.cls_head(self.dropout(feats))

    class FERModel:
        def __init__(self, run_name="fer7_v9_5class_mixup", model_dir=None):
            if model_dir is None:
                candidates = [
                    Path(r"C:\Users\carla\PyCharmMiscProject\models_cnn_multi"),
                    Path(__file__).parent.parent / "models_cnn_multi",
                    Path("models_cnn_multi"),
                ]
                for c in candidates:
                    if (c / f"{run_name}_meta.json").exists():
                        model_dir = c
                        break
                if model_dir is None:
                    raise FileNotFoundError(
                        f"Cannot find {run_name}_meta.json. "
                        f"Searched: {[str(c) for c in candidates]}")

            meta_path = Path(model_dir) / f"{run_name}_meta.json"
            ckpt_path = Path(model_dir) / f"{run_name}_best.pt"
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            self.classes  = meta["classes"]
            arch          = meta["arch"]
            self.img_size = int(meta["img_size"])
            use_au        = bool(meta.get("use_au", False))
            au_cols       = meta.get("au_cols", [])
            dropout       = float(meta.get("dropout", 0.2))

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.net = _FERNet(
                arch=arch, num_classes=len(self.classes),
                au_dim=len(au_cols) if au_cols else 9,
                use_au=use_au, dropout=dropout,
            ).to(self.device)

            ckpt = torch.load(ckpt_path, map_location=self.device,
                              weights_only=False)
            self.net.load_state_dict(ckpt["state_dict"])
            self.net.eval()

            self.transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225]),
            ])

            _script_dir = Path(__file__).parent
            _fd_model = _script_dir / "blaze_face_short_range.tflite"
            if not _fd_model.exists():
                _fd_model = Path(
                    r"C:\Users\carla\PyCharmMiscProject"
                    r"\blaze_face_short_range.tflite")
            if not _fd_model.exists():
                raise FileNotFoundError(
                    "blaze_face_short_range.tflite not found.")
            fd_options = mp.tasks.vision.FaceDetectorOptions(
                base_options=mp.tasks.BaseOptions(
                    model_asset_path=str(_fd_model)),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                min_detection_confidence=0.5,
            )
            self.face_det = mp.tasks.vision.FaceDetector.create_from_options(
                fd_options)
            self._fd_timestamp_ms = 0

            print(f"  [FER] Loaded {run_name} ({arch}, "
                  f"{len(self.classes)}-class) on {self.device}")

        def predict(self, frame):
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            self._fd_timestamp_ms += 33
            res = self.face_det.detect_for_video(mp_image,
                                                 self._fd_timestamp_ms)
            if not res.detections:
                return "no_face", 0.0, {}

            det = res.detections[0]
            bb = det.bounding_box
            x0, y0 = max(0, bb.origin_x), max(0, bb.origin_y)
            x1 = min(w, bb.origin_x + bb.width)
            y1 = min(h, bb.origin_y + bb.height)
            bw, bh = x1 - x0, y1 - y0
            mx, my = int(bw * 0.3), int(bh * 0.3)
            x0, y0 = max(0, x0 - mx), max(0, y0 - my)
            x1, y1 = min(w, x1 + mx), min(h, y1 + my)

            face = frame[y0:y1, x0:x1]
            if face.size == 0:
                return "no_face", 0.0, {}

            pil = PILImage.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
            inp = self.transform(pil).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.net(inp)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

            idx = int(np.argmax(probs))
            model_label = self.classes[idx]
            confidence = float(probs[idx])
            exp_label = _MODEL_TO_EXPERIMENT.get(model_label, model_label)
            all_confs = {}
            for ci, cls_name in enumerate(self.classes):
                exp_name = _MODEL_TO_EXPERIMENT.get(cls_name, cls_name)
                all_confs[exp_name] = float(probs[ci])
            return exp_label, confidence, all_confs

        def predict_all(self, frame):
            """Detect ALL faces, return list sorted left-to-right from the
            participants' perspective (mirrored / selfie view).
            Each element: (emotion, confidence, all_confs).
            """
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            self._fd_timestamp_ms += 33
            res = self.face_det.detect_for_video(mp_image,
                                                  self._fd_timestamp_ms)
            if not res.detections:
                return []

            # Sort detections left-to-right from PARTICIPANT perspective
            # (mirrored: camera-left = participant-right, so reverse sort)
            dets_sorted = sorted(
                res.detections,
                key=lambda d: d.bounding_box.origin_x + d.bounding_box.width / 2,
                reverse=True)

            results = []
            for det in dets_sorted:
                bb = det.bounding_box
                x0, y0 = max(0, bb.origin_x), max(0, bb.origin_y)
                x1 = min(w, bb.origin_x + bb.width)
                y1 = min(h, bb.origin_y + bb.height)
                bw, bh = x1 - x0, y1 - y0
                mx, my = int(bw * 0.3), int(bh * 0.3)
                x0, y0 = max(0, x0 - mx), max(0, y0 - my)
                x1, y1 = min(w, x1 + mx), min(h, y1 + my)

                face = frame[y0:y1, x0:x1]
                if face.size == 0:
                    continue

                pil = PILImage.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
                inp = self.transform(pil).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    logits = self.net(inp)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

                idx = int(np.argmax(probs))
                model_label = self.classes[idx]
                confidence = float(probs[idx])
                exp_label = _MODEL_TO_EXPERIMENT.get(model_label, model_label)
                all_confs = {}
                for ci, cls_name in enumerate(self.classes):
                    exp_name = _MODEL_TO_EXPERIMENT.get(cls_name, cls_name)
                    all_confs[exp_name] = float(probs[ci])
                results.append((exp_label, confidence, all_confs))

            return results
else:
    FERModel = None


# ════════════════════════════════════════════════════════════════════════════
# WEBCAM + FER ENGINE  (pure OpenCV — no PsychoPy)
# ════════════════════════════════════════════════════════════════════════════

class WebcamFER:
    def __init__(self, fer_model, output_dir, participant_ids):
        """participant_ids: single string OR list of two strings for dual mode."""
        self.fer_model = fer_model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if isinstance(participant_ids, str):
            participant_ids = [participant_ids]
        self.participant_ids = participant_ids
        self.participant_id = participant_ids[0]   # backward compat
        self.dual = len(participant_ids) == 2
        self.cap = None
        self.video_writer = None
        self.recording = False
        # Per-slot results (slot 0 = left, slot 1 = right)
        self.trial_fer_results = [[] for _ in participant_ids]
        self.last_fer_time = 0
        self.latest_frame = None
        self.fer_log_paths = [None] * len(participant_ids)
        self.fer_log_files = [None] * len(participant_ids)
        self.fer_csv_writers = [None] * len(participant_ids)
        self.current_stimulus = None
        # Background thread
        self._thread = None
        self._running = False
        self._lock = threading.Lock()

    def open_camera(self, camera_index=CAMERA_INDEX):
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            print(f"  WARNING: Could not open camera {camera_index}")
            return False
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"  Webcam opened: {w}x{h} (index {camera_index})")
        # Start background capture thread
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        return True

    def open_fer_log(self, fer_log_dir, participant_id=None):
        header = [
            "timestamp", "stimulus_id", "fer_emotion", "fer_confidence",
            "conf_happy", "conf_sad", "conf_angry", "conf_surprised",
            "conf_neutral", "face_detected"
        ]
        for i, pid in enumerate(self.participant_ids):
            path = Path(fer_log_dir) / f"{pid}_fer_log.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            f = open(path, "w", newline="", encoding="utf-8")
            w = csv.writer(f)
            w.writerow(header)
            self.fer_log_paths[i] = path
            self.fer_log_files[i] = f
            self.fer_csv_writers[i] = w
            pos = ["LEFT", "RIGHT"][i] if self.dual else ""
            print(f"  FER log {pos}: {path}")

    def start_trial(self, stimulus_id):
        with self._lock:
            self.current_stimulus = stimulus_id
            self.trial_fer_results = [[] for _ in self.participant_ids]
            self.last_fer_time = 0
        if ENABLE_VIDEO_SAVE and self.cap and self.cap.isOpened():
            fn = self.output_dir / f"{self.participant_id}_{stimulus_id}.avi"
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            with self._lock:
                self.video_writer = cv2.VideoWriter(str(fn), fourcc, fps, (w, h))
                self.recording = True

    def _capture_loop(self):
        """Background thread: grabs webcam frames, records video, runs FER."""
        while self._running:
            if self.cap is None or not self.cap.isOpened():
                time.sleep(0.01)
                continue
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.005)
                continue

            self.latest_frame = frame  # atomic ref assignment — safe

            with self._lock:
                if self.recording and self.video_writer is not None:
                    self.video_writer.write(frame)

            now = time.time()
            if ENABLE_FER and self.fer_model and (now - self.last_fer_time) >= FER_INTERVAL:
                try:
                    if self.dual:
                        all_faces = self.fer_model.predict_all(frame)
                        # Pad to 2 slots: left face → slot 0, right → slot 1
                        while len(all_faces) < 2:
                            all_faces.append(("no_face", 0.0, {}))
                        face_list = all_faces[:2]
                    else:
                        emotion, confidence, all_confs = self.fer_model.predict(frame)
                        face_list = [(emotion, confidence, all_confs)]

                    for slot, (emotion, confidence, all_confs) in enumerate(face_list):
                        face_det = emotion != "no_face"
                        with self._lock:
                            self.trial_fer_results[slot].append({
                                "emotion": emotion, "confidence": confidence,
                                "timestamp": now, "face_detected": face_det,
                                "all_confs": all_confs,
                            })
                            if self.fer_csv_writers[slot]:
                                self.fer_csv_writers[slot].writerow([
                                    f"{now:.3f}", self.current_stimulus or "",
                                    emotion, f"{confidence:.3f}",
                                    f"{all_confs.get('happy',0):.3f}",
                                    f"{all_confs.get('sad',0):.3f}",
                                    f"{all_confs.get('angry',0):.3f}",
                                    f"{all_confs.get('surprised',0):.3f}",
                                    f"{all_confs.get('neutral',0):.3f}",
                                    int(face_det),
                                ])
                                self.fer_log_files[slot].flush()
                except Exception as e:
                    print(f"  FER error: {e}")
                self.last_fer_time = now

            time.sleep(0.01)  # ~100 fps cap, avoid CPU spin

    def process_frame(self):
        """No-op — kept for compatibility. Background thread does the work."""
        pass

    def stop_trial(self):
        with self._lock:
            self.recording = False
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None

    def get_trial_fer_summary(self, slot=0):
        with self._lock:
            results = list(self.trial_fer_results[slot])
        emo_names = ["happy", "sad", "angry", "surprised", "neutral"]
        empty_avgs = {e: 0.0 for e in emo_names}
        if not results:
            return "no_data", 0.0, 0, empty_avgs
        emotions = [r["emotion"] for r in results]
        counter = Counter(emotions)
        majority, count = counter.most_common(1)[0]
        avg_conf = (sum(r["confidence"] for r in results
                        if r["emotion"] == majority) / count)
        # Average probability per emotion across all frames
        avg_probs = {}
        for e in emo_names:
            vals = [r["all_confs"].get(e, 0.0) for r in results
                    if "all_confs" in r]
            avg_probs[e] = sum(vals) / len(vals) if vals else 0.0
        return majority, avg_conf, len(results), avg_probs

    def release(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.stop_trial()
        if self.cap is not None:
            self.cap.release()
        for f in self.fer_log_files:
            if f is not None:
                f.close()


# ════════════════════════════════════════════════════════════════════════════
# DEMOGRAPHICS  (tkinter)
# ════════════════════════════════════════════════════════════════════════════

def collect_demographics():
    result = {}
    root = tk.Tk()
    root.title("LUMI FER Validation \u2014 Participant Info")
    _dlg_h = 780 if DUAL_PARTICIPANT else 420
    root.geometry(f"460x{_dlg_h}")
    root.resizable(False, False)
    root.update_idletasks()
    x = (root.winfo_screenwidth() - 460) // 2
    y = (root.winfo_screenheight() - 420) // 2
    root.geometry(f"+{x}+{y}")

    fr = ttk.Frame(root, padding=20)
    fr.pack(fill="both", expand=True)

    fields = {}
    row = 0

    def add_entry(label, default=""):
        nonlocal row
        ttk.Label(fr, text=label).grid(row=row, column=0, sticky="w", pady=5)
        var = tk.StringVar(value=default)
        ttk.Entry(fr, textvariable=var, width=25).grid(row=row, column=1, pady=5)
        fields[label] = var
        row += 1

    def add_combo(label, values):
        nonlocal row
        ttk.Label(fr, text=label).grid(row=row, column=0, sticky="w", pady=5)
        var = tk.StringVar()
        cb = ttk.Combobox(fr, textvariable=var, width=22, values=values,
                          state="readonly")
        cb.grid(row=row, column=1, pady=5)
        cb.current(0)
        fields[label] = var
        row += 1

    if DUAL_PARTICIPANT:
        add_entry("Participant ID (LEFT):", "P001")
        add_entry("Age (LEFT):")
        add_combo("Gender (LEFT):", ["Female", "Male", "Non-binary", "Prefer not to say"])
        add_entry("Ethnicity (LEFT):")
        add_combo("Fitzpatrick (LEFT):", FITZPATRICK_TYPES)
        add_combo("Glasses (LEFT):", ["No", "Yes \u2014 glasses", "Yes \u2014 contact lenses"])
        add_combo("Facial hair (LEFT):", ["No", "Yes"])

        # Separator
        ttk.Separator(fr, orient="horizontal").grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=10)
        row += 1

        add_entry("Participant ID (RIGHT):", "P002")
        add_entry("Age (RIGHT):")
        add_combo("Gender (RIGHT):", ["Female", "Male", "Non-binary", "Prefer not to say"])
        add_entry("Ethnicity (RIGHT):")
        add_combo("Fitzpatrick (RIGHT):", FITZPATRICK_TYPES)
        add_combo("Glasses (RIGHT):", ["No", "Yes \u2014 glasses", "Yes \u2014 contact lenses"])
        add_combo("Facial hair (RIGHT):", ["No", "Yes"])
    else:
        add_entry("Participant ID:", "P001")
        add_entry("Age:")
        add_combo("Gender:", ["Female", "Male", "Non-binary", "Prefer not to say"])
        add_entry("Ethnicity:")
        add_combo("Fitzpatrick Type:", FITZPATRICK_TYPES)
        add_combo("Glasses:", ["No", "Yes \u2014 glasses", "Yes \u2014 contact lenses"])
        add_combo("Facial hair:", ["No", "Yes"])

    submitted = [False]

    def on_submit():
        if DUAL_PARTICIPANT:
            result["participant_id"]    = fields["Participant ID (LEFT):"].get()
            result["age"]               = fields["Age (LEFT):"].get()
            result["gender"]            = fields["Gender (LEFT):"].get()
            result["ethnicity"]         = fields["Ethnicity (LEFT):"].get()
            result["fitzpatrick"]       = fields["Fitzpatrick (LEFT):"].get()
            result["glasses"]           = fields["Glasses (LEFT):"].get()
            result["facial_hair"]       = fields["Facial hair (LEFT):"].get()
            result["session_date"]      = datetime.datetime.now().isoformat()
            result["participant_id_r"]  = fields["Participant ID (RIGHT):"].get()
            result["age_r"]             = fields["Age (RIGHT):"].get()
            result["gender_r"]          = fields["Gender (RIGHT):"].get()
            result["ethnicity_r"]       = fields["Ethnicity (RIGHT):"].get()
            result["fitzpatrick_r"]     = fields["Fitzpatrick (RIGHT):"].get()
            result["glasses_r"]         = fields["Glasses (RIGHT):"].get()
            result["facial_hair_r"]     = fields["Facial hair (RIGHT):"].get()
        else:
            result["participant_id"]  = fields["Participant ID:"].get()
            result["age"]             = fields["Age:"].get()
            result["gender"]          = fields["Gender:"].get()
            result["ethnicity"]       = fields["Ethnicity:"].get()
            result["fitzpatrick"]     = fields["Fitzpatrick Type:"].get()
            result["glasses"]         = fields["Glasses:"].get()
            result["facial_hair"]     = fields["Facial hair:"].get()
            result["session_date"]    = datetime.datetime.now().isoformat()
        submitted[0] = True
        root.destroy()

    btn = ttk.Frame(fr)
    btn.grid(row=row, column=0, columnspan=2, pady=20)
    ttk.Button(btn, text="Start Experiment", command=on_submit).pack(
        side="left", padx=10)
    ttk.Button(btn, text="Cancel", command=root.destroy).pack(
        side="left", padx=10)

    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()

    if not submitted[0]:
        print("Experiment cancelled.")
        sys.exit(0)
    return result


# ════════════════════════════════════════════════════════════════════════════
# STIMULUS LOADER
# ════════════════════════════════════════════════════════════════════════════

def load_stimulus_list(csv_path):
    if not csv_path.exists():
        print(f"ERROR: Stimulus list not found at {csv_path}")
        sys.exit(1)
    stimuli = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row["stimulus_id"]
            filepath = None
            for ext in [".mp4", ".jpg", ".png"]:
                candidate = STIMULI_DIR / f"{sid}{ext}"
                if candidate.exists():
                    filepath = candidate
                    break
            if filepath is None:
                filepath = STIMULI_DIR / f"{sid}.mp4"
            row["filepath"] = str(filepath)
            row["is_image"] = filepath.suffix.lower() in [".jpg", ".jpeg", ".png"]
            row["exists"] = filepath.exists()
            stimuli.append(row)
    missing = [s["stimulus_id"] for s in stimuli if not s["exists"]]
    if missing:
        print(f"WARNING: {len(missing)} stimulus files not found:")
        for m in missing[:5]:
            print(f"  - {STIMULI_DIR}/{m}.*")
        resp = input("Continue anyway? (y/n): ").strip().lower()
        if resp != "y":
            sys.exit(0)
    return stimuli


# ════════════════════════════════════════════════════════════════════════════
# EXPERIMENT PHASES
# ════════════════════════════════════════════════════════════════════════════

def _frame_loop(base_canvas, webcam_fer, duration=None, wait_keys=None):
    """
    Generic frame loop: show base_canvas with live preview overlay.
      duration  – seconds (None = wait for key only)
      wait_keys – list of key codes to accept (None = any / duration only)
    Returns the key code that ended the loop, or -1.
    """
    t0 = time.perf_counter()
    while True:
        canvas = base_canvas.copy()
        if webcam_fer is not None:
            webcam_fer.process_frame()
            if ENABLE_CAM_PREVIEW:
                paste_preview(canvas, webcam_fer)
        show(canvas)
        key = check_key(16)        # ~60 fps
        if key == 27:              # ESC always works
            return 27
        if wait_keys and key in wait_keys:
            return key
        if duration and (time.perf_counter() - t0) >= duration:
            return -1


def show_instructions(webcam_fer):
    txt = ("Welcome to the LUMI Facial Expression Recognition Study\n\n"
           "You will watch a series of short video clips and images.\n"
           "After each one, you will be asked which emotion you felt.\n\n"
           "Please keep your face visible to the camera at all times.\n"
           "Try to react naturally \u2014 there are no right or wrong answers.\n\n"
           "The session will take approximately 40 minutes.\n\n"
           "Press SPACE to begin.")
    base = render_text(make_canvas(), txt, font_size=34)
    _frame_loop(base, webcam_fer, wait_keys=[32])   # 32 = space


def run_baseline(duration, webcam_fer):
    # First 3 seconds: message + cross
    c1 = render_text(make_canvas(),
                     "Please relax and look at the cross.\n"
                     "This will take 30 seconds.",
                     font_size=32, y_offset=-80)
    c1 = render_text(c1, "+", font_size=80, y_offset=60)
    _frame_loop(c1, webcam_fer, duration=min(3.0, duration))

    if duration > 3.0:
        c2 = render_text(make_canvas(), "+", font_size=80)
        _frame_loop(c2, webcam_fer, duration=duration - 3.0)


def run_washout(duration, webcam_fer):
    base = render_text(make_canvas(), "+", font_size=80)
    _frame_loop(base, webcam_fer, duration=duration)


def play_stimulus(stim_path, is_image, beep_snd, webcam_fer,
                   preloaded_audio=None):
    """
    Play a video (OpenCV + pygame audio) or display an image.
    Returns (onset_unix, offset_unix, aborted).
    """
    # Beep
    if beep_snd is not None:
        beep_snd.play()
        time.sleep(BEEP_DURATION + 0.05)

    onset = time.time()
    aborted = False

    try:
        if is_image:
            # ── Image ──────────────────────────────────────────────────
            img = cv2.imread(str(stim_path))
            if img is None:
                raise RuntimeError(f"Cannot read image: {stim_path}")
            ih, iw = img.shape[:2]
            scale = min(SCREEN_W / iw, SCREEN_H / ih)
            nw, nh = int(iw * scale), int(ih * scale)
            img_r = cv2.resize(img, (nw, nh))

            base = make_canvas()
            y1 = (SCREEN_H - nh) // 2
            x1 = (SCREEN_W - nw) // 2
            base[y1:y1 + nh, x1:x1 + nw] = img_r

            key = _frame_loop(base, webcam_fer,
                              duration=IMAGE_DISPLAY_DURATION)
            if key == 27:
                aborted = True

        else:
            # ── Video ──────────────────────────────────────────────────
            vid = cv2.VideoCapture(str(stim_path))
            if not vid.isOpened():
                raise RuntimeError(f"Cannot open video: {stim_path}")

            fps = vid.get(cv2.CAP_PROP_FPS) or 30.0
            frame_dur = 1.0 / fps

            # Use pre-extracted audio (loaded at startup)
            audio_snd = preloaded_audio

            # Pre-compute scaled size (all frames are same resolution)
            vw = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
            vh = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
            scale = min(SCREEN_W / vw, SCREEN_H / vh)
            nw, nh = int(vw * scale), int(vh * scale)
            y1 = (SCREEN_H - nh) // 2
            x1 = (SCREEN_W - nw) // 2

            # Start audio + clock together
            if audio_snd is not None:
                audio_snd.play()
            clock_start = time.perf_counter()
            frame_idx = 0

            while True:
                elapsed = time.perf_counter() - clock_start
                target_frame = int(elapsed * fps)

                # Skip frames to stay in sync with audio
                frames_behind = target_frame - frame_idx
                if frames_behind > 1:
                    for _ in range(frames_behind - 1):
                        if not vid.grab():
                            break
                        frame_idx += 1

                ret, frame = vid.read()
                if not ret:
                    break
                frame_idx += 1

                # Render frame onto canvas
                frame_r = cv2.resize(frame, (nw, nh))
                canvas = make_canvas()
                canvas[y1:y1 + nh, x1:x1 + nw] = frame_r

                if webcam_fer is not None:
                    webcam_fer.process_frame()
                    if ENABLE_CAM_PREVIEW:
                        paste_preview(canvas, webcam_fer)

                show(canvas)

                # Wait only if we're ahead of schedule
                next_target = frame_idx * frame_dur
                now = time.perf_counter() - clock_start
                wait_ms = max(1, int((next_target - now) * 1000))
                key = check_key(wait_ms)
                if key == 27:
                    aborted = True
                    break

            vid.release()
            if audio_snd is not None:
                audio_snd.stop()

    except Exception as e:
        print(f"  WARNING: Could not play {stim_path}: {e}")
        err = render_text(make_canvas(),
                          f"[Playback error]\n{Path(stim_path).name}",
                          font_size=32, color=(0, 0, 255))
        _frame_loop(err, webcam_fer, duration=5.0)

    offset = time.time()
    return onset, offset, aborted


def collect_self_report(trial_num, total, webcam_fer, participant_ids=None):
    emo_map = {49: "happy", 50: "sad", 51: "angry",
               52: "surprised", 53: "neutral"}   # keys '1'-'5'
    keys_list = list(emo_map.keys())

    if DUAL_PARTICIPANT and participant_ids and len(participant_ids) == 2:
        results = []
        for slot, (pid, side) in enumerate(
                zip(participant_ids, ["LEFT", "RIGHT"])):
            txt = (f"Trial {trial_num}/{total}\n\n"
                   f"Participant {pid} ({side}):\n"
                   "Which emotion did you feel most strongly?\n\n"
                   "1 = Happy       2 = Sad       3 = Angry\n"
                   "4 = Surprised       5 = Neutral")
            base = render_text(make_canvas(), txt, font_size=38)
            key = _frame_loop(base, webcam_fer, wait_keys=keys_list)
            if key == 27:
                return "escape"
            chosen = emo_map.get(key, "unknown")
            conf = render_text(make_canvas(),
                               f"{pid}: {chosen.upper()}", font_size=64)
            _frame_loop(conf, webcam_fer, duration=0.5)
            results.append(chosen)
        return results   # [left_choice, right_choice]
    else:
        txt = (f"Trial {trial_num}/{total}\n\n"
               "Which emotion did you feel most strongly?\n\n"
               "1 = Happy       2 = Sad       3 = Angry\n"
               "4 = Surprised       5 = Neutral")
        base = render_text(make_canvas(), txt, font_size=38)
        key = _frame_loop(base, webcam_fer, wait_keys=keys_list)
        if key == 27:
            return "escape"
        chosen = emo_map.get(key, "unknown")
        conf = render_text(make_canvas(), chosen.upper(), font_size=64)
        _frame_loop(conf, webcam_fer, duration=0.5)
        return chosen


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    for d in [DATA_DIR, RAW_VIDEO_DIR, FER_LOG_DIR, RESULTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # Demographics (tkinter dialog — before fullscreen)
    demo = collect_demographics()
    pid = demo["participant_id"]
    if DUAL_PARTICIPANT:
        pid_r = demo["participant_id_r"]
        pids = [pid, pid_r]
        print(f"\nParticipant LEFT:  {pid}")
        print(f"Participant RIGHT: {pid_r}")
        # Save demographics for LEFT participant
        demo_left = {k: v for k, v in demo.items() if not k.endswith("_r")}
        demo_file = RESULTS_DIR / f"{pid}_demographics.csv"
        with open(demo_file, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=demo_left.keys())
            w.writeheader()
            w.writerow(demo_left)
        # Save demographics for RIGHT participant
        demo_right = {
            "participant_id": pid_r,
            "age": demo.get("age_r", ""),
            "gender": demo.get("gender_r", ""),
            "ethnicity": demo.get("ethnicity_r", ""),
            "fitzpatrick": demo.get("fitzpatrick_r", ""),
            "glasses": demo.get("glasses_r", ""),
            "facial_hair": demo.get("facial_hair_r", ""),
            "session_date": demo.get("session_date", ""),
        }
        demo_file_r = RESULTS_DIR / f"{pid_r}_demographics.csv"
        with open(demo_file_r, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=demo_right.keys())
            w.writeheader()
            w.writerow(demo_right)
    else:
        pids = [pid]
        print(f"\nParticipant: {pid}")
        print(f"Session:     {demo['session_date']}")
        demo_file = RESULTS_DIR / f"{pid}_demographics.csv"
        with open(demo_file, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=demo.keys())
            w.writeheader()
            w.writerow(demo)

    # Stimuli
    stimuli = load_stimulus_list(STIMULUS_CSV)
    stimuli = [s for s in stimuli if s["exists"]]
    print(f"  Loaded {len(stimuli)} stimuli")

    seed = hash(pid) % (2**32)
    rng = random.Random(seed)
    rng.shuffle(stimuli)
    print(f"  Randomised with seed {seed}")

    for p in pids:
        order_file = RESULTS_DIR / f"{p}_stimulus_order.csv"
        with open(order_file, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["trial_num", "stimulus_id", "target_emotion"])
            for i, s in enumerate(stimuli, 1):
                w.writerow([i, s["stimulus_id"], s["target_emotion"]])

    # Pre-extract audio for all video stimuli → avoids per-video ffmpeg delay
    audio_cache = {}  # stimulus_id → pygame.mixer.Sound or None
    if HAS_AUDIO:
        audio_dir = DATA_DIR / "audio_cache"
        audio_dir.mkdir(parents=True, exist_ok=True)
        ff = FFMPEG_BIN if os.path.isfile(FFMPEG_BIN) else "ffmpeg"
        video_stimuli = [s for s in stimuli if not s.get("is_image", False)]
        print(f"  Pre-extracting audio for {len(video_stimuli)} videos...")
        for s in video_stimuli:
            sid = s["stimulus_id"]
            wav_path = audio_dir / f"{sid}.wav"
            if not wav_path.exists():
                r = subprocess.run(
                    [ff, "-y", "-i", s["filepath"],
                     "-vn", "-acodec", "pcm_s16le",
                     "-ar", "44100", "-ac", "2",
                     "-loglevel", "error",
                     str(wav_path)],
                    capture_output=True, timeout=30,
                )
                if r.returncode != 0:
                    print(f"    Audio failed for {sid}: "
                          f"{r.stderr.decode(errors='replace')[:80]}")
                    continue
            if wav_path.exists() and wav_path.stat().st_size > 44:
                try:
                    audio_cache[sid] = pygame.mixer.Sound(str(wav_path))
                except Exception as e:
                    print(f"    Could not load audio for {sid}: {e}")
        print(f"  Audio ready: {len(audio_cache)}/{len(video_stimuli)} videos")

    # Beep
    beep = generate_beep()

    # Webcam + FER
    fer_model = FERModel() if (ENABLE_FER and FERModel) else None
    webcam_fer = WebcamFER(fer_model, RAW_VIDEO_DIR / pid, pids)
    cam_ok = webcam_fer.open_camera(CAMERA_INDEX)
    if cam_ok:
        webcam_fer.open_fer_log(FER_LOG_DIR)
    else:
        print("  Continuing without webcam / FER")
        webcam_fer = None

    # ── Fullscreen window ─────────────────────────────────────────────────
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WIN_NAME, cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN)
    # Show blank canvas so the window appears immediately
    show(make_canvas())
    check_key(100)

    # ── Results CSV ───────────────────────────────────────────────────────
    results_files = {}
    for p in pids:
        results_files[p] = RESULTS_DIR / f"{p}_trial_results.csv"
    results_fields = [
        "participant_id", "trial_num", "stimulus_id", "target_emotion",
        "source_database", "stimulus_type",
        "onset_unix", "offset_unix", "duration_actual_sec",
        "self_report_emotion", "self_report_correct",
        "fer_emotion", "fer_confidence", "fer_n_samples", "fer_correct",
        "fer_avg_happy", "fer_avg_sad", "fer_avg_angry",
        "fer_avg_surprised", "fer_avg_neutral",
    ]
    for rf in results_files.values():
        with open(rf, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=results_fields).writeheader()

    # ══════════════════════════════════════════════════════════════════════
    #  RUN EXPERIMENT
    # ══════════════════════════════════════════════════════════════════════
    try:
        show_instructions(webcam_fer)
        run_baseline(10.0, webcam_fer)

        total = len(stimuli)
        print(f"\n  Starting {total} trials...")

        for trial_idx, stim in enumerate(stimuli, 1):
            stim_id    = stim["stimulus_id"]
            stim_path  = stim["filepath"]
            target_emo = stim["target_emotion"]
            print(f"  Trial {trial_idx}/{total}: {stim_id} ({target_emo})")

            run_washout(WASHOUT_DURATION, webcam_fer)

            if webcam_fer is not None:
                webcam_fer.start_trial(stim_id)

            is_img = stim.get("is_image", False)
            cached_audio = audio_cache.get(stim_id)
            onset, offset, aborted = play_stimulus(
                stim_path, is_img, beep, webcam_fer,
                preloaded_audio=cached_audio)

            if webcam_fer is not None:
                webcam_fer.stop_trial()

            if aborted:
                print("  Experiment aborted (Escape).")
                break

            reported = collect_self_report(trial_idx, total, webcam_fer,
                                            participant_ids=pids)
            if reported == "escape":
                print("  Experiment aborted (Escape).")
                break

            # Normalise to list (single mode returns string)
            if isinstance(reported, str):
                reported_list = [reported]
            else:
                reported_list = reported

            for slot, p in enumerate(pids):
                self_rep = reported_list[slot] if slot < len(reported_list) else ""
                self_correct = 1 if self_rep == target_emo else 0

                fer_emo, fer_conf, fer_n = "", 0.0, 0
                fer_correct = ""
                fer_avg = {e: 0.0 for e in ["happy","sad","angry","surprised","neutral"]}
                if webcam_fer is not None:
                    fer_emo, fer_conf, fer_n, fer_avg = webcam_fer.get_trial_fer_summary(slot)
                    side = ["LEFT", "RIGHT"][slot] if len(pids) > 1 else ""
                    if fer_n > 0:
                        fer_correct = 1 if fer_emo == target_emo else 0
                        ok = "\u2713" if fer_correct else "\u2717"
                        print(f"    FER {side}: {fer_emo} ({fer_conf:.2f}, "
                              f"n={fer_n}) {ok}")
                        probs = " | ".join(f"{e[:3]}={fer_avg[e]:.2f}"
                                           for e in fer_avg)
                        print(f"    Avg {side}: {probs}")

                trial_data = {
                    "participant_id": p,
                    "trial_num": trial_idx,
                    "stimulus_id": stim_id,
                    "target_emotion": target_emo,
                    "source_database": stim.get("source_database", ""),
                    "stimulus_type": stim.get("stimulus_type", ""),
                    "onset_unix": f"{onset:.3f}" if onset else "",
                    "offset_unix": f"{offset:.3f}" if offset else "",
                    "duration_actual_sec": (f"{offset-onset:.2f}"
                                            if onset and offset else ""),
                    "self_report_emotion": self_rep,
                    "self_report_correct": self_correct,
                    "fer_emotion": fer_emo,
                    "fer_confidence": f"{fer_conf:.3f}" if fer_n > 0 else "",
                    "fer_n_samples": fer_n,
                    "fer_correct": fer_correct,
                    "fer_avg_happy": f"{fer_avg['happy']:.3f}" if fer_n > 0 else "",
                    "fer_avg_sad": f"{fer_avg['sad']:.3f}" if fer_n > 0 else "",
                    "fer_avg_angry": f"{fer_avg['angry']:.3f}" if fer_n > 0 else "",
                    "fer_avg_surprised": f"{fer_avg['surprised']:.3f}" if fer_n > 0 else "",
                    "fer_avg_neutral": f"{fer_avg['neutral']:.3f}" if fer_n > 0 else "",
                }
                with open(results_files[p], "a", newline="", encoding="utf-8") as f:
                    csv.DictWriter(f, fieldnames=results_fields).writerow(
                        trial_data)

        # ── End screen ────────────────────────────────────────────────────
        end = render_text(
            make_canvas(),
            "Thank you for participating!\n\n"
            "The experiment is now complete.\n"
            "Please let the experimenter know you have finished.\n\n"
            "Press SPACE to exit.",
            font_size=36)
        _frame_loop(end, webcam_fer, wait_keys=[32])

    except Exception as e:
        print(f"\n  ERROR: {e}")
        import traceback
        traceback.print_exc()

    finally:
        if webcam_fer is not None:
            webcam_fer.release()
        cv2.destroyAllWindows()
        if HAS_AUDIO:
            pygame.mixer.quit()

    for p in pids:
        print(f"\n  Results saved to {results_files[p]}")
    print(f"  FER logs in {FER_LOG_DIR}/")
    print("  Done!")


if __name__ == "__main__":
    main()

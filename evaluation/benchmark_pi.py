"""
benchmark_pi.py — Raspberry Pi FER pipeline benchmarking.

Run this on the Pi after the LUMI software is set up. It loads the trained
EfficientNet-B0 (v9) and Swin-Tiny (v10) checkpoints, runs the full pipeline
(face detection -> preprocessing -> model inference -> ensemble averaging
-> temporal smoothing) over a fixed input source, and reports per-stage
latency, throughput, RAM, CPU% and CPU temperature.

NO external hardware required for the timing/RAM/CPU/temperature numbers.
For POWER measurements you need a USB inline power meter — instructions at
the bottom; just record idle and active wattages and email them to Carla.

Usage on the Pi
---------------
    # 1) put benchmark_pi.py in the same folder as the trained models
    #    (the folder containing models_cnn_multi/ and the webcam scripts)
    # 2) run on a test video file (recommended — reproducible)
    python3 benchmark_pi.py --video test_clip.mp4 --frames 300

    # OR run on the live camera
    python3 benchmark_pi.py --cam 0 --frames 300

    # OR if both v9 and v10 checkpoints are not available, use just v9
    python3 benchmark_pi.py --video test_clip.mp4 --models v9_only

Outputs
-------
    benchmark_results.csv      — one row per frame, per-stage timing in ms
    benchmark_summary.txt      — aggregate stats (median, p95, FPS, RAM, temp)

Send both files to Carla.
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

# ---- Try to import dependencies, fail with a clear message if missing
try:
    import numpy as np
    import cv2
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    from PIL import Image
except ImportError as e:
    print(f"[ERROR] Missing dependency: {e}")
    print("Install with: pip install torch torchvision opencv-python pillow numpy")
    sys.exit(1)

try:
    import psutil
except ImportError:
    print("[WARN] psutil not installed — RAM and CPU% will be N/A.")
    print("       Install with: pip install psutil")
    psutil = None


# ----------------------------------------------------------------------
# Backbone factory (must match training script)
# ----------------------------------------------------------------------
def make_backbone(arch):
    arch = arch.lower()
    if arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None)
        feat_dim = m.classifier[-1].in_features
        m.classifier[-1] = nn.Identity()
        return m, feat_dim
    if arch == "swin_tiny":
        try:
            import timm
        except ImportError:
            raise ImportError("pip install timm  to use swin_tiny")
        m = timm.create_model("swin_tiny_patch4_window7_224",
                              pretrained=False, num_classes=0)
        return m, m.num_features
    raise ValueError(f"Unsupported arch: {arch}")


class FERModel(nn.Module):
    def __init__(self, arch, num_classes=5, au_dim=9, use_au=True, dropout=0.2):
        super().__init__()
        self.backbone, feat_dim = make_backbone(arch)
        self.dropout  = nn.Dropout(p=dropout)
        self.cls_head = nn.Linear(feat_dim, num_classes)
        self.use_au   = use_au
        if use_au:
            self.au_head = nn.Sequential(
                nn.Linear(feat_dim, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.1),
                nn.Linear(256, au_dim),
            )

    def forward(self, x):
        feats = self.backbone(x)
        logits = self.cls_head(self.dropout(feats))
        return logits


def load_model(model_dir, run_name, device):
    meta_path = os.path.join(model_dir, f"{run_name}_meta.json")
    ckpt_path = os.path.join(model_dir, f"{run_name}_best.pt")
    if not os.path.isfile(meta_path) or not os.path.isfile(ckpt_path):
        return None, None
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    model = FERModel(arch=meta["arch"],
                     num_classes=len(meta["classes"]),
                     au_dim=len(meta.get("au_cols", [])) or 9,
                     use_au=bool(meta.get("use_au", False)),
                     dropout=float(meta.get("dropout", 0.2))).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    img_size = int(meta["img_size"])
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return model, tf


# ----------------------------------------------------------------------
# System metric helpers
# ----------------------------------------------------------------------
def cpu_temp_celsius():
    """Read CPU temperature on Raspberry Pi."""
    paths = [
        "/sys/class/thermal/thermal_zone0/temp",
    ]
    for p in paths:
        try:
            with open(p) as f:
                return float(f.read().strip()) / 1000.0
        except FileNotFoundError:
            continue
    # Fallback: vcgencmd
    try:
        import subprocess
        out = subprocess.check_output(["vcgencmd", "measure_temp"]).decode().strip()
        return float(out.replace("temp=", "").replace("'C", ""))
    except Exception:
        return None


def system_snapshot():
    snap = {
        "cpu_pct": None,
        "ram_mb": None,
        "ram_pct": None,
        "cpu_temp_c": cpu_temp_celsius(),
    }
    if psutil is not None:
        proc = psutil.Process(os.getpid())
        snap["cpu_pct"] = psutil.cpu_percent(interval=None)  # since last call
        snap["ram_mb"]  = proc.memory_info().rss / (1024 * 1024)
        vm = psutil.virtual_memory()
        snap["ram_pct"] = vm.percent
    return snap


def detect_face_haar(frame, cascade):
    """Simple, dependency-free face detector to keep the script self-contained.
    If you have MediaPipe / MTCNN deployed already, swap in that path here."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5,
                                     minSize=(80, 80))
    if len(faces) == 0:
        return None
    # take the largest face
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
    pad = int(0.15 * max(w, h))
    x0 = max(0, x - pad); y0 = max(0, y - pad)
    x1 = min(frame.shape[1], x + w + pad); y1 = min(frame.shape[0], y + h + pad)
    return frame[y0:y1, x0:x1]


# ----------------------------------------------------------------------
# Main benchmarking loop
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="models_cnn_multi")
    ap.add_argument("--models", default="ensemble",
                    choices=["v9_only", "v10_only", "ensemble"])
    ap.add_argument("--video", default=None,
                    help="Path to a test video file (recommended for reproducibility)")
    ap.add_argument("--cam", type=int, default=None,
                    help="Camera index if no video file")
    ap.add_argument("--frames", type=int, default=300,
                    help="Number of frames to benchmark")
    ap.add_argument("--warmup", type=int, default=20,
                    help="Warm-up frames before measurement")
    ap.add_argument("--out_csv", default="benchmark_results.csv")
    ap.add_argument("--out_summary", default="benchmark_summary.txt")
    args = ap.parse_args()

    if args.video is None and args.cam is None:
        print("[ERROR] Specify either --video PATH or --cam N")
        sys.exit(1)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[DEVICE] {device}")

    # Pi info
    try:
        with open("/proc/device-tree/model") as f:
            print(f"[HARDWARE] {f.read().strip()}")
    except Exception:
        pass
    print(f"[OS] {sys.platform}  Python {sys.version.split()[0]}")
    print(f"[TORCH] {torch.__version__}")

    # Load models
    runs = []
    if args.models in ("v9_only", "ensemble"):
        runs.append("fer7_v9_5class_mixup")
    if args.models in ("v10_only", "ensemble"):
        runs.append("fer7_v10_swin_5class")

    model_list = []
    tf_list = []
    for rn in runs:
        m, tf = load_model(args.model_dir, rn, device)
        if m is None:
            print(f"[WARN] checkpoint missing for {rn} — skipping")
            continue
        n_params = sum(p.numel() for p in m.parameters())
        # model size on disk
        ckpt_size_mb = os.path.getsize(os.path.join(args.model_dir, f"{rn}_best.pt")) / (1024 * 1024)
        print(f"[MODEL] {rn}  params={n_params/1e6:.2f}M  ckpt={ckpt_size_mb:.1f} MB")
        model_list.append((rn, m, tf))
    if not model_list:
        print("[ERROR] No models loaded — aborting.")
        sys.exit(1)

    # Open input
    if args.video:
        cap = cv2.VideoCapture(args.video)
        print(f"[INPUT] video={args.video}")
    else:
        cap = cv2.VideoCapture(args.cam)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        print(f"[INPUT] camera index {args.cam}")
    if not cap.isOpened():
        print("[ERROR] Cannot open input.")
        sys.exit(1)

    # Face detector (Haar — dependency-free; swap in MediaPipe/MTCNN if available)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    print(f"[FACE-DETECTOR] OpenCV Haar cascade")

    # CSV logging
    csv_f = open(args.out_csv, "w", newline="")
    writer = csv.writer(csv_f)
    writer.writerow([
        "frame", "t_capture_ms", "t_face_det_ms", "t_preprocess_ms",
        "t_inference_ms", "t_postprocess_ms", "t_total_ms",
        "cpu_pct", "ram_mb", "ram_pct", "cpu_temp_c", "face_found",
    ])

    # Warm-up
    print(f"[WARMUP] running {args.warmup} frames...")
    for _ in range(args.warmup):
        ok, frame = cap.read()
        if not ok: break
        face = detect_face_haar(frame, cascade)
        if face is None or face.size == 0: continue
        pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        for _, m, tf in model_list:
            inp = tf(pil).unsqueeze(0).to(device)
            with torch.no_grad():
                _ = m(inp)
    if psutil is not None:
        psutil.cpu_percent(interval=None)  # prime cpu meter

    print(f"[RUN] {args.frames} measured frames...")
    t_run_start = time.perf_counter()
    n_done = 0

    while n_done < args.frames:
        # Capture
        t0 = time.perf_counter()
        ok, frame = cap.read()
        if not ok: break
        t1 = time.perf_counter()
        # Face detection
        face = detect_face_haar(frame, cascade)
        t2 = time.perf_counter()
        face_found = 0 if (face is None or face.size == 0) else 1

        if face_found:
            # Preprocess
            pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
            inps = [tf(pil).unsqueeze(0).to(device) for _, _, tf in model_list]
            t3 = time.perf_counter()
            # Inference (each model)
            probs_list = []
            with torch.no_grad():
                for (rn, m, _), inp in zip(model_list, inps):
                    logits = m(inp)
                    probs_list.append(torch.softmax(logits, dim=1).cpu().numpy()[0])
            t4 = time.perf_counter()
            # Post-process (ensemble average)
            avg_probs = np.mean(probs_list, axis=0) if len(probs_list) > 1 else probs_list[0]
            top1 = int(np.argmax(avg_probs))
            t5 = time.perf_counter()
        else:
            t3 = t4 = t5 = t2

        snap = system_snapshot()
        writer.writerow([
            n_done,
            (t1 - t0) * 1000,        # capture
            (t2 - t1) * 1000,        # face det
            (t3 - t2) * 1000,        # preprocess
            (t4 - t3) * 1000,        # inference
            (t5 - t4) * 1000,        # postprocess
            (t5 - t0) * 1000,        # total
            snap["cpu_pct"],
            snap["ram_mb"],
            snap["ram_pct"],
            snap["cpu_temp_c"],
            face_found,
        ])
        n_done += 1
        if n_done % 50 == 0:
            print(f"  [{n_done}/{args.frames}]  total={1000*(t5-t0):.1f} ms  "
                  f"temp={snap['cpu_temp_c']} C  ram_mb={snap['ram_mb']}")

    t_run_end = time.perf_counter()
    csv_f.close()
    cap.release()

    # ---- Summary
    import statistics as st
    rows = []
    with open(args.out_csv) as f:
        rd = csv.reader(f)
        header = next(rd)
        for r in rd:
            rows.append(r)
    if not rows:
        print("[ERROR] No frames captured.")
        sys.exit(1)

    def col(idx, rows=rows, only_face=False):
        face_idx = header.index("face_found")
        out = []
        for r in rows:
            if only_face and int(r[face_idx]) == 0:
                continue
            try:
                v = float(r[idx])
                out.append(v)
            except (ValueError, TypeError):
                continue
        return out

    def stats(vals, unit=""):
        if not vals: return f"N/A"
        med = st.median(vals)
        p95 = sorted(vals)[int(0.95 * (len(vals) - 1))]
        return f"median {med:.2f}{unit} | p95 {p95:.2f}{unit} | mean {st.mean(vals):.2f}{unit}"

    H = lambda name: header.index(name)
    elapsed = t_run_end - t_run_start
    fps = len(rows) / elapsed if elapsed > 0 else 0
    face_count = sum(int(r[H('face_found')]) for r in rows)

    summary = []
    summary.append("=" * 60)
    summary.append("LUMI FER pipeline benchmark — Raspberry Pi")
    summary.append("=" * 60)
    try:
        with open("/proc/device-tree/model") as f:
            summary.append(f"Hardware: {f.read().strip()}")
    except Exception:
        pass
    summary.append(f"Python:   {sys.version.split()[0]}")
    summary.append(f"PyTorch:  {torch.__version__}")
    summary.append(f"Device:   {device}")
    summary.append(f"Models:   {[rn for rn, _, _ in model_list]}")
    summary.append("")
    summary.append(f"Frames measured: {len(rows)}  (face found in {face_count})")
    summary.append(f"Wall time: {elapsed:.2f} s   sustained FPS: {fps:.2f}")
    summary.append("")
    summary.append("Per-stage latency (ms, only frames where a face was detected):")
    summary.append(f"  capture     : {stats(col(H('t_capture_ms'), only_face=True), ' ms')}")
    summary.append(f"  face detect : {stats(col(H('t_face_det_ms'), only_face=True), ' ms')}")
    summary.append(f"  preprocess  : {stats(col(H('t_preprocess_ms'), only_face=True), ' ms')}")
    summary.append(f"  inference   : {stats(col(H('t_inference_ms'), only_face=True), ' ms')}")
    summary.append(f"  postprocess : {stats(col(H('t_postprocess_ms'), only_face=True), ' ms')}")
    summary.append(f"  TOTAL       : {stats(col(H('t_total_ms'), only_face=True), ' ms')}")
    summary.append("")
    summary.append("System resources during run:")
    summary.append(f"  CPU %       : {stats(col(H('cpu_pct')), ' %')}")
    summary.append(f"  RAM (MB)    : {stats(col(H('ram_mb')), ' MB')}")
    summary.append(f"  RAM %       : {stats(col(H('ram_pct')), ' %')}")
    deg_c = "\u00b0C"
    summary.append(f"  CPU temp ({deg_c}): {stats(col(H('cpu_temp_c')), ' ' + deg_c)}")
    summary.append("")
    summary.append("Power measurements (manual — fill in if a USB inline meter is available):")
    summary.append("  Pi idle     : ____ W")
    summary.append("  Pi + camera + active inference : ____ W")
    summary.append("")
    summary.append("=" * 60)
    txt = "\n".join(summary)
    print()
    print(txt)
    with open(args.out_summary, "w") as f:
        f.write(txt + "\n")
    print(f"\n[OK] saved {args.out_csv} and {args.out_summary}")


if __name__ == "__main__":
    main()

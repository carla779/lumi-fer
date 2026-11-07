#!/usr/bin/env python3
"""
03_lumi_bridge.py
Run this on the Raspberry Pi alongside LUMI's FER system.

This script:
  1. Listens for trial markers from PsychoPy (ONSET/OFFSET)
  2. Runs your FER model on the Pi camera feed during each trial
  3. Sends each FER classification back to PsychoPy in real time

Network setup:
  - Pi listens on port 5005 for markers from PsychoPy
  - Pi sends FER results to PsychoPy's IP on port 5006

Usage:
  python 03_lumi_bridge.py --psychopy-ip 192.168.1.50

Prerequisites:
  pip install opencv-python numpy
  (+ whatever your FER model needs, e.g. tensorflow-lite, onnxruntime)
"""

import json
import time
import socket
import argparse
import threading
from collections import deque

# ══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════

LISTEN_PORT = 5005       # Receives markers from PsychoPy
SEND_PORT = 5006         # Sends FER results to PsychoPy
FER_INTERVAL = 0.5       # Seconds between FER classifications (adjust based on Pi speed)

EMOTIONS = ["happy", "sad", "angry", "surprised", "neutral"]


# ══════════════════════════════════════════════════════════════════════════
# FER MODEL — REPLACE THIS WITH YOUR ACTUAL MODEL
# ══════════════════════════════════════════════════════════════════════════

class FERModel:
    """
    Placeholder for your actual FER model.

    Replace the __init__ and predict methods with your real model.
    For example:
      - TFLite: load a .tflite model and use tflite_runtime
      - ONNX: load a .onnx model and use onnxruntime
      - OpenCV DNN: load a .caffemodel or .pb
      - Haar cascades + simple classifier

    The predict method should:
      1. Take an OpenCV frame (BGR numpy array)
      2. Detect the face
      3. Classify the facial expression
      4. Return (emotion_label, confidence_score)
    """

    def __init__(self):
        """Load your FER model here."""
        # ─── EXAMPLE: Replace with your actual model loading ───
        # self.face_cascade = cv2.CascadeClassifier(
        #     cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        # )
        # self.model = tf.lite.Interpreter(model_path='fer_model.tflite')
        # self.model.allocate_tensors()
        print("  [FER] Model loaded (placeholder — replace with your actual model)")

    def predict(self, frame):
        """
        Run FER on a single frame.

        Args:
            frame: OpenCV BGR image (numpy array)

        Returns:
            (emotion_label, confidence) where:
              - emotion_label is one of: "happy", "sad", "angry", "surprised", "neutral"
              - confidence is a float between 0.0 and 1.0
        """
        # ─── REPLACE THIS with your actual FER prediction ───
        #
        # Example structure:
        #   gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #   faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        #   if len(faces) == 0:
        #       return "neutral", 0.0
        #   x, y, w, h = faces[0]
        #   face_roi = gray[y:y+h, x:x+w]
        #   face_resized = cv2.resize(face_roi, (48, 48)) / 255.0
        #   prediction = self.model.predict(face_resized)
        #   emotion_idx = prediction.argmax()
        #   return EMOTIONS[emotion_idx], float(prediction[emotion_idx])
        #
        # For now, return a placeholder:
        import random
        emotion = random.choice(EMOTIONS)
        confidence = random.uniform(0.3, 0.95)
        return emotion, confidence


# ══════════════════════════════════════════════════════════════════════════
# BRIDGE LOGIC
# ══════════════════════════════════════════════════════════════════════════

class LumiBridge:
    """Receives markers from PsychoPy and sends back FER classifications."""

    def __init__(self, psychopy_ip, fer_model, fer_interval=FER_INTERVAL):
        self.psychopy_ip = psychopy_ip
        self.fer_model = fer_model
        self.fer_interval = fer_interval

        # Sockets
        self.recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.recv_sock.bind(("0.0.0.0", LISTEN_PORT))
        self.recv_sock.settimeout(0.1)

        self.send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # State
        self.running = True
        self.trial_active = False
        self.current_stimulus = None
        self.current_target = None

        # Camera
        self.cap = None

    def open_camera(self, camera_index=0):
        """Open the Pi camera."""
        try:
            import cv2
            self.cap = cv2.VideoCapture(camera_index)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 15)  # Pi3 may struggle with 30fps
                print(f"  Camera opened (index {camera_index})")
                return True
            else:
                print("  WARNING: Could not open camera")
                return False
        except ImportError:
            print("  WARNING: OpenCV not installed. Install with: pip install opencv-python")
            return False

    def send_fer(self, emotion, confidence):
        """Send a FER classification to PsychoPy."""
        msg = {
            "type": "FER",
            "emotion": emotion,
            "confidence": round(confidence, 3),
            "timestamp": time.time()
        }
        data = json.dumps(msg).encode("utf-8")
        self.send_sock.sendto(data, (self.psychopy_ip, SEND_PORT))

    def handle_marker(self, msg):
        """Process an incoming marker from PsychoPy."""
        msg_type = msg.get("type", "")

        if msg_type == "ONSET":
            self.trial_active = True
            self.current_stimulus = msg.get("stimulus_id", "?")
            self.current_target = msg.get("target_emotion", "?")
            print(f"  ▶ ONSET: {self.current_stimulus} (target: {self.current_target})")

        elif msg_type == "OFFSET":
            self.trial_active = False
            print(f"  ■ OFFSET: {msg.get('stimulus_id', '?')}")

        elif msg_type == "BASELINE_START":
            print("  ▶ Baseline started")

        elif msg_type == "BASELINE_END":
            print("  ■ Baseline ended")

        elif msg_type == "SESSION_END":
            print(f"  ✓ Session ended for {msg.get('participant_id', '?')}")
            self.running = False

        elif msg_type == "PING":
            print("  ← Ping received from PsychoPy")

    def run(self):
        """Main loop: listen for markers and run FER during active trials."""
        print(f"\n  Listening for PsychoPy markers on port {LISTEN_PORT}...")
        print(f"  Sending FER results to {self.psychopy_ip}:{SEND_PORT}")
        print(f"  FER interval: {self.fer_interval}s")
        print("  Waiting for experiment to start...\n")

        last_fer_time = 0

        while self.running:
            # Check for incoming markers
            try:
                data, addr = self.recv_sock.recvfrom(4096)
                msg = json.loads(data.decode("utf-8"))
                self.handle_marker(msg)
            except socket.timeout:
                pass
            except Exception as e:
                print(f"  Recv error: {e}")

            # Run FER during active trials
            if self.trial_active and (time.time() - last_fer_time) >= self.fer_interval:
                frame = None
                if self.cap is not None and self.cap.isOpened():
                    import cv2
                    ret, frame = self.cap.read()
                    if not ret:
                        frame = None

                if frame is not None:
                    emotion, confidence = self.fer_model.predict(frame)
                    self.send_fer(emotion, confidence)
                    last_fer_time = time.time()

        print("\n  Bridge stopped.")

    def close(self):
        if self.cap is not None:
            self.cap.release()
        self.recv_sock.close()
        self.send_sock.close()


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="LUMI FER Bridge for Raspberry Pi")
    parser.add_argument("--psychopy-ip", required=True,
                        help="IP address of the lab PC running PsychoPy (e.g. 192.168.1.50)")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera index (default: 0)")
    parser.add_argument("--fer-interval", type=float, default=FER_INTERVAL,
                        help=f"Seconds between FER classifications (default: {FER_INTERVAL})")
    args = parser.parse_args()

    fer_interval = args.fer_interval

    print("=" * 60)
    print("  LUMI FER Bridge")
    print("=" * 60)

    # Load FER model
    fer_model = FERModel()

    # Create bridge
    bridge = LumiBridge(args.psychopy_ip, fer_model, fer_interval)
    bridge.open_camera(args.camera)

    try:
        bridge.run()
    except KeyboardInterrupt:
        print("\n  Interrupted by user (Ctrl+C)")
    finally:
        bridge.close()


if __name__ == "__main__":
    main()

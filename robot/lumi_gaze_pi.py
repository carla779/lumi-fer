#!/usr/bin/env python3
"""
LUMI gaze tracker — Raspberry Pi 5 side.

Camera -> face detection -> (X, Y) in 0..1023 -> serial -> Arduino eye controller.

This is the "camera-to-servo loop" referenced in the design: it runs on its own,
independent of the language-model pipeline, so a slow LLM can never freeze the eyes.

Hardware assumptions:
  - USB webcam (OpenCV index 0).  For the Pi Camera Module, see note in capture section.
  - Arduino on /dev/ttyACM0 (USB).  Check with:  ls /dev/ttyACM* /dev/ttyUSB*

Dependencies (install once):
  sudo apt install python3-opencv
  pip3 install pyserial
"""

import cv2
import serial
import time

# ---------------- config ----------------
# ESP32 boards usually appear as /dev/ttyUSB0; some as /dev/ttyACM0. Auto-detect below.
SERIAL_PORT = None            # None = auto-detect; or hard-set e.g. "/dev/ttyUSB0"
BAUD        = 115200
CAM_INDEX   = 0
SMOOTH      = 0.25            # 0..1 ; lower = smoother/slower eyes, higher = snappier
SEND_HZ     = 30             # how often to send to the ESP32
RECENTER_WHEN_LOST = True    # ease eyes back to centre when no face is seen
# -----------------------------------------

import glob
def find_port():
    # prefer USB-serial (ESP32), then ACM (Arduino-style)
    for pattern in ("/dev/ttyUSB*", "/dev/ttyACM*"):
        hits = sorted(glob.glob(pattern))
        if hits:
            return hits[0]
    raise RuntimeError("No ESP32/Arduino serial port found. Plug it in, or set SERIAL_PORT manually.")

if SERIAL_PORT is None:
    SERIAL_PORT = find_port()
    print(f"Using serial port: {SERIAL_PORT}")

# face detector (lightweight Haar cascade, ships with OpenCV — fine for one face)
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

# open serial to the Arduino
ser = serial.Serial(SERIAL_PORT, BAUD, timeout=0)
time.sleep(2)   # let the Arduino reset after the port opens

# open camera
cap = cv2.VideoCapture(CAM_INDEX)
# For the Pi Camera Module instead of USB webcam, replace the line above with picamera2:
#   from picamera2 import Picamera2 ; picam = Picamera2() ; picam.start()
# and grab frames with  frame = picam.capture_array()
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# smoothed eye target, start centred (0..1023 scale the Arduino expects)
sx, sy = 512.0, 512.0
last_send = 0.0

def to_servo_range(cx, cy, w, h):
    """Map a face-centre pixel (cx,cy) in a w x h frame to the Arduino's 0..1023 X,Y.
    Note: image X grows right; we invert so the eyes look TOWARD the face (mirror),
    which feels natural to the child standing in front of LUMI."""
    x = (1.0 - cx / w) * 1023.0     # invert X so eyes mirror the viewer
    y = (cy / h) * 1023.0           # Y: top of frame -> look up (0)
    return x, y

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2,
                                              minNeighbors=5, minSize=(60, 60))

        if len(faces) > 0:
            # track the LARGEST face (closest / most prominent child)
            x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
            cx, cy = x + fw / 2.0, y + fh / 2.0
            tx, ty = to_servo_range(cx, cy, w, h)
        elif RECENTER_WHEN_LOST:
            tx, ty = 512.0, 512.0     # no face -> drift back to centre
        else:
            tx, ty = sx, sy           # no face -> hold last position

        # exponential smoothing so the eyes glide instead of twitching
        sx += (tx - sx) * SMOOTH
        sy += (ty - sy) * SMOOTH

        # send to Arduino at a fixed rate
        now = time.time()
        if now - last_send >= 1.0 / SEND_HZ:
            ser.write(f"{int(sx)},{int(sy)}\n".encode())
            last_send = now

        # (optional) press q in the preview window to quit; comment out if headless
        # cv2.imshow("LUMI gaze", frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

except KeyboardInterrupt:
    pass
finally:
    cap.release()
    ser.close()
    # cv2.destroyAllWindows()

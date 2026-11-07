"""
Eye/gaze tracker using MediaPipe Tasks API (0.10.33+).
Uses FaceLandmarker (VIDEO mode) for iris tracking + servo angle output.
"""

import os
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple, List

import cv2
import numpy as np
import mediapipe as mp

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# ── Landmark indices (478-point model: 468 mesh + 10 iris) ────────────────────

# Left eye contour (MediaPipe canonical indices)
LEFT_EYE_IDX = [
    362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398,
]
# Right eye contour
RIGHT_EYE_IDX = [
    33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
]
# Iris landmarks (always present in the 478-point model)
LEFT_IRIS_IDX = [473, 474, 475, 476, 477]   # 473 = center
RIGHT_IRIS_IDX = [468, 469, 470, 471, 472]  # 468 = center

# Default model path (relative to script directory)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_MODEL = os.path.join(_SCRIPT_DIR, "face_landmarker_v2.task")


@dataclass
class GazeResult:
    # normalized gaze within eye box: (0..1, 0..1)
    gx: float
    gy: float

    # labels
    dir_x: str  # LEFT/CENTER/RIGHT (screen coords)
    dir_y: str  # UP/CENTER/DOWN
    label: str  # combined like LEFT_UP, CENTER, ...

    # pixel positions for drawing
    left_iris_px: Tuple[int, int]
    right_iris_px: Tuple[int, int]

    # pixel coordinates of where the person is looking on the frame
    gaze_target_px: Tuple[int, int] = (0, 0)


class EyeTracker:
    """
    Coarse gaze estimator using MediaPipe FaceLandmarker (Tasks API).
    Uses iris landmarks for gaze direction + servo angle output.
    """

    def __init__(
        self,
        *,
        smoothing: int = 7,
        update_every: int = 2,
        mirror_view: bool = False,
        x_thresh: Tuple[float, float] = (0.42, 0.58),
        y_thresh: Tuple[float, float] = (0.42, 0.58),
        max_faces: int = 1,
        min_det_conf: float = 0.5,
        min_track_conf: float = 0.5,
        model_path: Optional[str] = None,
    ):
        self.mirror_view = mirror_view
        self.x_lo, self.x_hi = x_thresh
        self.y_lo, self.y_hi = y_thresh

        self._frame_i = 0
        self._timestamp_ms = 0
        self.update_every = max(1, int(update_every))
        self._last: Optional[GazeResult] = None

        self._hx = deque(maxlen=max(1, int(smoothing)))
        self._hy = deque(maxlen=max(1, int(smoothing)))

        # Resolve model path
        model = model_path or _DEFAULT_MODEL
        if not os.path.isfile(model):
            raise FileNotFoundError(
                f"FaceLandmarker model not found at: {model}\n"
                f"Download from: https://storage.googleapis.com/mediapipe-models/"
                f"face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            )

        # Create FaceLandmarker (VIDEO mode = synchronous per-frame)
        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model),
            running_mode=VisionRunningMode.VIDEO,
            num_faces=max_faces,
            min_face_detection_confidence=min_det_conf,
            min_face_presence_confidence=min_det_conf,
            min_tracking_confidence=min_track_conf,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self.landmarker = FaceLandmarker.create_from_options(options)

        # Store indices
        self.left_eye_idx = LEFT_EYE_IDX
        self.right_eye_idx = RIGHT_EYE_IDX
        self.left_iris_idx = LEFT_IRIS_IDX
        self.right_iris_idx = RIGHT_IRIS_IDX

    def close(self):
        try:
            self.landmarker.close()
        except Exception:
            pass

    @staticmethod
    def _lm_xy(lm, w: int, h: int) -> Tuple[float, float]:
        return float(lm.x * w), float(lm.y * h)

    def _box_from_indices(self, lms, idxs: List[int], w: int, h: int) -> Tuple[float, float, float, float]:
        xs, ys = [], []
        for i in idxs:
            if i >= len(lms):
                continue
            x, y = self._lm_xy(lms[i], w, h)
            xs.append(x)
            ys.append(y)
        if not xs:
            return 0.0, 0.0, 1.0, 1.0
        return min(xs), min(ys), max(xs), max(ys)

    def _center_from_indices(self, lms, idxs: List[int], w: int, h: int) -> Tuple[float, float]:
        xs, ys = [], []
        for i in idxs:
            if i >= len(lms):
                continue
            x, y = self._lm_xy(lms[i], w, h)
            xs.append(x)
            ys.append(y)
        if not xs:
            return float(w / 2), float(h / 2)
        return float(np.mean(xs)), float(np.mean(ys))

    @staticmethod
    def _dir_from_ratio(r: float, lo: float, hi: float, neg: str, mid: str, pos: str) -> str:
        if r < lo:
            return neg
        if r > hi:
            return pos
        return mid

    def process(self, frame_bgr) -> Optional[GazeResult]:
        """Return a smoothed gaze direction for the first detected face."""
        self._frame_i += 1
        if (self._frame_i % self.update_every) != 0:
            return self._last

        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Create MediaPipe Image and run detection
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._timestamp_ms += 33  # ~30 fps
        result = self.landmarker.detect_for_video(mp_image, self._timestamp_ms)

        if not result.face_landmarks:
            self._hx.clear()
            self._hy.clear()
            self._last = None
            return None

        lms = result.face_landmarks[0]  # List of NormalizedLandmark

        # iris centers (px)
        lix, liy = self._center_from_indices(lms, self.left_iris_idx, w, h)
        rix, riy = self._center_from_indices(lms, self.right_iris_idx, w, h)

        # eye boxes (px)
        lminx, lminy, lmaxx, lmaxy = self._box_from_indices(lms, self.left_eye_idx, w, h)
        rminx, rminy, rmaxx, rmaxy = self._box_from_indices(lms, self.right_eye_idx, w, h)

        # ratios inside each eye box
        def safe_ratio(x, a, b):
            denom = max(1e-6, (b - a))
            return float(np.clip((x - a) / denom, 0.0, 1.0))

        lgx = safe_ratio(lix, lminx, lmaxx)
        lgy = safe_ratio(liy, lminy, lmaxy)
        rgx = safe_ratio(rix, rminx, rmaxx)
        rgy = safe_ratio(riy, rminy, rmaxy)

        # combine both eyes (average)
        gx = float((lgx + rgx) / 2.0)
        gy = float((lgy + rgy) / 2.0)

        # smooth
        self._hx.append(gx)
        self._hy.append(gy)
        gx_s = float(np.mean(self._hx))
        gy_s = float(np.mean(self._hy))

        # classify directions in SCREEN coordinates
        dir_x = self._dir_from_ratio(gx_s, self.x_lo, self.x_hi, "LEFT", "CENTER", "RIGHT")
        dir_y = self._dir_from_ratio(gy_s, self.y_lo, self.y_hi, "UP", "CENTER", "DOWN")

        if self.mirror_view:
            if dir_x == "LEFT":
                dir_x = "RIGHT"
            elif dir_x == "RIGHT":
                dir_x = "LEFT"

        label = dir_x if dir_y == "CENTER" else f"{dir_x}_{dir_y}"

        gaze_px_x = int(gx_s * w)
        gaze_px_y = int(gy_s * h)

        out = GazeResult(
            gx=gx_s, gy=gy_s,
            dir_x=dir_x, dir_y=dir_y, label=label,
            left_iris_px=(int(lix), int(liy)),
            right_iris_px=(int(rix), int(riy)),
            gaze_target_px=(gaze_px_x, gaze_px_y),
        )
        self._last = out
        return out

    def draw(self, frame_bgr, gaze: GazeResult, org=(20, 140)):
        """Draw iris points, gaze target crosshair, and pixel coordinates."""
        cv2.circle(frame_bgr, gaze.left_iris_px, 3, (0, 255, 255), -1)
        cv2.circle(frame_bgr, gaze.right_iris_px, 3, (0, 255, 255), -1)

        tx, ty = gaze.gaze_target_px
        cv2.drawMarker(frame_bgr, (tx, ty), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)

        txt = f"GAZE: ({tx}, {ty})  {gaze.label}"
        cv2.putText(frame_bgr, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    @staticmethod
    def gaze_to_servo_angles(
        gaze: GazeResult,
        frame_w: int = 1280,
        frame_h: int = 720,
        h_range: Tuple[float, float] = (30.0, 150.0),
        v_range: Tuple[float, float] = (60.0, 120.0),
        h_center: float = 90.0,
        v_center: float = 90.0,
        deadzone: float = 0.05,
        invert_h: bool = True,
    ) -> Tuple[float, float]:
        """
        Convert gaze pixel coordinates to servo angles for LUMI's eyes.

        Parameters
        ----------
        gaze : GazeResult
            The gaze result from process().
        frame_w, frame_h : int
            Camera resolution (default 1280x720).
        h_range : (min_deg, max_deg)
            Horizontal servo range. Default (30, 150) = 120 degrees of travel.
        v_range : (min_deg, max_deg)
            Vertical servo range. Default (60, 120) = 60 degrees of travel.
        h_center, v_center : float
            Servo angle when looking straight ahead (default 90 = midpoint).
        deadzone : float
            Normalized gaze range around centre where servos stay still.
            Default 0.05 means gx/gy between 0.45-0.55 keeps servos at centre.
        invert_h : bool
            If True (default), horizontal axis is flipped so LUMI mirrors
            the person's gaze direction.

        Returns
        -------
        (h_angle, v_angle) : (float, float)
            Servo angles in degrees, clamped to the specified ranges.
        """
        gx = gaze.gx
        gy = gaze.gy

        if abs(gx - 0.5) < deadzone:
            gx = 0.5
        if abs(gy - 0.5) < deadzone:
            gy = 0.5

        h_min, h_max = h_range
        if invert_h:
            h_angle = h_max - gx * (h_max - h_min)
        else:
            h_angle = h_min + gx * (h_max - h_min)

        v_min, v_max = v_range
        v_angle = v_min + gy * (v_max - v_min)

        h_angle = float(np.clip(h_angle, h_min, h_max))
        v_angle = float(np.clip(v_angle, v_min, v_max))

        return h_angle, v_angle

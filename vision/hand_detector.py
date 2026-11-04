"""
vision/hand_detector.py
Modular, isolated hand detection interface for MediaPipe.
Provides seamless dual-backend support:
  1. MediaPipe Tasks API (modern laptops, Python 3.8 - 3.12+, mediapipe >= 0.10.x)
  2. MediaPipe Solutions API (NVIDIA Jetson Nano, Python 3.6, mediapipe == 0.8.5)

Stable interface contract:
  detector = HandDetector(min_confidence=0.5)
  landmarks = detector.detect(rgb_frame)  # returns Optional[List[Tuple[float, float, float]]] (21 items)
  detector.close()
"""

import os
import sys
import urllib.request
from pathlib import Path
from typing import Optional, List, Tuple

# Safe imports: Do not fail module import if mediapipe or numpy is missing
try:
    import numpy as np
except ImportError:
    np = None

try:
    import mediapipe as mp
except ImportError:
    mp = None

# Check Tasks API availability
TASKS_API_AVAILABLE = False
if mp is not None:
    try:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
        TASKS_API_AVAILABLE = True
    except (ImportError, AttributeError):
        TASKS_API_AVAILABLE = False

# Check Solutions API availability (MediaPipe 0.8.5 on Jetson Nano / Python 3.6)
SOLUTIONS_API_AVAILABLE = False
if mp is not None:
    try:
        if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'hands'):
            SOLUTIONS_API_AVAILABLE = True
    except (ImportError, AttributeError):
        SOLUTIONS_API_AVAILABLE = False


class MediaPipeNotAvailableError(Exception):
    """Raised when MediaPipe is not installed or neither Tasks nor Solutions backend could initialize."""
    pass


MODEL_FILENAME = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"


def _ensure_model_file() -> Optional[str]:
    """Ensures the MediaPipe Hand Landmarker model file exists locally for Tasks API."""
    dest_path = Path(__file__).resolve().parent.parent / MODEL_FILENAME
    if dest_path.exists():
        return str(dest_path)
    print(f"[HandDetector] Downloading HandLandmarker model to {dest_path}...")
    try:
        urllib.request.urlretrieve(MODEL_URL, str(dest_path))
        print("[HandDetector] Model downloaded successfully.")
        return str(dest_path)
    except Exception as e:
        print(f"[HandDetector] Failed to download model: {e}")
        return None


class BaseHandDetector:
    """Abstract base class for MediaPipe hand detector implementations."""

    def detect(self, rgb_frame) -> Optional[List[Tuple[float, float, float]]]:
        raise NotImplementedError

    def close(self):
        pass

    def get_backend_name(self) -> str:
        raise NotImplementedError


class TasksHandDetector(BaseHandDetector):
    """MediaPipe Tasks API backend (used on modern laptops with mediapipe >= 0.10.x)."""

    def __init__(self, min_confidence: float = 0.5):
        if not TASKS_API_AVAILABLE:
            raise MediaPipeNotAvailableError("MediaPipe Tasks API is not available.")

        model_path = _ensure_model_file()
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"Hand landmarker model file not found at {model_path}")

        base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=min_confidence,
            min_hand_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence
        )
        self.landmarker = mp_vision.HandLandmarker.create_from_options(options)

    def detect(self, rgb_frame) -> Optional[List[Tuple[float, float, float]]]:
        if not self.landmarker:
            return None
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self.landmarker.detect(mp_image)
        if result.hand_landmarks and len(result.hand_landmarks) > 0:
            hand = result.hand_landmarks[0]
            return [(lm.x, lm.y, lm.z) for lm in hand]
        return None

    def close(self):
        if self.landmarker:
            try:
                self.landmarker.close()
            except Exception:
                pass
            self.landmarker = None

    def get_backend_name(self) -> str:
        return "tasks"


class SolutionsHandDetector(BaseHandDetector):
    """MediaPipe Solutions API backend (used on NVIDIA Jetson Nano / Python 3.6 with mediapipe == 0.8.5)."""

    def __init__(self, min_confidence: float = 0.5):
        if not SOLUTIONS_API_AVAILABLE:
            raise MediaPipeNotAvailableError("MediaPipe Solutions API (mp.solutions.hands) is not available.")

        self.landmarker = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=min_confidence,
            min_tracking_confidence=min_confidence
        )

    def detect(self, rgb_frame) -> Optional[List[Tuple[float, float, float]]]:
        if not self.landmarker:
            return None
        result = self.landmarker.process(rgb_frame)
        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            return [(lm.x, lm.y, lm.z) for lm in hand.landmark]
        return None

    def close(self):
        if self.landmarker:
            try:
                self.landmarker.close()
            except Exception:
                pass
            self.landmarker = None

    def get_backend_name(self) -> str:
        return "solutions"


class HandDetector(BaseHandDetector):
    """
    Unified MediaPipe Hand Detector.
    Auto-detects and initializes the appropriate backend:
      - Preferred: TasksHandDetector (if mediapipe.tasks is present and model exists)
      - Fallback: SolutionsHandDetector (if mp.solutions.hands is present, e.g. Jetson Nano)
    Returns exactly 21 normalized (x, y, z) landmark coordinates or None.
    """

    def __init__(self, min_confidence: float = 0.5):
        self.min_confidence = min_confidence
        self.delegate: Optional[BaseHandDetector] = None

        if mp is None:
            raise MediaPipeNotAvailableError(
                "MediaPipe library is not installed. "
                "Run with Keyboard Mode or install MediaPipe dependencies."
            )

        # 1. Attempt Tasks API (Modern Laptops)
        if TASKS_API_AVAILABLE:
            try:
                self.delegate = TasksHandDetector(min_confidence=self.min_confidence)
                print("[HandDetector] Initialized MediaPipe HandLandmarker (Tasks API).")
                return
            except Exception as e:
                print(f"[HandDetector] Tasks API unavailable: {e}. Checking Solutions API...")

        # 2. Attempt Solutions API (Jetson Nano / Python 3.6)
        if SOLUTIONS_API_AVAILABLE:
            try:
                self.delegate = SolutionsHandDetector(min_confidence=self.min_confidence)
                print("[HandDetector] Initialized MediaPipe Hands (Solutions API).")
                return
            except Exception as e:
                print(f"[HandDetector] Solutions API error: {e}")

        raise MediaPipeNotAvailableError(
            "Neither MediaPipe Tasks API nor Solutions API is functional on this system."
        )

    def detect(self, rgb_frame) -> Optional[List[Tuple[float, float, float]]]:
        if self.delegate:
            return self.delegate.detect(rgb_frame)
        return None

    def close(self):
        if self.delegate:
            self.delegate.close()
            self.delegate = None

    def get_backend_name(self) -> str:
        return self.delegate.get_backend_name() if self.delegate else "none"


def is_mediapipe_available() -> bool:
    """Checks if any functional MediaPipe backend is present without raising an exception."""
    return TASKS_API_AVAILABLE or SOLUTIONS_API_AVAILABLE

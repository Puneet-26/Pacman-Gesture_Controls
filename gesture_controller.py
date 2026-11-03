"""
gesture_controller.py
Real-time webcam-based hand gesture controller for Pac-Man using OpenCV and MediaPipe.
Runs webcam capture and gesture recognition asynchronously in a background thread
so that the Pygame main game loop is never blocked or frozen.
"""

import math
import os
import sys
import threading
import time
import urllib.request
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import cv2
import numpy as np

# Try importing MediaPipe Tasks API, with fallback to Solutions API
import mediapipe as mp

try:
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    TASKS_API_AVAILABLE = True
except (ImportError, AttributeError):
    TASKS_API_AVAILABLE = False


class GestureType(Enum):
    """Enumeration of recognized hand gestures and game actions."""
    NONE = "NONE"
    UP = "UP"
    DOWN = "DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    PAUSE = "PAUSE"
    RESUME = "RESUME"


# Model download configuration for MediaPipe Tasks API
MODEL_FILENAME = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

# Hand skeleton connections (21 landmarks)
HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Knuckles / Palm Base
    (5, 9), (9, 13), (13, 17)
]


def _ensure_model_file() -> str:
    """Ensures the MediaPipe Hand Landmarker model file exists locally."""
    dest_path = Path(__file__).parent / MODEL_FILENAME
    if not dest_path.exists():
        print(f"[GestureController] Downloading HandLandmarker model to {dest_path}...")
        try:
            urllib.request.urlretrieve(MODEL_URL, str(dest_path))
            print("[GestureController] Model downloaded successfully.")
        except Exception as e:
            print(f"[GestureController] Failed to download model: {e}")
    return str(dest_path)


class HandDetector:
    """
    Encapsulates MediaPipe Hand Landmark detection.
    Supports MediaPipe Tasks API with automatic fallback to Solutions API.
    """

    def __init__(self, min_confidence: float = 0.5):
        self.min_confidence = min_confidence
        self.landmarker = None
        self.backend = None

        if TASKS_API_AVAILABLE:
            try:
                model_path = _ensure_model_file()
                if os.path.exists(model_path):
                    base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
                    options = mp_vision.HandLandmarkerOptions(
                        base_options=base_options,
                        num_hands=1,
                        min_hand_detection_confidence=self.min_confidence,
                        min_hand_presence_confidence=self.min_confidence,
                        min_tracking_confidence=self.min_confidence
                    )
                    self.landmarker = mp_vision.HandLandmarker.create_from_options(options)
                    self.backend = "tasks"
                    print("[GestureController] Initialized MediaPipe HandLandmarker (Tasks API).")
            except Exception as e:
                print(f"[GestureController] Tasks API init notice: {e}")

        if not self.landmarker and hasattr(mp, 'solutions') and hasattr(mp.solutions, 'hands'):
            try:
                self.landmarker = mp.solutions.hands.Hands(
                    static_image_mode=False,
                    max_num_hands=1,
                    min_detection_confidence=self.min_confidence,
                    min_tracking_confidence=self.min_confidence
                )
                self.backend = "solutions"
                print("[GestureController] Initialized MediaPipe Hands (Solutions API).")
            except Exception as e:
                print(f"[GestureController] Solutions API init error: {e}")

        if not self.landmarker:
            raise RuntimeError("Failed to initialize MediaPipe Hands landmarker.")

    def detect(self, rgb_frame: np.ndarray) -> Optional[List[Tuple[float, float, float]]]:
        """
        Detects 21 hand landmarks from an RGB frame.
        Returns a list of 21 (x, y, z) normalized coordinates or None if no hand detected.
        """
        if self.backend == "tasks":
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = self.landmarker.detect(mp_image)
            if result.hand_landmarks and len(result.hand_landmarks) > 0:
                hand = result.hand_landmarks[0]
                return [(lm.x, lm.y, lm.z) for lm in hand]
            return None
        elif self.backend == "solutions":
            result = self.landmarker.process(rgb_frame)
            if result.multi_hand_landmarks:
                hand = result.multi_hand_landmarks[0]
                return [(lm.x, lm.y, lm.z) for lm in hand.landmark]
            return None
        return None

    def close(self):
        """Releases MediaPipe landmarker resources."""
        if self.landmarker:
            try:
                self.landmarker.close()
            except Exception:
                pass
            self.landmarker = None


class GestureRecognizer:
    """
    Analyzes 21 MediaPipe hand landmarks to classify gestures:
    - POINTING UP / DOWN / LEFT / RIGHT
    - CLOSED FIST (PAUSE)
    - OPEN PALM (RESUME)
    Includes temporal smoothing, stability verification, and debounce cooldowns.
    """

    def __init__(self, stability_frames: int = 2, cooldown_seconds: float = 0.10):
        self.stability_frames = stability_frames
        self.cooldown_seconds = cooldown_seconds

        self.last_committed_time = 0.0
        self.recent_raw_gestures: List[GestureType] = []
        self.active_gesture = GestureType.NONE
        self.last_confirmed_direction: Optional[GestureType] = None

    @staticmethod
    def _dist_sq(p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> float:
        """Computes squared 2D Euclidean distance between two landmarks."""
        return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2

    @staticmethod
    def _is_finger_extended(tip: Tuple[float, float, float],
                            pip: Tuple[float, float, float],
                            mcp: Tuple[float, float, float]) -> bool:
        """
        Determines if a finger is extended using both joint alignment cosine
        and relative tip-to-MCP vs pip-to-MCP distance.
        Works across all hand orientations and avoids 2D projection foreshortening artifacts.
        """
        v1_x, v1_y = pip[0] - mcp[0], pip[1] - mcp[1]
        v2_x, v2_y = tip[0] - pip[0], tip[1] - pip[1]
        n1 = math.hypot(v1_x, v1_y)
        n2 = math.hypot(v2_x, v2_y)
        if n1 < 1e-4 or n2 < 1e-4:
            return False

        cos_alignment = (v1_x * v2_x + v1_y * v2_y) / (n1 * n2)
        d_tip_mcp = math.hypot(tip[0] - mcp[0], tip[1] - mcp[1])
        d_pip_mcp = math.hypot(pip[0] - mcp[0], pip[1] - mcp[1])

        # Extended when segments align (cos > 0.35) and tip extends past PIP
        return (cos_alignment > 0.35) and (d_tip_mcp > d_pip_mcp * 1.1)

    def classify_raw(self, landmarks: List[Tuple[float, float, float]]) -> Tuple[GestureType, Dict[str, Any]]:
        """
        Interprets the hand pose using landmark vectors and relative finger extensions.
        landmarks: list of 21 (x, y, z) normalized points in mirrored camera coordinates.
        """
        wrist = landmarks[0]

        # Check finger extension relative to their joint alignments
        is_index_extended = self._is_finger_extended(landmarks[8], landmarks[6], landmarks[5])
        is_middle_extended = self._is_finger_extended(landmarks[12], landmarks[10], landmarks[9])
        is_ring_extended = self._is_finger_extended(landmarks[16], landmarks[14], landmarks[13])
        is_pinky_extended = self._is_finger_extended(landmarks[20], landmarks[18], landmarks[17])

        # Span lengths from MCP to TIP for relative dominance check
        d_index_span = math.hypot(landmarks[8][0] - landmarks[5][0], landmarks[8][1] - landmarks[5][1])
        d_middle_span = math.hypot(landmarks[12][0] - landmarks[9][0], landmarks[12][1] - landmarks[9][1])
        d_ring_span = math.hypot(landmarks[16][0] - landmarks[13][0], landmarks[16][1] - landmarks[13][1])

        # 1. Closed Fist: all 4 fingers curled inward
        if not is_index_extended and not is_middle_extended and not is_ring_extended and not is_pinky_extended:
            return GestureType.PAUSE, {
                "desc": "CLOSED FIST",
                "extended": [False, False, False, False]
            }

        # 2. Open Palm: all 4 fingers extended
        if is_index_extended and is_middle_extended and is_ring_extended and is_pinky_extended:
            return GestureType.RESUME, {
                "desc": "OPEN PALM",
                "extended": [True, True, True, True]
            }

        # 3. Pointing Index Finger:
        # Index is extended, and other fingers are either classified curled or significantly shorter
        is_pointing_pose = is_index_extended and (
            (not is_middle_extended and not is_ring_extended and not is_pinky_extended) or
            (d_index_span > d_middle_span * 1.25 and d_index_span > d_ring_span * 1.25)
        )

        if is_pointing_pose:
            mcp = landmarks[5]
            tip = landmarks[8]

            # Vector from MCP to TIP in mirrored screen coordinates
            # Note: Screen Y increases downward, so -dy points UP
            dx = tip[0] - mcp[0]
            dy = tip[1] - mcp[1]

            magnitude = math.hypot(dx, dy)
            if magnitude < 0.04:
                # Foreshortened / ambiguous pointing direction
                return GestureType.NONE, {"desc": "POINTING (AMBIGUOUS)"}

            # Angle in Cartesian degrees (-180 to 180) where 0 is Right, 90 is Up
            angle = math.degrees(math.atan2(-dy, dx))

            # Cones: generous 100-degree cone for UP [40°, 140°] for natural hand ergonomics
            if 40.0 <= angle < 140.0:
                direction = GestureType.UP
            elif -140.0 <= angle < -40.0:
                direction = GestureType.DOWN
            elif -40.0 <= angle < 40.0:
                direction = GestureType.RIGHT
            else:
                direction = GestureType.LEFT

            return direction, {
                "desc": f"POINTING {direction.value}",
                "angle": angle,
                "magnitude": magnitude
            }

        return GestureType.NONE, {"desc": "NEUTRAL / UNKNOWN"}

    def update(self, landmarks: Optional[List[Tuple[float, float, float]]]) -> Tuple[GestureType, Dict[str, Any]]:
        """
        Updates recognizer with frame landmarks, applies stability filtering and debouncing.
        Returns (committed_gesture, debug_info).
        """
        if landmarks is None:
            self.recent_raw_gestures.clear()
            return GestureType.NONE, {"desc": "NO HAND DETECTED", "raw": GestureType.NONE}

        raw_gesture, info = self.classify_raw(landmarks)
        info["raw"] = raw_gesture

        self.recent_raw_gestures.append(raw_gesture)
        if len(self.recent_raw_gestures) > self.stability_frames:
            self.recent_raw_gestures.pop(0)

        # Require stability: all recent frames must agree on the same gesture
        now = time.time()
        if len(self.recent_raw_gestures) >= self.stability_frames:
            first = self.recent_raw_gestures[0]
            if first != GestureType.NONE and all(g == first for g in self.recent_raw_gestures):
                # Apply cooldown
                if (now - self.last_committed_time) >= self.cooldown_seconds:
                    self.active_gesture = first
                    self.last_committed_time = now
                    if first in (GestureType.UP, GestureType.DOWN, GestureType.LEFT, GestureType.RIGHT):
                        self.last_confirmed_direction = first
                    return first, info

        return GestureType.NONE, info


class GestureController:
    """
    Main Computer Vision module for Pac-Man.
    Runs non-blocking background capture and landmark detection.
    Renders an OpenCV preview window with HUD overlay and directional indicators.
    """

    def __init__(self, camera_index: int = 0, show_preview: bool = True):
        self.camera_index = camera_index
        self.show_preview = show_preview

        self.detector: Optional[HandDetector] = None
        self.recognizer = GestureRecognizer()

        self.cap: Optional[cv2.VideoCapture] = None
        self.worker_thread: Optional[threading.Thread] = None
        self.running = False
        self.lock = threading.Lock()

        # Thread-safe state queried by the game loop
        self.latest_command: Optional[GestureType] = None
        self.current_direction: GestureType = GestureType.LEFT
        self.hand_detected: bool = False
        self.detected_gesture_name: str = "INITIALIZING..."
        self.fps: float = 0.0
        self.error_message: Optional[str] = None
        self.is_paused: bool = False

    def start(self) -> bool:
        """Initializes camera and launches background worker thread."""
        if self.running:
            return True

        self.error_message = None
        try:
            # Initialize MediaPipe Detector
            self.detector = HandDetector()
        except Exception as e:
            self.error_message = f"MediaPipe initialization failed: {e}"
            print(f"[GestureController] {self.error_message}")
            return False

        # Attempt opening camera with CAP_DSHOW on Windows first
        backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if sys.platform == 'win32' else [cv2.CAP_ANY]
        opened = False
        for backend in backends:
            try:
                self.cap = cv2.VideoCapture(self.camera_index, backend)
                if self.cap.isOpened():
                    opened = True
                    break
            except Exception:
                continue

        if not opened or not self.cap.isOpened():
            # Try camera index 1 fallback
            for backend in backends:
                try:
                    self.cap = cv2.VideoCapture(1, backend)
                    if self.cap.isOpened():
                        opened = True
                        break
                except Exception:
                    continue

        if not opened or not self.cap.isOpened():
            self.error_message = "Webcam unavailable or already in use by another application."
            print(f"[GestureController] {self.error_message}")
            if self.detector:
                self.detector.close()
                self.detector = None
            return False

        # Configure camera capture properties for smooth low-latency feed
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True, name="GestureControllerWorker")
        self.worker_thread.start()
        print("[GestureController] Asynchronous camera worker started.")
        return True

    def stop(self) -> None:
        """Gracefully terminates background thread, releases webcam, and closes preview window."""
        self.running = False
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.5)
        self.worker_thread = None

        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        if self.detector:
            self.detector.close()
            self.detector = None

        if self.show_preview:
            try:
                cv2.destroyWindow("Pac-Man Gesture Controller")
            except Exception:
                pass

        print("[GestureController] Camera released and resources cleaned up.")

    def is_active(self) -> bool:
        """Returns True if the controller is actively running."""
        return self.running and (self.worker_thread is not None and self.worker_thread.is_alive())

    def get_command(self) -> Optional[GestureType]:
        """
        Thread-safe retrieval of latest confirmed command.
        Non-blocking: called directly inside the Pygame event loop.
        """
        with self.lock:
            cmd = self.latest_command
            self.latest_command = None
            return cmd

    def get_status(self) -> Dict[str, Any]:
        """Thread-safe retrieval of telemetry and status."""
        with self.lock:
            return {
                "active": self.is_active(),
                "hand_detected": self.hand_detected,
                "current_direction": self.current_direction.value,
                "gesture_name": self.detected_gesture_name,
                "fps": self.fps,
                "paused": self.is_paused,
                "error": self.error_message
            }

    def _worker_loop(self) -> None:
        """Continuous background capture and processing loop."""
        fps_counter = 0
        fps_timer = time.time()
        current_fps = 0.0

        window_name = "Pac-Man Gesture Controller"
        window_created = False

        while self.running:
            if not self.cap or not self.cap.isOpened():
                with self.lock:
                    self.error_message = "Camera disconnected during gameplay."
                break

            ret, frame = self.cap.read()
            if not ret or frame is None:
                # Brief sleep to avoid CPU spinning if frame drop occurs
                time.sleep(0.01)
                continue

            # Mirror frame horizontally so controls match user's perspective
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process hand landmarks
            landmarks = None
            try:
                if self.detector:
                    landmarks = self.detector.detect(rgb_frame)
            except Exception as e:
                print(f"[GestureController] Detection warning: {e}")

            # Recognize gesture
            gesture, info = self.recognizer.update(landmarks)

            # Update shared thread-safe state
            with self.lock:
                self.hand_detected = (landmarks is not None)
                self.detected_gesture_name = info.get("desc", "NONE")
                if gesture in (GestureType.UP, GestureType.DOWN, GestureType.LEFT, GestureType.RIGHT):
                    self.latest_command = gesture
                    self.current_direction = gesture
                elif gesture == GestureType.PAUSE:
                    self.latest_command = GestureType.PAUSE
                    self.is_paused = True
                elif gesture == GestureType.RESUME:
                    self.latest_command = GestureType.RESUME
                    self.is_paused = False

            # Calculate FPS
            fps_counter += 1
            now = time.time()
            if now - fps_timer >= 1.0:
                current_fps = fps_counter / (now - fps_timer)
                fps_counter = 0
                fps_timer = now
                with self.lock:
                    self.fps = round(current_fps, 1)

            # Render OpenCV Preview Window if enabled
            if self.show_preview:
                self._draw_overlay(frame, landmarks, info, current_fps)
                if not window_created:
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(window_name, 480, 360)
                    window_created = True
                cv2.imshow(window_name, frame)
                # Keep OpenCV event loop responsive (1ms)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    # Allow closing preview with Q
                    self.show_preview = False
                    cv2.destroyWindow(window_name)

        if self.show_preview:
            try:
                cv2.destroyWindow(window_name)
            except Exception:
                pass

    def _draw_overlay(self, frame: np.ndarray, landmarks: Optional[List[Tuple[float, float, float]]],
                      info: Dict[str, Any], fps: float) -> None:
        """Renders hand skeleton, directional vectors, status banners, and instructions."""
        h, w, _ = frame.shape

        # 1. Draw Skeleton Connections and Landmarks
        if landmarks:
            # Draw connection lines
            for idx1, idx2 in HAND_CONNECTIONS:
                pt1 = (int(landmarks[idx1][0] * w), int(landmarks[idx1][1] * h))
                pt2 = (int(landmarks[idx2][0] * w), int(landmarks[idx2][1] * h))
                cv2.line(frame, pt1, pt2, (0, 255, 128), 2, cv2.LINE_AA)

            # Draw joints
            for i, lm in enumerate(landmarks):
                cx, cy = int(lm[0] * w), int(lm[1] * h)
                # Color code key landmarks
                if i == 8:    # Index tip
                    cv2.circle(frame, (cx, cy), 9, (0, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(frame, (cx, cy), 11, (0, 0, 255), 2, cv2.LINE_AA)
                elif i in (4, 12, 16, 20):  # Other tips
                    cv2.circle(frame, (cx, cy), 6, (255, 100, 0), -1, cv2.LINE_AA)
                elif i == 0:  # Wrist
                    cv2.circle(frame, (cx, cy), 7, (255, 0, 255), -1, cv2.LINE_AA)
                else:
                    cv2.circle(frame, (cx, cy), 4, (0, 200, 255), -1, cv2.LINE_AA)

            # Draw directional pointer arrow if pointing
            if "angle" in info:
                mcp = (int(landmarks[5][0] * w), int(landmarks[5][1] * h))
                tip = (int(landmarks[8][0] * w), int(landmarks[8][1] * h))
                # Extended arrow along finger vector
                arrow_end = (tip[0] + (tip[0] - mcp[0]) // 2, tip[1] + (tip[1] - mcp[1]) // 2)
                cv2.arrowedLine(frame, mcp, arrow_end, (0, 255, 255), 3, tipLength=0.3)

        # 2. Semi-transparent Top Status Banner
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 85), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Status & Gesture Texts
        status_color = (0, 255, 0) if landmarks else (0, 0, 255)
        status_text = "HAND DETECTED" if landmarks else "NO HAND DETECTED"
        cv2.putText(frame, f"STATUS: {status_text}", (15, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2, cv2.LINE_AA)

        gesture_desc = info.get("desc", "SEARCHING...")
        cv2.putText(frame, f"GESTURE: {gesture_desc}", (15, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)

        with self.lock:
            cur_dir = self.current_direction.value
            paused = self.is_paused

        state_str = "PAUSED" if paused else f"MOVING {cur_dir}"
        cv2.putText(frame, f"PAC-MAN: {state_str}", (15, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 100), 1, cv2.LINE_AA)

        # Top Right FPS & Mode Badge
        cv2.putText(frame, f"FPS: {fps:.0f}", (w - 110, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2, cv2.LINE_AA)
        cv2.putText(frame, "GESTURE MODE", (w - 170, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 200), 1, cv2.LINE_AA)

        # 3. Semi-transparent Bottom Controls Reference
        cv2.rectangle(overlay, (0, h - 35), (w, h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        help_text = "Point Index: UP/DOWN/LEFT/RIGHT | Fist: PAUSE | Palm: RESUME"
        cv2.putText(frame, help_text, (15, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

"""
vision/gesture_controller.py
Coordinates camera frame capture, MediaPipe hand detection,
and gesture recognition in an asynchronous background thread.
Provides thread-safe state access for the Pygame event loop.
"""

import sys
import threading
import time
from typing import Optional, Dict, Any, Tuple, List, Union

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import numpy as np
except ImportError:
    np = None

from vision.gesture_recognizer import GestureType, GestureRecognizer, HAND_CONNECTIONS
from vision.hand_detector import HandDetector, MediaPipeNotAvailableError, is_mediapipe_available
from vision.camera import Camera, is_cv2_available


class GestureController:
    """
    Main Computer Vision Controller for Pac-Man.
    Runs non-blocking background capture and landmark detection.
    Optionally displays an OpenCV preview window with HUD overlays.
    """

    def __init__(self,
                 camera_source: Union[int, str] = 0,
                 show_preview: bool = True,
                 width: int = 640,
                 height: int = 480):
        self.camera_source = camera_source
        self.show_preview = show_preview
        self.width = width
        self.height = height

        self.camera: Optional[Camera] = None
        self.detector: Optional[HandDetector] = None
        self.recognizer = GestureRecognizer()

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

    @classmethod
    def is_available(cls) -> bool:
        """Returns True if both OpenCV and MediaPipe backends are installed."""
        return is_cv2_available() and is_mediapipe_available()

    def start(self) -> bool:
        """Initializes detector and camera, then launches background worker thread."""
        if self.running:
            return True

        self.error_message = None

        # 1. Verify OpenCV
        if not is_cv2_available():
            self.error_message = "OpenCV (cv2) is not installed."
            print(f"[GestureController] {self.error_message}")
            return False

        # 2. Verify MediaPipe
        try:
            self.detector = HandDetector()
        except Exception as e:
            self.error_message = f"MediaPipe initialization failed: {e}"
            print(f"[GestureController] {self.error_message}")
            return False

        # 3. Open Camera
        try:
            self.camera = Camera(source=self.camera_source, width=self.width, height=self.height)
            if not self.camera.open():
                self.error_message = "Webcam or CSI camera unavailable or already in use."
                print(f"[GestureController] {self.error_message}")
                if self.detector:
                    self.detector.close()
                    self.detector = None
                return False
        except Exception as e:
            self.error_message = f"Camera initialization error: {e}"
            print(f"[GestureController] {self.error_message}")
            if self.detector:
                self.detector.close()
                self.detector = None
            return False

        # 4. Start Worker Thread
        self.running = True
        self.worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="GestureControllerWorker"
        )
        self.worker_thread.start()
        print("[GestureController] Asynchronous camera worker started.")
        return True

    def stop(self) -> None:
        """Gracefully terminates background thread, releases camera, and closes windows."""
        self.running = False
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.5)
        self.worker_thread = None

        if self.camera:
            try:
                self.camera.release()
            except Exception:
                pass
            self.camera = None

        if self.detector:
            try:
                self.detector.close()
            except Exception:
                pass
            self.detector = None

        if self.show_preview and cv2 is not None:
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
            if not self.camera or not self.camera.is_opened():
                with self.lock:
                    self.error_message = "Camera disconnected during gameplay."
                break

            ret, frame = self.camera.read()
            if not ret or frame is None:
                # Brief sleep to avoid CPU spinning if frame drop occurs
                time.sleep(0.01)
                continue

            # Mirror frame horizontally so controls match user's perspective
            frame = cv2.flip(frame, 1)

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
            if self.show_preview and cv2 is not None:
                self._draw_overlay(frame, landmarks, info, current_fps)
                if not window_created:
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(window_name, 480, 360)
                    window_created = True
                cv2.imshow(window_name, frame)
                # Keep OpenCV event loop responsive (1ms)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.show_preview = False
                    cv2.destroyWindow(window_name)

        if self.show_preview and cv2 is not None:
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


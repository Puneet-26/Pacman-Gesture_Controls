"""
gesture_controller.py
Top-level entry and backwards-compatible facade for PAC-MAN VISION.
Delegates to the modular vision package while ensuring safe, lazy imports.
Importing this file will NEVER crash even if OpenCV or MediaPipe is absent.
"""

from typing import Optional, Dict, Any, Union

from vision.gesture_recognizer import (
    GestureType,
    GestureRecognizer,
    HAND_CONNECTIONS
)

try:
    from vision.hand_detector import (
        HandDetector,
        MediaPipeNotAvailableError,
        is_mediapipe_available
    )
except ImportError:
    HandDetector = None
    MediaPipeNotAvailableError = Exception
    is_mediapipe_available = lambda: False

try:
    from vision.camera import (
        Camera,
        is_cv2_available,
        build_csi_gstreamer_pipeline
    )
except ImportError:
    Camera = None
    is_cv2_available = lambda: False
    build_csi_gstreamer_pipeline = None

try:
    from vision.gesture_controller import GestureController as _BaseGestureController
    VISION_CONTROLLER_AVAILABLE = True
except ImportError:
    VISION_CONTROLLER_AVAILABLE = False


class GestureController:
    """
    Backwards-compatible wrapper around vision.GestureController.
    Safely degrades to a stub if OpenCV or MediaPipe is missing,
    allowing Pac-Man to run in Keyboard Mode without errors.
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

        self._impl = None
        self.error_message: Optional[str] = None

        if self.is_available():
            try:
                self._impl = _BaseGestureController(
                    camera_source=self.camera_source,
                    show_preview=self.show_preview,
                    width=self.width,
                    height=self.height
                )
            except Exception as e:
                self.error_message = f"Failed to initialize vision controller: {e}"
        else:
            missing = []
            if not is_cv2_available():
                missing.append("OpenCV (cv2)")
            if not is_mediapipe_available() or (VISION_CONTROLLER_AVAILABLE and not _BaseGestureController.is_available()):
                missing.append("MediaPipe")
            if not missing:
                missing.append("Vision controller dependencies")
            self.error_message = f"Vision libraries unavailable: {', '.join(missing)}."

    @classmethod
    def is_available(cls) -> bool:
        """Returns True if vision dependencies are present."""
        if not VISION_CONTROLLER_AVAILABLE:
            return False
        return _BaseGestureController.is_available()

    def start(self) -> bool:
        """Starts gesture tracking."""
        if self._impl is None:
            print(f"[GestureController] Cannot start: {self.error_message}")
            return False
        started = self._impl.start()
        if not started:
            self.error_message = self._impl.error_message
        return started

    def stop(self) -> None:
        """Stops gesture tracking."""
        if self._impl:
            self._impl.stop()

    def is_active(self) -> bool:
        """Returns True if tracking is currently active."""
        return self._impl.is_active() if self._impl else False

    def get_command(self) -> Optional[GestureType]:
        """Returns latest gesture command."""
        return self._impl.get_command() if self._impl else None

    def get_status(self) -> Dict[str, Any]:
        """Returns telemetry status dictionary."""
        if self._impl:
            return self._impl.get_status()
        return {
            "active": False,
            "hand_detected": False,
            "current_direction": "NONE",
            "gesture_name": "UNAVAILABLE",
            "fps": 0.0,
            "paused": False,
            "error": self.error_message
        }


__all__ = [
    'GestureType',
    'GestureRecognizer',
    'HAND_CONNECTIONS',
    'HandDetector',
    'Camera',
    'GestureController',
    'is_cv2_available',
    'is_mediapipe_available',
]

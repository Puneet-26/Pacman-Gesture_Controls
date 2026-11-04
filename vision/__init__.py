"""
vision package for PAC-MAN VISION.
Modular computer vision and hand tracking components.
"""

from vision.gesture_recognizer import (
    GestureType,
    GestureRecognizer,
    HAND_CONNECTIONS
)

from vision.hand_detector import (
    HandDetector,
    MediaPipeNotAvailableError,
    is_mediapipe_available
)

from vision.camera import (
    Camera,
    is_cv2_available,
    build_csi_gstreamer_pipeline
)

from vision.gesture_controller import (
    GestureController
)

__all__ = [
    'GestureType',
    'GestureRecognizer',
    'HAND_CONNECTIONS',
    'HandDetector',
    'MediaPipeNotAvailableError',
    'is_mediapipe_available',
    'Camera',
    'is_cv2_available',
    'build_csi_gstreamer_pipeline',
    'GestureController',
]

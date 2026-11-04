"""
vision/gesture_recognizer.py
Pure-Python hand gesture classification and temporal smoothing.
No dependencies on OpenCV or MediaPipe.
Compatible with Python 3.6+ and all platforms (Windows, Linux, Jetson Nano).
"""

import math
import time
from enum import Enum
from typing import List, Tuple, Dict, Any, Optional


class GestureType(Enum):
    """Enumeration of recognized hand gestures and game actions."""
    NONE = "NONE"
    UP = "UP"
    DOWN = "DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    PAUSE = "PAUSE"
    RESUME = "RESUME"


# Hand skeleton connections for 21 MediaPipe landmarks
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


class GestureRecognizer:
    """
    Analyzes 21 hand landmarks to classify gestures:
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


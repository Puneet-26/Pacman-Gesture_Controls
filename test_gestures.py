"""
test_gestures.py
Standalone test script to verify webcam access, MediaPipe hand landmark detection,
and gesture recognition independently of the main Pac-Man game.

Usage:
  python test_gestures.py             # Live webcam testing with OpenCV preview
  python test_gestures.py --synthetic # Automated synthetic test without camera
"""

import sys
import time
from typing import List, Tuple

from gesture_controller import GestureController, GestureRecognizer, GestureType


def run_synthetic_tests():
    """Runs automated verification of gesture classification logic on landmark vectors."""
    print("=" * 60)
    print("RUNNING SYNTHETIC GESTURE RECOGNIZER TESTS")
    print("=" * 60)

    recognizer = GestureRecognizer(stability_frames=1, cooldown_seconds=0.0)

    # Base hand template (wrist at 0.5, 0.8)
    def make_hand(direction: str = "UP",
                  index_extended: bool = True,
                  middle_extended: bool = False,
                  ring_extended: bool = False,
                  pinky_extended: bool = False) -> List[Tuple[float, float, float]]:
        wrist = (0.50, 0.80, 0.0)
        landmarks = [wrist] * 21

        # Knuckles (MCPs)
        landmarks[2] = (0.42, 0.72, 0.0)  # thumb mcp
        landmarks[5] = (0.45, 0.60, 0.0)  # index mcp
        landmarks[9] = (0.50, 0.60, 0.0)  # middle mcp
        landmarks[13] = (0.55, 0.62, 0.0) # ring mcp
        landmarks[17] = (0.60, 0.64, 0.0) # pinky mcp

        # Index PIP and TIP according to pointing direction
        if index_extended:
            if direction == "UP":
                landmarks[6] = (0.45, 0.50, 0.0)
                landmarks[7] = (0.45, 0.42, 0.0)
                landmarks[8] = (0.45, 0.35, 0.0)
            elif direction == "DOWN":
                landmarks[6] = (0.45, 0.68, 0.0)
                landmarks[7] = (0.45, 0.75, 0.0)
                landmarks[8] = (0.45, 0.85, 0.0)
            elif direction == "RIGHT":
                landmarks[6] = (0.55, 0.60, 0.0)
                landmarks[7] = (0.63, 0.60, 0.0)
                landmarks[8] = (0.72, 0.60, 0.0)
            elif direction == "LEFT":
                landmarks[6] = (0.35, 0.60, 0.0)
                landmarks[7] = (0.27, 0.60, 0.0)
                landmarks[8] = (0.18, 0.60, 0.0)
        else:
            # Index curled
            landmarks[6] = (0.45, 0.55, 0.0)
            landmarks[7] = (0.45, 0.58, 0.0)
            landmarks[8] = (0.45, 0.60, 0.0)

        # Middle
        if middle_extended:
            landmarks[10] = (0.50, 0.48, 0.0)
            landmarks[11] = (0.50, 0.40, 0.0)
            landmarks[12] = (0.50, 0.32, 0.0)
        else:
            landmarks[10] = (0.50, 0.55, 0.0)
            landmarks[11] = (0.50, 0.58, 0.0)
            landmarks[12] = (0.50, 0.60, 0.0)

        # Ring
        if ring_extended:
            landmarks[14] = (0.55, 0.50, 0.0)
            landmarks[15] = (0.55, 0.42, 0.0)
            landmarks[16] = (0.55, 0.35, 0.0)
        else:
            landmarks[14] = (0.55, 0.56, 0.0)
            landmarks[15] = (0.55, 0.59, 0.0)
            landmarks[16] = (0.55, 0.62, 0.0)

        # Pinky
        if pinky_extended:
            landmarks[18] = (0.60, 0.52, 0.0)
            landmarks[19] = (0.60, 0.45, 0.0)
            landmarks[20] = (0.60, 0.38, 0.0)
        else:
            landmarks[18] = (0.60, 0.58, 0.0)
            landmarks[19] = (0.60, 0.61, 0.0)
            landmarks[20] = (0.60, 0.64, 0.0)

        # Thumb
        landmarks[3] = (0.38, 0.68, 0.0)
        landmarks[4] = (0.35, 0.65, 0.0)

        return landmarks

    test_cases = [
        ("Point UP", make_hand("UP"), GestureType.UP),
        ("Point DOWN", make_hand("DOWN"), GestureType.DOWN),
        ("Point RIGHT", make_hand("RIGHT"), GestureType.RIGHT),
        ("Point LEFT", make_hand("LEFT"), GestureType.LEFT),
        ("Closed Fist (PAUSE)", make_hand("UP", index_extended=False), GestureType.PAUSE),
        ("Open Palm (RESUME)", make_hand("UP", index_extended=True,
                                         middle_extended=True, ring_extended=True, pinky_extended=True), GestureType.RESUME),
    ]

    all_passed = True
    for name, landmarks, expected in test_cases:
        gesture, info = recognizer.update(landmarks)
        status = "PASS" if gesture == expected else "FAIL"
        if gesture != expected:
            all_passed = False
        print(f"[{status}] {name:20s} -> Detected: {gesture.value:8s} | Expected: {expected.value} | Info: {info.get('desc')}")

    print("-" * 60)
    if all_passed:
        print("ALL SYNTHETIC GESTURE RECOGNITION TESTS PASSED!")
    else:
        print("SOME TESTS FAILED.")
        sys.exit(1)


def run_live_test():
    """Runs interactive live webcam gesture testing with real-time feedback."""
    print("=" * 60)
    print("STARTING LIVE PAC-MAN WEBCAM GESTURE TESTING")
    print("=" * 60)
    print("Controls:")
    print("  - Point Index UP    : Move UP")
    print("  - Point Index DOWN  : Move DOWN")
    print("  - Point Index LEFT  : Move LEFT")
    print("  - Point Index RIGHT : Move RIGHT")
    print("  - Closed Fist       : PAUSE")
    print("  - Open Palm         : RESUME")
    print("  - Press 'q' in OpenCV window or Ctrl+C in terminal to exit")
    print("=" * 60)

    controller = GestureController(show_preview=True)
    if not controller.start():
        print(f"[ERROR] Failed to start gesture controller: {controller.error_message}")
        sys.exit(1)

    print("[INFO] Camera started. Look at the 'Pac-Man Gesture Controller' window.")

    last_gesture = None
    try:
        while controller.is_active():
            cmd = controller.get_command()
            status = controller.get_status()

            if cmd is not None:
                print(f"[EVENT] New Command Triggered: {cmd.value} (FPS: {status['fps']})")

            # Periodically report status if gesture changes
            cur_desc = status.get("gesture_name", "")
            if cur_desc != last_gesture and cur_desc != "INITIALIZING...":
                print(f"[STATUS] {cur_desc} | Hand Detected: {status['hand_detected']} | FPS: {status['fps']}")
                last_gesture = cur_desc

            time.sleep(0.03)
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        controller.stop()
        print("[INFO] Clean shutdown complete.")


if __name__ == "__main__":
    if "--synthetic" in sys.argv:
        run_synthetic_tests()
    else:
        run_live_test()

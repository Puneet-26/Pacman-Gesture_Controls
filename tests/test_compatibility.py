"""
tests/test_compatibility.py
Comprehensive automated test suite verifying Python 3.6 & Jetson Nano compatibility,
modular vision subsystem, error handling, resource cleanup, and game engine integration.

Usage:
  python tests/test_compatibility.py
"""

import ast
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure repository root is working directory and on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(str(REPO_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Set dummy audio and video drivers to allow headless tests
os.environ['SDL_AUDIODRIVER'] = 'dummy'
os.environ['SDL_VIDEODRIVER'] = 'dummy'

from model.direction import Direction
from vision.gesture_recognizer import GestureType, GestureRecognizer, HAND_CONNECTIONS


class TestPython36SyntaxAudit(unittest.TestCase):
    """Test 7: Python 3.6 AST syntax audit across all project source files."""

    def test_all_python_files_are_python_36_compatible(self):
        """Walk all .py files and verify no Python 3.8+ or PEP 585 features exist."""
        forbidden_features = []

        for py_path in REPO_ROOT.rglob("*.py"):
            # Skip virtual environments or caches if any exist
            if any(part.startswith('.') or part in ('venv', 'env', '__pycache__') for part in py_path.parts):
                continue

            with open(str(py_path), 'r', encoding='utf-8') as f:
                source = f.read()

            try:
                tree = ast.parse(source, filename=str(py_path))
            except SyntaxError as e:
                self.fail(f"SyntaxError in {py_path}: {e}")

            # AST walk to check for features not supported in Python 3.6
            for node in ast.walk(tree):
                # 1. Walrus operator (NamedExpr in 3.8+)
                if hasattr(ast, 'NamedExpr') and isinstance(node, ast.NamedExpr):
                    forbidden_features.append((str(py_path), node.lineno, "Walrus operator (:=) not in Python 3.6"))

                # 2. Match / Case (Match in 3.10+)
                if hasattr(ast, 'Match') and isinstance(node, ast.Match):
                    forbidden_features.append((str(py_path), node.lineno, "Pattern matching not in Python 3.6"))

                # 3. PEP 585 Subscripting of builtins (list[...], dict[...], tuple[...], etc.)
                if isinstance(node, ast.Subscript):
                    if isinstance(node.value, ast.Name):
                        if node.value.id in ('list', 'dict', 'tuple', 'set', 'frozenset', 'type'):
                            forbidden_features.append(
                                (str(py_path), node.lineno, f"PEP 585 subscript '{node.value.id}[...]' not supported in Python 3.6 (use typing module)")
                            )

        if forbidden_features:
            msg = "\n".join(f"{f}:{line} - {reason}" for f, line, reason in forbidden_features)
            self.fail(f"Found Python 3.6 incompatible AST nodes:\n{msg}")


class TestMissingMediaPipeHandling(unittest.TestCase):
    """Test 1: Game imports and starts without MediaPipe installed."""

    def test_import_without_mediapipe(self):
        """Simulate system without mediapipe and verify safe import & degradation."""
        with patch('vision.gesture_controller.is_mediapipe_available', return_value=False):
            # Importing gesture_controller should not raise an ImportError
            from gesture_controller import GestureController, GestureType
            self.assertEqual(GestureType.UP.value, "UP")
            self.assertEqual(GestureType.PAUSE.value, "PAUSE")

            # GestureController should gracefully report unavailable
            self.assertFalse(GestureController.is_available())
            gc = GestureController()
            self.assertFalse(gc.is_available())
            started = gc.start()
            self.assertFalse(started)
            self.assertIsNotNone(gc.error_message)
            self.assertIn("MediaPipe", gc.error_message)


class TestKeyboardModeOperation(unittest.TestCase):
    """Test 2: Keyboard mode operates without opening camera."""

    def test_keyboard_mode_initialization(self):
        """Verify Keyboard mode initializes without opening camera devices."""
        import pygame
        pygame.init()
        from pacman import Game

        # Initialize Game directly in KEYBOARD mode
        game = Game(initial_mode="keyboard")
        self.assertEqual(game.control_mode, "KEYBOARD")
        self.assertIsNone(game.gesture_controller)

        # Confirm direction dispatch functions without camera
        game.game_engine.set_direction(Direction.UP)
        self.assertEqual(game.game_engine.direction_command, Direction.UP)
        game.game_engine.set_direction(Direction.RIGHT)
        self.assertEqual(game.game_engine.direction_command, Direction.RIGHT)
        game.cleanup()


class TestGestureToDirectionMapping(unittest.TestCase):
    """Test 3: GestureType commands map 1:1 to game Direction enum."""

    def test_command_mapping(self):
        """Verify each GestureType correctly controls Pac-Man and game state."""
        import pygame
        pygame.init()
        from pacman import Game

        game = Game(initial_mode="keyboard")
        mock_gc = MagicMock()
        game.gesture_controller = mock_gc
        game.control_mode = "GESTURE"

        mappings = [
            (GestureType.UP, Direction.UP),
            (GestureType.DOWN, Direction.DOWN),
            (GestureType.LEFT, Direction.LEFT),
            (GestureType.RIGHT, Direction.RIGHT),
        ]

        for gesture, expected_direction in mappings:
            mock_gc.get_command.return_value = gesture
            mock_gc.error_message = None
            game.process_gesture_input()
            self.assertEqual(game.game_engine.direction_command, expected_direction)

        # Test PAUSE / RESUME
        mock_gc.get_command.return_value = GestureType.PAUSE
        game.process_gesture_input()
        self.assertTrue(game.game_engine.pause)

        mock_gc.get_command.return_value = GestureType.RESUME
        game.process_gesture_input()
        self.assertFalse(game.game_engine.pause)

        game.cleanup()


class TestHandDetectorOutputFormat(unittest.TestCase):
    """Test 4: Hand detector output format validation (21 landmarks, normalized coords)."""

    def test_detector_output_contract(self):
        """Verify HandDetector contract returns 21 normalized (x, y, z) coordinates."""
        from vision.hand_detector import BaseHandDetector

        class MockLandmarkDetector(BaseHandDetector):
            def detect(self, rgb_frame):
                return [(float(i) / 21.0, 0.5, 0.0) for i in range(21)]

            def get_backend_name(self):
                return "mock"

        detector = MockLandmarkDetector()
        landmarks = detector.detect(None)
        self.assertIsNotNone(landmarks)
        self.assertEqual(len(landmarks), 21)

        for idx, pt in enumerate(landmarks):
            self.assertEqual(len(pt), 3, f"Landmark {idx} must be a 3-tuple (x, y, z)")
            x, y, z = pt
            self.assertTrue(0.0 <= x <= 1.0, f"X coordinate {x} must be normalized [0, 1]")
            self.assertTrue(0.0 <= y <= 1.0, f"Y coordinate {y} must be normalized [0, 1]")

        detector.close()


class TestCameraFailureHandling(unittest.TestCase):
    """Test 5: Missing or failing camera handled gracefully."""

    def test_invalid_camera_fallback(self):
        """Verify GestureController and Pacman handle camera failure without crashing."""
        import pygame
        pygame.init()
        from pacman import Game

        game = Game(camera_source=9999, initial_mode=None)
        # Attempt to switch to gesture mode with failing camera
        with patch('vision.camera.Camera.open', return_value=False):
            success = game.set_control_mode("GESTURE")
            self.assertFalse(success)
            self.assertEqual(game.control_mode, "KEYBOARD")
            self.assertIsNone(game.gesture_controller)
            self.assertIn("Camera Error", game.status_message)
        game.cleanup()


class TestResourceCleanup(unittest.TestCase):
    """Test 6: Resource cleanup verification (camera.release(), detector.close())."""

    def test_proper_shutdown(self):
        """Verify stop() releases camera and detector resources properly."""
        from vision.gesture_controller import GestureController

        controller = GestureController(show_preview=False)
        mock_camera = MagicMock()
        mock_detector = MagicMock()

        controller.camera = mock_camera
        controller.detector = mock_detector
        controller.running = True

        controller.stop()

        self.assertFalse(controller.running)
        mock_camera.release.assert_called_once()
        mock_detector.close.assert_called_once()
        self.assertIsNone(controller.camera)
        self.assertIsNone(controller.detector)


if __name__ == '__main__':
    unittest.main()

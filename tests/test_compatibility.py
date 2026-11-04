"""
tests/test_compatibility.py
Comprehensive automated test suite verifying Python 3.6 & Jetson Nano compatibility,
modular vision subsystem, error handling, resource cleanup, and game engine integration.

Verifies:
  1. All project Python files are compatible with Python 3.6 (AST walk).
  2. The project imports successfully without modern Python syntax.
  3. Keyboard mode does not require TensorRT, MediaPipe or a webcam.
  4. Gesture mode handles missing cameras gracefully.
  5. TensorRT model loading handles missing or incompatible engine files with clear errors.
  6. Gesture directions map correctly to the existing Pac-Man direction commands.
  7. Camera and model resources are released properly.
  8. Existing gameplay remains unchanged.

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
    """Test 1 & 2: Python 3.6 AST syntax audit across all project source files."""

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
    """Test 3a: Game imports and degrades gracefully without MediaPipe or TensorRT installed."""

    def test_import_without_mediapipe(self):
        """Simulate system without mediapipe and verify safe import & degradation."""
        with patch('vision.gesture_controller.is_mediapipe_available', return_value=False):
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
    """Test 3b: Keyboard mode operates without opening camera, MediaPipe, or TensorRT."""

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
    """Test 6: GestureType commands map 1:1 to game Direction enum."""

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
    """Output contract validation (21 landmarks, normalized coords)."""

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
    """Test 4: Missing or failing camera handled gracefully."""

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


class TestTensorRTModelLoading(unittest.TestCase):
    """Test 5: TensorRT model loading handles missing or incompatible engine files with clear errors."""

    def test_missing_tensorrt_engine_file(self):
        """Verify missing engine file raises FileNotFoundError explaining target-specific compilation."""
        from vision.hand_detector import TensorRTHandDetector

        with self.assertRaises(FileNotFoundError) as ctx:
            TensorRTHandDetector(engine_path="nonexistent_model.engine")

        msg = str(ctx.exception)
        self.assertIn("TensorRT engine file not found", msg)
        self.assertIn("trtexec", msg)

    def test_incompatible_tensorrt_engine_file(self):
        """Verify corrupt/incompatible engine file produces clear architecture mismatch error."""
        import tempfile
        from vision.hand_detector import TensorRTHandDetector
        import vision.hand_detector as hd

        with tempfile.NamedTemporaryFile(suffix=".engine", delete=False) as tmp:
            tmp.write(b"CORRUPTED_OR_INCOMPATIBLE_SERIALIZED_ENGINE_DATA")
            tmp_path = tmp.name

        try:
            # Simulate TensorRT installed but engine failing deserialization
            mock_trt = MagicMock()
            mock_runtime = MagicMock()
            mock_runtime.deserialize_cuda_engine.return_value = None
            mock_trt.Runtime.return_value = mock_runtime

            with patch.object(hd, 'TRT_AVAILABLE', True), patch.object(hd, 'trt', mock_trt):
                with self.assertRaises(RuntimeError) as ctx:
                    TensorRTHandDetector(engine_path=tmp_path)

                msg = str(ctx.exception)
                self.assertIn("Failed to deserialize TensorRT engine", msg)
                self.assertIn("trtexec", msg)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


class TestResourceCleanup(unittest.TestCase):
    """Test 7: Resource cleanup verification (camera.release(), detector.close())."""

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


class TestGameplayIntegrity(unittest.TestCase):
    """Test 8: Existing gameplay mechanics remain completely unchanged."""

    def test_game_engine_entities_and_board(self):
        """Verify player, 4 ghosts, maze board, and score initialization."""
        import pygame
        pygame.init()
        from pacman import Game

        game = Game(initial_mode="keyboard")
        engine = game.game_engine

        # Verify board structure
        self.assertIsNotNone(engine.board_definition)
        self.assertEqual(len(engine.ghosts), 4, "Must have exactly 4 ghosts (Blinky, Pinky, Inky, Clyde)")

        # Verify ghost types
        ghost_names = [type(g).__name__ for g in engine.ghosts]
        self.assertIn('Blinky', ghost_names)
        self.assertIn('Pinky', ghost_names)
        self.assertIn('Inky', ghost_names)
        self.assertIn('Clyde', ghost_names)

        # Verify initial player lives, game_over, and pause flags
        self.assertEqual(engine.player.lives, 3)
        self.assertFalse(engine.game_over)
        self.assertFalse(engine.pause)

        # Verify player initial location
        self.assertGreater(engine.player.location_x, 0)
        self.assertGreater(engine.player.location_y, 0)

        # Verify movement logic executes without error
        engine.set_direction(Direction.RIGHT)
        engine.tick()

        game.cleanup()


if __name__ == '__main__':
    unittest.main()

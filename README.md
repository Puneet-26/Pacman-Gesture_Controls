# PAC-MAN VISION (Gesture & Keyboard Controls)

Classic Pac-Man arcade game rebuilt in Python with dual control options:
1. **Keyboard Mode**: Traditional arcade controls (Arrow keys or WASD). Fully functional without any camera or vision dependencies.
2. **Gesture Mode**: Real-time vision-based hand tracking using OpenCV and MediaPipe. Control Pac-Man by pointing your index finger, pausing with a closed fist, and resuming with an open palm.

Cross-platform compatible with **modern laptops** (Windows, macOS, Linux with Python 3.8 - 3.12+) and **NVIDIA Jetson Nano 4GB** (Python 3.6 on JetPack 4.x / L4T Ubuntu 18.04 LTS aarch64 on the **Tanna TechBiz EAGLE-101** carrier board).

---

## Hand Gesture Controls

| Gesture | Action |
| :--- | :--- |
| **Point Index UP** | Move Pac-Man UP |
| **Point Index DOWN** | Move Pac-Man DOWN |
| **Point Index LEFT** | Move Pac-Man LEFT |
| **Point Index RIGHT** | Move Pac-Man RIGHT |
| **Closed Fist** | PAUSE Game |
| **Open Palm** | RESUME Game |
| **Key 'M'** | Toggle between Keyboard & Gesture Modes anytime |
| **Key 'P' / ESC** | Toggle Pause |

---

## Quick Start: Modern Laptop (Python 3.8 - 3.12+)

### 1. Install Dependencies
```bash
pip install -r requirements-laptop.txt
```

### 2. Launch the Game
```bash
python pacman.py
```
Or start directly in Keyboard Mode:
```bash
python pacman.py --mode keyboard
```

---

## Quick Start: NVIDIA Jetson Nano (Python 3.6 / JetPack 4.x)

Detailed hardware and setup instructions are provided in [docs/JETSON_SETUP.md](docs/JETSON_SETUP.md).

### 1. System Packages (APT)
Do **not** install OpenCV via pip on Jetson Nano; use system OpenCV to retain hardware GStreamer and CUDA support:
```bash
sudo apt-get update
sudo apt-get install -y python3-pip python3-dev python3-opencv python3-pygame
```

### 2. Python Packages
```bash
pip3 install -r requirements-jetson.txt
```

### 3. MediaPipe for Python 3.6 aarch64
Install the precompiled MediaPipe 0.8.5 aarch64 wheel (see [docs/JETSON_SETUP.md](docs/JETSON_SETUP.md) for direct links and build options):
```bash
pip3 install mediapipe-0.8.5-cp36-cp36m-linux_aarch64.whl
```

### 4. Run on Jetson Nano
- **Keyboard Mode (Zero camera/MediaPipe dependencies):**
  ```bash
  python3 pacman.py --mode keyboard
  ```
- **Gesture Mode with USB Webcam:**
  ```bash
  python3 pacman.py --mode gesture --camera 0
  ```
- **Gesture Mode with MIPI CSI Camera (Tanna TechBiz EAGLE-101 CAM0):**
  ```bash
  python3 pacman.py --mode gesture --camera csi
  ```
- **Performance Optimized (Low CPU/GPU overhead):**
  ```bash
  python3 pacman.py --mode gesture --camera csi --no-preview --width 320 --height 240
  ```

---

## Testing & Verification

Run the automated test suites:
```bash
# 1. Synthetic Gesture Classification Tests (Runs without camera)
python test_gestures.py --synthetic

# 2. Jetson Nano & Python 3.6 Compatibility Suite (All 7 Tests)
python tests/test_compatibility.py
```

---

## Project Architecture

```
PacMan-Gesture_Controls/
├── assets/                     # Sprites, animations, and sound effects
├── docs/
│   └── JETSON_SETUP.md         # Complete Jetson Nano & EAGLE-101 carrier board guide
├── draw/
│   └── game_engine.py          # Pygame graphics rendering and arcade loop
├── levels/
│   └── level_content_initializer.py # Level and sprite loading (Python 3.6 / Pygame 1.9.6 compatible)
├── model/                      # Direction, entities (Pac-Man, Ghosts), maze geometry
├── tests/
│   └── test_compatibility.py   # Automated 7-point compatibility and safety test suite
├── vision/                     # Modular Computer Vision package
│   ├── __init__.py             # Public vision exports
│   ├── camera.py               # Multi-platform camera wrapper (USB V4L2/DSHOW & Jetson CSI GStreamer)
│   ├── hand_detector.py        # Isolated MediaPipe detector (Tasks API & Solutions API)
│   ├── gesture_recognizer.py   # Pure-Python gesture classifier (no cv2/mediapipe dependencies)
│   └── gesture_controller.py   # Threaded camera capture & HUD overlay coordinator
├── gesture_controller.py       # Backwards-compatible root facade with safe degradation
├── pacman.py                   # Main game entry point & mode selection menu
├── requirements-jetson.txt     # Jetson Nano pinned dependencies
├── requirements-laptop.txt     # Modern laptop dependencies
└── test_gestures.py            # Live webcam and synthetic gesture test runner
```

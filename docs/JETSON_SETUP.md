# NVIDIA Jetson Nano Setup Guide: PAC-MAN VISION

This guide provides step-by-step instructions for deploying and running **PAC-MAN VISION** on an **NVIDIA Jetson Nano 4GB** mounted on a **Tanna TechBiz EAGLE-101** carrier board running **Python 3.6** under **JetPack 4.6.x (Ubuntu 18.04 LTS aarch64)**.

---

## 1. Hardware Overview

### Target Specification
- **Compute Module:** NVIDIA Jetson Nano 4GB Developer Kit (B01 module, Quad-core ARM Cortex-A57 @ 1.43 GHz, 128-core Maxwell GPU, 4GB 64-bit LPDDR4).
- **Carrier Board:** Tanna TechBiz EAGLE-101.
  - Dual 2-lane MIPI CSI-2 camera connectors (CAM0 and CAM1).
  - USB 3.0 Type-A and USB 2.0 ports.
  - Full-size HDMI 2.0 and DisplayPort.
  - Gigabit Ethernet (RJ45).
  - Power: 5V/4A DC barrel jack (recommended for vision workloads).
- **Supported Cameras:**
  - Standard UVC USB Webcam (e.g., Logitech C270, C920).
  - MIPI-CSI2 Camera (Sony IMX219 / Raspberry Pi Camera Module v2).

---

## 2. Software Prerequisites & Constraints

- **Operating System:** NVIDIA Linux for Tegra (L4T R32.7.x) / JetPack 4.6+.
- **Python Version:** **Python 3.6.9** (Default Ubuntu 18.04 LTS system Python).
  > **STRICT CONSTRAINT:** Do **NOT** attempt to upgrade the system Python to Python 3.8+ using third-party PPAs (such as deadsnakes). Upgrading the system Python breaks core JetPack bindings (`jetson-gpio`, `tensorrt`, `graphsurgeon`, and hardware-accelerated `python3-opencv`).

---

## 3. Initial Jetson System Setup

### 3.1 Maximize Performance Mode
Before running vision workloads, lock the CPU and GPU clocks to maximum:
```bash
# Set 10W Max Performance Mode
sudo nvpmodel -m 0

# Maximize clock frequencies
sudo jetson_clocks
```

### 3.2 Enable Adequate Swap Space (Crucial)
The Jetson Nano has 4GB of shared CPU/GPU RAM. Compiling or running vision models without adequate swap may trigger the Linux OOM (Out Of Memory) killer. Create an 6GB swapfile:
```bash
# Check existing swap
free -h

# Create a 6GB swapfile if needed
sudo fallocate -l 6G /var/swapfile
sudo chmod 600 /var/swapfile
sudo mkswap /var/swapfile
sudo swapon /var/swapfile

# Make swap persistent across reboots
echo "/var/swapfile swap swap defaults 0 0" | sudo tee -a /etc/fstab
```

---

## 4. Install System Dependencies

Use the Ubuntu `apt` package manager for hardware-accelerated libraries.

> **CRITICAL RULE FOR OPENCV:**
> Do **NOT** run `pip3 install opencv-python`.
> Prebuilt pip wheels for ARM64 lack CUDA and GStreamer support, which breaks the `nvarguscamerasrc` pipeline for CSI cameras. Use the system OpenCV provided by NVIDIA.

```bash
sudo apt-get update
sudo apt-get install -y \
    python3-pip \
    python3-dev \
    python3-opencv \
    python3-numpy \
    python3-pygame \
    v4l-utils \
    libcanberra-gtk-module \
    libcanberra-gtk3-module \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad
```

Grant your user account permission to access hardware video devices:
```bash
sudo usermod -aG video $USER
```
*(Log out and log back in for group changes to take effect).*

---

## 5. Install Python Dependencies

Install the Python packages using `requirements-jetson.txt`:
```bash
pip3 install -r requirements-jetson.txt
```

This installs:
- `numpy==1.19.5` (the highest version supporting Python 3.6 on ARM64)
- `pygame==2.0.3` (or uses system `python3-pygame`)

---

## 6. MediaPipe on Jetson Nano (Python 3.6 aarch64)

Official Google MediaPipe PyPI wheels only support Python 3.8+ for x86_64. On Jetson Nano (Python 3.6 on aarch64), install MediaPipe 0.8.5.

### Option A: Install Pre-compiled Community Wheel (Recommended)
Pre-built wheels of **MediaPipe 0.8.5 for Python 3.6 aarch64** are available through the Jetson community:
```bash
# Download precompiled MediaPipe 0.8.5 wheel for Jetson Nano Python 3.6
wget https://github.com/K-Zheng/mediapipe-jetson/releases/download/v0.8.5/mediapipe-0.8.5-cp36-cp36m-linux_aarch64.whl

# Install the wheel
pip3 install mediapipe-0.8.5-cp36-cp36m-linux_aarch64.whl
```

### Option B: Build MediaPipe 0.8.5 from Source
If building from source on Jetson Nano:
1. Ensure Bazel 3.7.2 is installed (`bazel version`).
2. Toolchain requirements: GCC 7.5.0, Protobuf 3.14.0.
3. Clone MediaPipe:
   ```bash
   git clone https://github.com/google/mediapipe.git
   cd mediapipe
   git checkout v0.8.5
   ```
4. Build with Bazel:
   ```bash
   bazel build -c opt --copt="-DMESA_EGL_NO_X11_HEADERS" --copt="-DEGL_NO_X11" //mediapipe/examples/desktop/hand_tracking:hand_tracking_gpu
   python3 setup.py bdist_wheel
   pip3 install dist/*.whl
   ```

---

## 7. Camera Configuration

### 7.1 USB Webcams
When plugging in a standard USB webcam to the EAGLE-101 board:
```bash
# Check device enumeration
ls -l /dev/video*
v4l2-ctl --list-devices
```
Typically, the USB camera will be `/dev/video0`.

Run Pac-Man with USB camera:
```bash
python3 pacman.py --camera 0
```

### 7.2 MIPI-CSI Cameras (Sony IMX219 / RPi Camera v2)
Connect the ribbon cable to `CAM0` on the Tanna TechBiz EAGLE-101 board (ensure the blue backing faces the Ethernet jack).

Verify the CSI camera sensor is recognized:
```bash
ls -l /dev/video0
```

Test hardware capture using GStreamer:
```bash
gst-launch-1.0 nvarguscamerasrc sensor-id=0 ! \
  'video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1' ! \
  nvvidconv ! xvimagesink
```

Run Pac-Man with CSI camera:
```bash
python3 pacman.py --camera csi
# OR specify sensor ID 0:
python3 pacman.py --camera csi://0
```

---

## 8. Running the Application

### 8.1 Keyboard Mode (Standalone)
Can be run on any Jetson without camera or MediaPipe:
```bash
python3 pacman.py --mode keyboard
```

### 8.2 Gesture Mode (Full Vision Control)
With USB camera:
```bash
python3 pacman.py --mode gesture --camera 0
```

With MIPI CSI camera:
```bash
python3 pacman.py --mode gesture --camera csi
```

### 8.3 Jetson Performance Optimization Flags
To maximize FPS and conserve CPU/GPU memory on Jetson Nano:
```bash
# Disable OpenCV preview window and use 320x240 capture resolution
python3 pacman.py --mode gesture --camera csi --no-preview --width 320 --height 240
```

### 8.4 Standalone Vision Test
Verify hand tracking independently before running the full game:
```bash
# Automated synthetic test (runs without camera)
python3 test_gestures.py --synthetic

# Live camera gesture verification
python3 test_gestures.py
```

---

## 9. Troubleshooting & FAQ

| Problem | Cause | Solution |
| :--- | :--- | :--- |
| `Cannot identify camera: /dev/video0` | `nvargus-daemon` service stalled | Run `sudo systemctl restart nvargus-daemon` |
| `Error opening camera via GStreamer` | Pipeline string or resolution mismatch | Verify sensor with `gst-launch-1.0` or run with `--camera 0` |
| `ImportError: libGL.so.1: cannot open shared object file` | Missing OpenGL runtime libraries | Run `sudo apt-get install -y libgl1-mesa-glx` |
| `ALSA lib pcm.c: playback open error` | Pygame audio device mismatch | Set `export SDL_AUDIODRIVER=alsa` before running `python3 pacman.py` |
| Pygame crashes with `No available video device` | Running over headless SSH | Forward X11 with `ssh -X` or run locally with `export DISPLAY=:0` |
| Low frame rate (< 15 FPS) | Power throttling or desktop GUI overhead | Enable max clocks (`sudo jetson_clocks`) and run with `--no-preview` |

"""
vision/camera.py
Robust camera capture module supporting USB webcams (Windows/Linux)
and MIPI-CSI cameras on NVIDIA Jetson Nano via GStreamer (nvarguscamerasrc).
"""

import os
import sys
from typing import Optional, Tuple, Union

try:
    import cv2
except ImportError:
    cv2 = None


def is_cv2_available() -> bool:
    """Returns True if OpenCV is installed and importable."""
    return cv2 is not None


def is_jetson_platform() -> bool:
    """Heuristic to detect if running on NVIDIA Jetson (aarch64 / Linux / tegra)."""
    if sys.platform != 'linux':
        return False
    # Check for Jetson Tegra release or devicetree model
    if os.path.exists('/etc/nv_tegra_release'):
        return True
    try:
        with open('/proc/device-tree/model', 'r') as f:
            model = f.read().lower()
            if 'jetson' in model or 'tegra' in model:
                return True
    except Exception:
        pass
    return False


def build_csi_gstreamer_pipeline(sensor_id: int = 0,
                                 capture_width: int = 1280,
                                 capture_height: int = 720,
                                 display_width: int = 640,
                                 display_height: int = 480,
                                 framerate: int = 30,
                                 flip_method: int = 0) -> str:
    """
    Constructs GStreamer pipeline string for Jetson Nano hardware-accelerated CSI cameras
    using nvarguscamerasrc and hardware ISP on Tanna TechBiz EAGLE-101 / standard carrier boards.
    """
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, "
        f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=(string)BGR ! appsink drop=1"
    )


class Camera:
    """
    Encapsulates OpenCV video capture with multi-platform fallbacks:
      - Windows: DirectShow (CAP_DSHOW) -> CAP_ANY
      - Linux USB: V4L2 (CAP_V4L2) -> CAP_ANY
      - Jetson CSI: GStreamer nvarguscamerasrc (CAP_GSTREAMER)
    """

    def __init__(self,
                 source: Union[int, str] = 0,
                 width: int = 640,
                 height: int = 480,
                 fps: int = 30):
        if cv2 is None:
            raise RuntimeError("OpenCV (cv2) is not installed. Please install opencv-python or python3-opencv.")

        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_csi = False

        # Check environment override
        env_csi = os.environ.get("PACMAN_USE_CSI", "").lower() in ("1", "true", "yes")
        if env_csi or str(source).startswith("csi:") or str(source).startswith("csi://"):
            self.is_csi = True

    def open(self) -> bool:
        """Attempts opening video capture with appropriate backends."""
        if self.cap is not None and self.cap.isOpened():
            return True

        # 1. CSI Camera on Jetson Nano via GStreamer
        if self.is_csi:
            sensor_id = 0
            if str(self.source).startswith("csi://") or str(self.source).startswith("csi:"):
                try:
                    sensor_id = int(str(self.source).split(":")[-1].replace("/", ""))
                except ValueError:
                    sensor_id = 0
            elif isinstance(self.source, int):
                sensor_id = self.source

            pipeline = build_csi_gstreamer_pipeline(
                sensor_id=sensor_id,
                display_width=self.width,
                display_height=self.height,
                framerate=self.fps
            )
            print(f"[Camera] Opening CSI camera via GStreamer pipeline:\n  {pipeline}")
            try:
                self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
                if self.cap.isOpened():
                    print("[Camera] Successfully opened CSI camera.")
                    return True
            except Exception as e:
                print(f"[Camera] CSI GStreamer failed: {e}")

        # 2. Direct GStreamer string source
        if isinstance(self.source, str) and ("!" in self.source or "nvarguscamerasrc" in self.source):
            try:
                self.cap = cv2.VideoCapture(self.source, cv2.CAP_GSTREAMER)
                if self.cap.isOpened():
                    return True
            except Exception as e:
                print(f"[Camera] GStreamer string capture failed: {e}")

        # 3. Standard USB / V4L2 / DirectShow
        camera_indices = [self.source] if isinstance(self.source, int) else [0]
        if 0 not in camera_indices:
            camera_indices.append(0)
        if 1 not in camera_indices:
            camera_indices.append(1)

        if sys.platform == 'win32':
            backends = [cv2.CAP_DSHOW, cv2.CAP_ANY]
        elif sys.platform.startswith('linux'):
            backends = [cv2.CAP_V4L2, cv2.CAP_ANY]
        else:
            backends = [cv2.CAP_ANY]

        for idx in camera_indices:
            for backend in backends:
                try:
                    self.cap = cv2.VideoCapture(idx, backend)
                    if self.cap.isOpened():
                        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
                        print(f"[Camera] Successfully opened camera index {idx} with backend {backend}.")
                        return True
                except Exception:
                    continue

        # 4. Fallback: If on Jetson and standard V4L2 failed, try default CSI pipeline
        if is_jetson_platform() and not self.is_csi:
            print("[Camera] Jetson platform detected; attempting fallback to CSI pipeline...")
            pipeline = build_csi_gstreamer_pipeline(
                sensor_id=0,
                display_width=self.width,
                display_height=self.height,
                framerate=self.fps
            )
            try:
                self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
                if self.cap.isOpened():
                    self.is_csi = True
                    print("[Camera] Fallback to Jetson CSI camera succeeded.")
                    return True
            except Exception:
                pass

        print("[Camera] Failed to open any camera device.")
        return False

    def is_opened(self) -> bool:
        """Returns True if camera is actively opened."""
        return self.cap is not None and self.cap.isOpened()

    def read(self) -> Tuple[bool, Optional[object]]:
        """Reads next frame. Returns (success, frame)."""
        if not self.is_opened():
            return False, None
        return self.cap.read()

    def release(self) -> None:
        """Releases video capture hardware."""
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        print("[Camera] VideoCapture released.")


"""
Camera module for VisionGuard AI.
Handles video capture from a webcam or video file using OpenCV.
"""

import threading
import time
import cv2
from config import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, FPS

# Global lock to prevent Windows DirectShow / MediaFoundation concurrency crashes
_camera_lock = threading.Lock()


class CameraStream:
    """
    Thread-safe, resilient OpenCV CameraStream wrapper.
    Supports both hardware webcams (by device index) and recorded video files / RTSP streams (by path).
    Prevents Windows COM / DirectShow access violation crashes during Streamlit reruns.
    """

    _active_cap = None  # Singleton underlying capture handle

    def __init__(self, source=CAMERA_INDEX, width=FRAME_WIDTH, height=FRAME_HEIGHT, loop_video=True):
        self.source = source
        self.width = width
        self.height = height
        self.loop_video = loop_video
        self.is_running = False

        # Determine if source is video file or camera index
        self.is_file_source = isinstance(source, str) and not source.isdigit()

        # FPS calculation variables
        self.prev_time = 0
        self.fps = 0.0

    def start(self):
        """
        Thread-safely opens or reuses the camera hardware or video file.
        """
        with _camera_lock:
            # If an existing capture is already open, cleanly release it first
            if CameraStream._active_cap is not None:
                try:
                    CameraStream._active_cap.release()
                except Exception:
                    pass
                CameraStream._active_cap = None
                time.sleep(0.1)  # Brief pause for hardware reset

            if self.is_file_source:
                # Open video file directly
                cap = cv2.VideoCapture(self.source)
            else:
                cam_idx = int(self.source)
                # Attempt DirectShow first for fast Windows startup
                cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
                # Fallback to default backend if CAP_DSHOW fails
                if not cap.isOpened():
                    cap = cv2.VideoCapture(cam_idx)

            if not cap.isOpened():
                print(f"[ERROR] Could not open video source: '{self.source}'.")
                self.is_running = False
                return False

            # Set resolution & FPS (for webcams)
            if not self.is_file_source:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                cap.set(cv2.CAP_PROP_FPS, FPS)

            CameraStream._active_cap = cap
            self.is_running = True
            self.prev_time = time.time()
            print(f"[INFO] Video source '{self.source}' initialized successfully.")
            return True

    def read_frame(self):
        """
        Reads a single frame thread-safely with auto-loop support for video files.
        """
        with _camera_lock:
            if not self.is_running or CameraStream._active_cap is None:
                return False, None

            try:
                ret, frame = CameraStream._active_cap.read()
                # If video file reached the end and looping is enabled
                if not ret and self.is_file_source and self.loop_video:
                    CameraStream._active_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = CameraStream._active_cap.read()
            except Exception as e:
                print(f"[ERROR] Exception during cap.read(): {e}")
                return False, None

            if not ret or frame is None:
                return False, None

            # Calculate FPS
            current_time = time.time()
            time_diff = current_time - self.prev_time
            if time_diff > 0:
                self.fps = 1.0 / time_diff
            self.prev_time = current_time

            return True, frame

    def get_fps(self):
        """Returns the current measured FPS."""
        return round(self.fps, 1)

    def release(self):
        """
        Releases the camera hardware cleanly.
        """
        with _camera_lock:
            if CameraStream._active_cap is not None:
                try:
                    CameraStream._active_cap.release()
                except Exception:
                    pass
                CameraStream._active_cap = None
            self.is_running = False
            print("[INFO] Camera released cleanly.")


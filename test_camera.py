"""
Test script for Milestone 1: Webcam access using OpenCV.

Usage:
  - Interactive test (opens live window, press 'q' to quit):
      python test_camera.py
  - Automated/Headless test (captures a few frames, saves a snapshot, exits):
      python test_camera.py --verify
"""

import sys
import cv2
from pathlib import Path
from src.camera import CameraStream
from config import DATA_DIR


def run_headless_verification():
    """Captures 5 frames to verify webcam functionality without opening a GUI window."""
    print("[TEST] Running automated camera verification...")
    camera = CameraStream()

    if not camera.start():
        print("[FAIL] Could not access the webcam.")
        return False

    captured = 0
    test_snapshot_path = DATA_DIR / "test_frame.jpg"

    for i in range(10):
        ret, frame = camera.read_frame()
        if ret and frame is not None:
            captured += 1
            if captured == 5:
                # Save snapshot to verify image quality
                cv2.imwrite(str(test_snapshot_path), frame)
                print(f"[SUCCESS] Captured frame with shape {frame.shape} at {camera.get_fps()} FPS.")
                print(f"[INFO] Saved test snapshot to: {test_snapshot_path}")
                break

    camera.release()

    if captured > 0:
        print("[PASS] Milestone 1 verification successful!")
        return True
    else:
        print("[FAIL] Camera opened, but failed to capture valid frames.")
        return False


def run_interactive_preview():
    """Opens a live OpenCV window for user visual testing. Press 'q' to exit."""
    print("[TEST] Starting live camera feed. Press 'q' in the video window to quit.")
    camera = CameraStream()

    if not camera.start():
        print("[ERROR] Camera failed to start.")
        return

    try:
        while True:
            ret, frame = camera.read_frame()
            if not ret or frame is None:
                print("[WARNING] Frame capture error.")
                break

            # Overlay FPS counter on the video frame
            fps_text = f"VisionGuard AI | FPS: {camera.get_fps()}"
            cv2.putText(
                frame,
                fps_text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("VisionGuard AI - Webcam Test (Press 'q' to quit)", frame)

            # Wait for 'q' key to quit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[INFO] User pressed 'q'. Exiting...")
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    if "--verify" in sys.argv:
        success = run_headless_verification()
        sys.exit(0 if success else 1)
    else:
        run_interactive_preview()


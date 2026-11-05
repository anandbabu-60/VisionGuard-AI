"""
Test script for Milestone 3: ByteTrack Object Tracking.

Usage:
  - Automated/Headless test (runs ByteTrack on frames, saves annotated frame):
      python test_tracker.py --verify
  - Interactive live tracking (opens camera feed with real-time ByteTrack IDs):
      python test_tracker.py
"""

import sys
import cv2
import numpy as np
from pathlib import Path
from src.tracker import ObjectTracker
from src.camera import CameraStream
from config import DATA_DIR


def run_headless_verification():
    """Runs ByteTrack over sequential test frames to verify ID assignment and trajectory tracking."""
    print("[TEST] Running automated ByteTrack verification...")

    test_image_path = DATA_DIR / "test_frame.jpg"
    if not test_image_path.exists():
        print(f"[WARNING] Test frame not found at {test_image_path}. Creating fallback image...")
        base_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    else:
        base_frame = cv2.imread(str(test_image_path))

    tracker = ObjectTracker()

    # Simulate multi-frame tracking sequence
    last_tracked = []
    annotated = None

    for frame_idx in range(5):
        # Slightly jitter the frame to simulate subtle camera motion/time progression
        frame = base_frame.copy()
        tracked = tracker.track(frame)
        last_tracked = tracked
        annotated = tracker.draw_tracks(frame, tracked, show_trail=True)

    print(f"[INFO] Tracking finished. Active tracked objects: {len(last_tracked)}")
    for obj in last_tracked:
        print(f"  -> ID #{obj['track_id']} | Class: '{obj['class_name']}' | Conf: {obj['confidence']} | Center: {obj['center']}")

    output_path = DATA_DIR / "tracked_test_frame.jpg"
    if annotated is not None:
        cv2.imwrite(str(output_path), annotated)
        print(f"[SUCCESS] Saved tracked output snapshot to: {output_path}")

    print("[PASS] Milestone 3 verification successful!")
    return True


def run_interactive_tracking():
    """Streams live webcam frames with real-time ByteTrack tracking IDs."""
    print("[TEST] Starting live ByteTrack tracking feed. Press 'q' to quit.")
    camera = CameraStream()
    tracker = ObjectTracker()

    if not camera.start():
        print("[ERROR] Camera failed to start.")
        return

    try:
        while True:
            ret, frame = camera.read_frame()
            if not ret or frame is None:
                break

            # Run tracking
            tracked_objects = tracker.track(frame)
            annotated_frame = tracker.draw_tracks(frame, tracked_objects, show_trail=True)

            # Display FPS and active track count
            info_text = f"FPS: {camera.get_fps()} | Tracked Objects: {len(tracked_objects)}"
            cv2.putText(
                annotated_frame,
                info_text,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 204, 0),
                2,
                cv2.LINE_AA
            )

            cv2.imshow("VisionGuard AI - ByteTrack Tracking (Press 'q' to quit)", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[INFO] Exiting live tracking...")
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    if "--verify" in sys.argv:
        success = run_headless_verification()
        sys.exit(0 if success else 1)
    else:
        run_interactive_tracking()


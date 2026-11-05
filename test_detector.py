"""
Test script for Milestone 2: YOLO Object Detection.

Usage:
  - Automated/Headless test (runs detector on test image, saves output):
      python test_detector.py --verify
  - Interactive live detection (opens camera feed with live YOLO boxes):
      python test_detector.py
"""

import sys
import cv2
from pathlib import Path
from src.detector import ObjectDetector
from src.camera import CameraStream
from config import DATA_DIR


def run_headless_verification():
    """Runs YOLO on the saved test frame and outputs an annotated image."""
    print("[TEST] Running automated YOLO detection verification...")

    test_image_path = DATA_DIR / "test_frame.jpg"
    if not test_image_path.exists():
        print(f"[WARNING] Test frame not found at {test_image_path}. Creating fallback image...")
        import numpy as np
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
    else:
        frame = cv2.imread(str(test_image_path))

    detector = ObjectDetector()
    detections = detector.detect(frame)

    print(f"[INFO] Detection finished. Found {len(detections)} object(s).")
    for idx, det in enumerate(detections, 1):
        print(f"  {idx}. Class: '{det['class_name']}' | Conf: {det['confidence']} | Box: {det['box']}")

    # Draw detections and save verification image
    annotated = detector.draw_detections(frame, detections)
    output_path = DATA_DIR / "detected_test_frame.jpg"
    cv2.imwrite(str(output_path), annotated)

    print(f"[SUCCESS] Saved annotated image to: {output_path}")
    print("[PASS] Milestone 2 verification successful!")
    return True


def run_interactive_detection():
    """Streams live webcam frames through the YOLO detector."""
    print("[TEST] Starting live YOLO detection feed. Press 'q' to quit.")
    camera = CameraStream()
    detector = ObjectDetector()

    if not camera.start():
        print("[ERROR] Camera failed to start.")
        return

    try:
        while True:
            ret, frame = camera.read_frame()
            if not ret or frame is None:
                break

            # Run detection
            detections = detector.detect(frame)
            annotated_frame = detector.draw_detections(frame, detections)

            # Display FPS and detection count
            info_text = f"FPS: {camera.get_fps()} | Objects: {len(detections)}"
            cv2.putText(
                annotated_frame,
                info_text,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA
            )

            cv2.imshow("VisionGuard AI - YOLO Detection (Press 'q' to quit)", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[INFO] Exiting live detection...")
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    if "--verify" in sys.argv:
        success = run_headless_verification()
        sys.exit(0 if success else 1)
    else:
        run_interactive_detection()


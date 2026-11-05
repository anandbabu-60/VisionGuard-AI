"""
Object Detection module for VisionGuard AI using YOLO.
Detects objects in real-time video frames and extracts bounding box coordinates,
class labels, and confidence scores.
"""

import cv2
from ultralytics import YOLO
from config import YOLO_MODEL_PATH, CONFIDENCE_THRESHOLD


class ObjectDetector:
    """
    Beginner-friendly wrapper around Ultralytics YOLO model.
    """

    def __init__(self, model_path=YOLO_MODEL_PATH, conf_threshold=CONFIDENCE_THRESHOLD):
        self.model_path = str(model_path)
        self.conf_threshold = conf_threshold
        print(f"[INFO] Loading YOLO model from: {self.model_path}...")
        self.model = YOLO(self.model_path)
        # Class names map (e.g., 0: 'person', 2: 'car', etc.)
        self.names = self.model.names
        print(f"[INFO] YOLO model loaded successfully with {len(self.names)} classes.")

    def detect(self, frame):
        """
        Runs object detection on a single frame.

        Args:
            frame (numpy.ndarray): Input BGR image from OpenCV.

        Returns:
            list[dict]: List of detected objects with keys:
                - 'class_id': integer ID
                - 'class_name': string label (e.g. 'person')
                - 'confidence': confidence score (0.0 to 1.0)
                - 'box': [x1, y1, x2, y2] pixel coordinates
        """
        if frame is None:
            return []

        # Run inference without spamming the console
        results = self.model(frame, conf=self.conf_threshold, verbose=False)[0]

        detections = []
        if results.boxes is not None:
            for box in results.boxes:
                coords = box.xyxy[0].cpu().numpy().astype(int).tolist()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                class_name = self.names.get(cls_id, f"class_{cls_id}")

                detections.append({
                    "class_id": cls_id,
                    "class_name": class_name,
                    "confidence": round(conf, 2),
                    "box": coords  # [x1, y1, x2, y2]
                })

        return detections

    def draw_detections(self, frame, detections):
        """
        Draws bounding boxes and labels on the frame.

        Args:
            frame (numpy.ndarray): Frame to draw onto.
            detections (list[dict]): Output from self.detect().

        Returns:
            numpy.ndarray: Annotated frame.
        """
        annotated_frame = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = det["box"]
            label = f"{det['class_name']} {int(det['confidence'] * 100)}%"

            # Distinctive styling: Green bounding box
            color = (0, 255, 128)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

            # Label background tag
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(
                annotated_frame,
                (x1, max(0, y1 - text_h - 8)),
                (x1 + text_w + 6, y1),
                color,
                -1
            )
            # Label text in dark color for contrast
            cv2.putText(
                annotated_frame,
                label,
                (x1 + 3, max(text_h + 2, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA
            )

        return annotated_frame


"""
Object Tracking module for VisionGuard AI using ByteTrack and YOLO.
Maintains persistent IDs and trajectories for detected objects across video frames.
"""

from collections import deque
import cv2
from ultralytics import YOLO
from config import YOLO_MODEL_PATH, CONFIDENCE_THRESHOLD, TRACKER_TYPE


# Pre-defined vibrant color palette for visual distinction across track IDs
TRACK_COLORS = [
    (255, 102, 0),   # Bright Orange
    (0, 204, 255),   # Sky Blue
    (51, 204, 51),   # Emerald Green
    (204, 0, 255),   # Purple
    (255, 204, 0),   # Golden Yellow
    (0, 102, 255),   # Dodger Blue
    (255, 51, 153),  # Hot Pink
    (0, 255, 153),   # Mint Green
]


class ObjectTracker:
    """
    Beginner-friendly wrapper for multi-object tracking with ByteTrack.
    """

    def __init__(
        self,
        model_path=YOLO_MODEL_PATH,
        conf_threshold=CONFIDENCE_THRESHOLD,
        tracker_type=TRACKER_TYPE,
        max_trail_length=30
    ):
        self.model_path = str(model_path)
        self.conf_threshold = conf_threshold
        self.tracker_type = tracker_type
        self.max_trail_length = max_trail_length

        print(f"[INFO] Initializing ObjectTracker with {self.tracker_type}...")
        self.model = YOLO(self.model_path)
        self.names = self.model.names

        # Stores trajectory points: track_id -> deque of (cx, cy)
        self.trails = {}

    def track(self, frame):
        """
        Runs detection and ByteTrack tracking on a single frame.

        Args:
            frame (numpy.ndarray): Input video frame.

        Returns:
            list[dict]: Tracked objects, each with:
                - 'track_id': int (unique persistent ID)
                - 'class_id': int
                - 'class_name': str
                - 'confidence': float
                - 'box': [x1, y1, x2, y2]
                - 'center': (cx, cy)
        """
        if frame is None:
            return []

        # Run YOLO with ByteTrack tracking
        results = self.model.track(
            source=frame,
            persist=True,
            tracker=self.tracker_type,
            conf=self.conf_threshold,
            verbose=False
        )[0]

        tracked_objects = []

        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results.boxes.id.cpu().numpy().astype(int)
            confs = results.boxes.conf.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy().astype(int)

            for box, track_id, conf, cls_id in zip(boxes, track_ids, confs, classes):
                x1, y1, x2, y2 = box.tolist()
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # Update trajectory trail
                if track_id not in self.trails:
                    self.trails[track_id] = deque(maxlen=self.max_trail_length)
                self.trails[track_id].append((cx, cy))

                class_name = self.names.get(cls_id, f"class_{cls_id}")

                tracked_objects.append({
                    "track_id": int(track_id),
                    "class_id": int(cls_id),
                    "class_name": class_name,
                    "confidence": round(float(conf), 2),
                    "box": [x1, y1, x2, y2],
                    "center": (cx, cy)
                })

        return tracked_objects

    def draw_tracks(self, frame, tracked_objects, show_trail=True):
        """
        Draws high-tech surveillance HUD bounding boxes, corner brackets,
        centroid target points, and motion trail paths.
        """
        annotated_frame = frame.copy()

        for obj in tracked_objects:
            track_id = obj["track_id"]
            x1, y1, x2, y2 = obj["box"]
            cls_name = obj["class_name"]
            conf = int(obj["confidence"] * 100)
            cx, cy = obj["center"]

            # Vibrant futuristic color palette
            color = TRACK_COLORS[track_id % len(TRACK_COLORS)]

            # 1. Bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

            # 2. Sleek Corner Brackets (surveillance HUD style)
            line_len = min(18, max(6, (x2 - x1) // 4), max(6, (y2 - y1) // 4))
            thickness = 3
            # Top-left corner
            cv2.line(annotated_frame, (x1, y1), (x1 + line_len, y1), color, thickness)
            cv2.line(annotated_frame, (x1, y1), (x1, y1 + line_len), color, thickness)
            # Top-right corner
            cv2.line(annotated_frame, (x2, y1), (x2 - line_len, y1), color, thickness)
            cv2.line(annotated_frame, (x2, y1), (x2, y1 + line_len), color, thickness)
            # Bottom-left corner
            cv2.line(annotated_frame, (x1, y2), (x1 + line_len, y2), color, thickness)
            cv2.line(annotated_frame, (x1, y2), (x1, y2 - line_len), color, thickness)
            # Bottom-right corner
            cv2.line(annotated_frame, (x2, y2), (x2 - line_len, y2), color, thickness)
            cv2.line(annotated_frame, (x2, y2), (x2, y2 - line_len), color, thickness)

            # 3. Centroid Target Point (red dot with white border as in surveillance HUDs)
            cv2.circle(annotated_frame, (cx, cy), 4, (0, 0, 255), -1)
            cv2.circle(annotated_frame, (cx, cy), 5, (255, 255, 255), 1)

            # 4. High-contrast Badge above bounding box (e.g., "ID-07 | person 89%")
            label = f"ID-{track_id:02d} | {cls_name} {conf}%"
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)

            badge_y1 = max(0, y1 - text_h - 10)
            badge_y2 = y1

            # Dark translucent backing with colored border
            cv2.rectangle(
                annotated_frame,
                (x1, badge_y1),
                (x1 + text_w + 10, badge_y2),
                (12, 16, 24),
                -1
            )
            cv2.rectangle(
                annotated_frame,
                (x1, badge_y1),
                (x1 + text_w + 10, badge_y2),
                color,
                1
            )
            # Badge text
            cv2.putText(
                annotated_frame,
                label,
                (x1 + 5, badge_y2 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                color,
                1,
                cv2.LINE_AA
            )

            # 5. Motion Trail with glowing gradient
            if show_trail and track_id in self.trails:
                points = list(self.trails[track_id])
                for i in range(1, len(points)):
                    if points[i - 1] is None or points[i] is None:
                        continue
                    trail_thickness = int(1.5 * (i / len(points)) + 1)
                    cv2.line(annotated_frame, points[i - 1], points[i], color, trail_thickness)

        return annotated_frame

    def reset(self):
        """Clears trail histories."""
        self.trails.clear()


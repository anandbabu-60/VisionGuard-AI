"""
Intelligent Surveillance Rules Engine for VisionGuard AI.
Implements:
  1. Virtual Tripwire (Bidirectional Entry & Exit Counting)
  2. Restricted Zone Perimeter Intrusion Detection
  3. Crowd Density & Overcrowding Monitoring
  4. Loitering & Dwell-Time Alerts
"""

import time
import cv2
import numpy as np


class SurveillanceEngine:
    """
    Evaluates real-time tracking coordinates against security rules.
    """

    def __init__(
        self,
        frame_width=640,
        frame_height=480,
        crowd_threshold=5,
        loitering_threshold_sec=10.0,
        tripwire_y_pct=0.58,
        zone_coords=None,
    ):
        self.width = frame_width
        self.height = frame_height
        self.crowd_threshold = crowd_threshold
        self.loitering_threshold_sec = loitering_threshold_sec

        # 1. Virtual Tripwire configuration: Horizontal line across mid-lower frame
        # Format: ((start_x, start_y), (end_x, end_y))
        mid_y = int(self.height * max(0.05, min(0.95, tripwire_y_pct)))
        self.tripwire_line = ((int(self.width * 0.05), mid_y), (int(self.width * 0.95), mid_y))

        # Tripwire statistics & state
        self.in_count = 0
        self.out_count = 0
        self._prev_centroids = {}  # {track_id: (cx, cy)}
        self._crossed_tracks = set()  # Tracks that already crossed in current trajectory

        # 2. Restricted Zone Polygon
        if zone_coords is not None and len(zone_coords) >= 3:
            self.restricted_zone = np.array(zone_coords, dtype=np.int32)
        else:
            # Default: Top-right quadrant
            zx1, zy1 = int(self.width * 0.60), int(self.height * 0.08)
            zx2, zy2 = int(self.width * 0.96), int(self.height * 0.52)
            self.restricted_zone = np.array([
                [zx1, zy1],
                [zx2, zy1],
                [zx2, zy2],
                [zx1, zy2]
            ], dtype=np.int32)

        # 3. Loitering tracking: {track_id: start_timestamp}
        self._track_first_seen = {}

    def set_tripwire_position(self, y_pct: float, start_x_pct: float = 0.05, end_x_pct: float = 0.95):
        """
        Dynamically reposition the virtual tripwire line based on percentages (0.0 - 1.0)
        of the frame dimensions.
        """
        y_pos = int(self.height * max(0.05, min(0.95, y_pct)))
        x1 = int(self.width * max(0.0, min(0.95, start_x_pct)))
        x2 = int(self.width * max(0.05, min(1.0, end_x_pct)))
        self.tripwire_line = ((x1, y_pos), (x2, y_pos))

    def set_restricted_zone(self, x_pct: float, y_pct: float, w_pct: float, h_pct: float):
        """
        Dynamically update the restricted rectangular perimeter based on percentages (0.0 - 1.0)
        of the frame dimensions.
        """
        x1 = int(self.width * max(0.0, min(0.95, x_pct)))
        y1 = int(self.height * max(0.0, min(0.95, y_pct)))
        x2 = int(self.width * max(0.05, min(1.0, x_pct + w_pct)))
        y2 = int(self.height * max(0.05, min(1.0, y_pct + h_pct)))
        self.restricted_zone = np.array([
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2]
        ], dtype=np.int32)

    def set_restricted_polygon(self, points: list):
        """
        Sets a custom polygon restricted zone from an arbitrary list of (x, y) coordinates.
        """
        if len(points) >= 3:
            self.restricted_zone = np.array(points, dtype=np.int32)

    # -------------------------------------------------------------------------
    # 1. Virtual Tripwire: Bidirectional Line Crossing
    # -------------------------------------------------------------------------
    def update_tripwire(self, tracked_objects):
        """
        Detects if tracked objects cross the virtual tripwire line.
        Downwards movement = IN (Entry)
        Upwards movement = OUT (Exit)

        Returns:
            list[dict]: Newly crossed events in this frame.
        """
        crossing_events = []
        line_y = self.tripwire_line[0][1]

        current_track_ids = set()

        for obj in tracked_objects:
            track_id = obj["track_id"]
            current_track_ids.add(track_id)
            cx, cy = obj["center"]
            cls_name = obj["class_name"]

            # Only count persons or vehicles for tripwires
            if cls_name not in ["person", "car", "bicycle", "motorcycle"]:
                continue

            prev_point = self._prev_centroids.get(track_id)

            if prev_point is not None:
                prev_x, prev_y = prev_point

                # Check if crossed the horizontal line between previous and current frame
                # Crossing downwards (Entry)
                if prev_y < line_y <= cy:
                    if (track_id, "IN") not in self._crossed_tracks:
                        self.in_count += 1
                        self._crossed_tracks.add((track_id, "IN"))
                        crossing_events.append({
                            "type": "ENTRY",
                            "track_id": track_id,
                            "class_name": cls_name,
                            "direction": "IN",
                            "net_occupancy": self.get_net_occupancy()
                        })

                # Crossing upwards (Exit)
                elif prev_y > line_y >= cy:
                    if (track_id, "OUT") not in self._crossed_tracks:
                        self.out_count += 1
                        self._crossed_tracks.add((track_id, "OUT"))
                        crossing_events.append({
                            "type": "EXIT",
                            "track_id": track_id,
                            "class_name": cls_name,
                            "direction": "OUT",
                            "net_occupancy": self.get_net_occupancy()
                        })

            # Update centroid for next frame
            self._prev_centroids[track_id] = (cx, cy)

        # Clean up centroids of tracks that left the frame
        disappeared = set(self._prev_centroids.keys()) - current_track_ids
        for tid in disappeared:
            self._prev_centroids.pop(tid, None)

        return crossing_events

    def get_net_occupancy(self):
        """Returns net people currently inside: max(0, in_count - out_count)."""
        return max(0, self.in_count - self.out_count)

    # -------------------------------------------------------------------------
    # 2. Restricted Zone Intrusion Detection
    # -------------------------------------------------------------------------
    def check_zone_intrusion(self, tracked_objects):
        """
        Determines whether any tracked objects are inside the restricted polygon.

        Returns:
            list[dict]: Intruding objects with their coordinates and IDs.
        """
        intruders = []

        for obj in tracked_objects:
            cx, cy = obj["center"]
            # cv2.pointPolygonTest returns positive if inside, 0 on edge, negative outside
            is_inside = cv2.pointPolygonTest(self.restricted_zone, (float(cx), float(cy)), False) >= 0

            if is_inside:
                intruders.append({
                    "track_id": obj["track_id"],
                    "class_name": obj["class_name"],
                    "confidence": obj["confidence"],
                    "box": obj["box"],
                    "center": (cx, cy)
                })

        return intruders

    # -------------------------------------------------------------------------
    # 3. Crowd Density Monitoring
    # -------------------------------------------------------------------------
    def check_crowd_density(self, tracked_objects):
        """
        Monitors number of humans in the camera view.

        Returns:
            dict: {is_crowded: bool, count: int, threshold: int}
        """
        person_count = sum(1 for obj in tracked_objects if obj["class_name"] == "person")
        is_crowded = person_count >= self.crowd_threshold

        return {
            "is_crowded": is_crowded,
            "person_count": person_count,
            "threshold": self.crowd_threshold
        }

    # -------------------------------------------------------------------------
    # 4. Loitering (Dwell Time) Detection
    # -------------------------------------------------------------------------
    def check_loitering(self, tracked_objects):
        """
        Identifies people lingering in the monitored scene longer than the allowed threshold.

        Returns:
            list[dict]: Loitering subjects with their dwell duration.
        """
        now = time.time()
        current_ids = set()
        loiterers = []

        for obj in tracked_objects:
            track_id = obj["track_id"]
            current_ids.add(track_id)

            if track_id not in self._track_first_seen:
                self._track_first_seen[track_id] = now

            dwell_time = now - self._track_first_seen[track_id]

            if dwell_time >= self.loitering_threshold_sec and obj["class_name"] == "person":
                loiterers.append({
                    "track_id": track_id,
                    "class_name": obj["class_name"],
                    "dwell_seconds": int(dwell_time),
                    "box": obj["box"]
                })

        # Purge disappeared tracks
        for tid in list(self._track_first_seen.keys()):
            if tid not in current_ids:
                del self._track_first_seen[tid]

        return loiterers

    # -------------------------------------------------------------------------
    # 5. PPE Safety Compliance Evaluation (Hard-Hats & Vests)
    # -------------------------------------------------------------------------
    def check_ppe_compliance(self, detections):
        """
        Evaluates worker detections for PPE compliance.
        Identifies missing hard-hats or missing high-visibility safety vests.

        Returns:
            dict: {
                "compliant_workers": int,
                "violations": list[dict],
                "compliance_rate_pct": float
            }
        """
        violations = []
        person_count = sum(1 for d in detections if "person" in str(d.get("class_name", "")).lower())

        for d in detections:
            cls_lower = str(d.get("class_name", "")).lower()
            if "no_hard_hat" in cls_lower or "no_helmet" in cls_lower:
                violations.append({
                    "type": "NO_HARD_HAT",
                    "description": "Worker missing required protective hard-hat / safety helmet",
                    "box": d.get("box"),
                    "confidence": d.get("confidence", 0.0)
                })
            elif "no_safety_vest" in cls_lower or "no_vest" in cls_lower:
                violations.append({
                    "type": "NO_SAFETY_VEST",
                    "description": "Worker missing high-visibility reflective safety vest",
                    "box": d.get("box"),
                    "confidence": d.get("confidence", 0.0)
                })

        compliant = max(0, person_count - len(violations))
        rate = 100.0 if person_count == 0 else round((compliant / max(1, person_count)) * 100.0, 1)

        return {
            "compliant_workers": compliant,
            "violations": violations,
            "compliance_rate_pct": rate
        }

    # -------------------------------------------------------------------------
    # Visual HUD Overlay Rendering
    # -------------------------------------------------------------------------
    def draw_hud(self, frame, has_intruders=False, is_crowded=False, ppe_violations=None):
        """
        Draws the visual tripwire line and restricted area polygon on the video frame.
        """
        annotated = frame.copy()

        # 1. Draw Tripwire Line (Cyan glowing line)
        pt1, pt2 = self.tripwire_line
        cv2.line(annotated, pt1, pt2, (255, 204, 0), 2, cv2.LINE_AA)

        # Tripwire counter tags
        text_y = max(18, pt1[1] - 8)
        cv2.putText(annotated, f"TRIPWIRE | IN: {self.in_count}  OUT: {self.out_count}  NET: {self.get_net_occupancy()}",
                    (pt1[0], text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 204, 0), 1, cv2.LINE_AA)

        # Small arrow icons indicating direction
        cv2.arrowedLine(annotated, (pt1[0] + 15, pt1[1] - 12), (pt1[0] + 15, pt1[1] + 12), (0, 255, 136), 1, tipLength=0.3)
        cv2.arrowedLine(annotated, (pt2[0] - 15, pt2[1] + 12), (pt2[0] - 15, pt2[1] - 12), (0, 102, 255), 1, tipLength=0.3)

        # 2. Draw Restricted Zone Polygon
        # If intrusion is active: flashing Red; if safe: translucent Cyan/Green
        zone_color = (0, 0, 255) if has_intruders else (0, 255, 200)

        # Create transparent overlay for restricted zone
        overlay = annotated.copy()
        cv2.fillPoly(overlay, [self.restricted_zone], zone_color)
        alpha = 0.20 if not has_intruders else 0.35
        cv2.addWeighted(overlay, alpha, annotated, 1 - alpha, 0, annotated)

        # Draw polygon border
        cv2.polylines(annotated, [self.restricted_zone], True, zone_color, 2, cv2.LINE_AA)

        # Zone Label
        label = "RESTRICTED ZONE [BREACH!]" if has_intruders else "RESTRICTED ZONE"
        top_left = self.restricted_zone[0]
        label_x = max(10, min(self.width - 220, top_left[0] + 8))
        label_y = max(25, min(self.height - 10, top_left[1] + 20))
        cv2.putText(annotated, label, (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, zone_color, 1, cv2.LINE_AA)

        # 3. Crowd warning banner if crowded
        if is_crowded:
            cv2.rectangle(annotated, (15, 45), (280, 75), (0, 0, 220), -1)
            cv2.putText(annotated, "WARNING: OVERCROWDING DETECTED", (22, 66),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # 4. PPE Safety Violation banner
        if ppe_violations:
            cv2.rectangle(annotated, (15, 82), (320, 112), (0, 140, 255), -1)
            cv2.putText(annotated, f"PPE ALERT: {len(ppe_violations)} SAFETY VIOLATION(S)", (22, 103),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        return annotated

    def reset_counts(self):
        """Resets tripwire counters and state."""
        self.in_count = 0
        self.out_count = 0
        self._prev_centroids.clear()
        self._crossed_tracks.clear()
        self._track_first_seen.clear()


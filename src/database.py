"""
MongoDB Database module for VisionGuard AI using PyMongo.
Manages connections, collections, automatic indexing, CRUD operations,
event logging, and analytics data retrieval.
"""

from datetime import datetime
from pathlib import Path
import cv2
import pandas as pd
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from config import (
    MONGODB_URI,
    MONGODB_DATABASE,
    DEFAULT_CAMERA_ID,
    DEFAULT_CAMERA_NAME,
    DEFAULT_CAMERA_LOCATION,
    SNAPSHOTS_DIR,
)


class DatabaseManager:
    """
    Singleton-style MongoDB Connection and Operations Manager for VisionGuard AI.
    """

    _instance = None
    _client = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(DatabaseManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, uri=MONGODB_URI, db_name=MONGODB_DATABASE, timeout_ms=3000):
        # Prevent re-initialization if already connected
        if hasattr(self, "_initialized") and self._initialized:
            return

        self.uri = uri
        self.db_name = db_name
        self.timeout_ms = timeout_ms
        self.client = None
        self.db = None
        self._connected = False

        # In-memory cache to throttle high-frequency tracking events
        # Structure: {(camera_id, track_id): last_logged_datetime}
        self._tracking_cache = {}

        self.connect()
        if self._connected:
            self.init_database()
            self._ensure_default_camera()
        self._initialized = True

    def connect(self):
        """Establishes connection to the MongoDB server with timeout handling."""
        try:
            self.client = MongoClient(self.uri, serverSelectionTimeoutMS=self.timeout_ms)
            # Ping server to confirm connection
            self.client.admin.command("ping")
            self.db = self.client[self.db_name]
            self._connected = True
            print(f"[INFO] Connected to MongoDB at '{self.uri}', database: '{self.db_name}'")
            return True
        except (ConnectionFailure, ServerSelectionTimeoutError) as err:
            self._connected = False
            self.client = None
            self.db = None
            print(f"[ERROR] Failed to connect to MongoDB ({self.uri}): {err}")
            return False

    def is_connected(self):
        """Returns True if the database connection is alive and healthy."""
        if not self._connected or self.client is None:
            return False
        try:
            self.client.admin.command("ping")
            return True
        except Exception:
            self._connected = False
            return False

    def init_database(self):
        """Creates required collections and builds optimized indexes automatically."""
        if not self._connected:
            print("[WARNING] Skipping init_database: MongoDB is not connected.")
            return

        # 1. cameras collection
        self.db.cameras.create_index([("camera_id", ASCENDING)], unique=True)

        # 2. detections collection
        self.db.detections.create_index([("camera_id", ASCENDING), ("timestamp", DESCENDING)])
        self.db.detections.create_index([("track_id", ASCENDING)])
        self.db.detections.create_index([("class_name", ASCENDING)])

        # 3. tracking_events collection
        self.db.tracking_events.create_index([("camera_id", ASCENDING), ("track_id", ASCENDING)])
        self.db.tracking_events.create_index([("last_seen", DESCENDING)])

        # 4. alerts collection
        self.db.alerts.create_index([("camera_id", ASCENDING), ("timestamp", DESCENDING)])
        self.db.alerts.create_index([("severity", ASCENDING)])

        # 5. snapshots collection
        self.db.snapshots.create_index([("camera_id", ASCENDING), ("timestamp", DESCENDING)])
        self.db.snapshots.create_index([("track_id", ASCENDING)])

        print("[INFO] MongoDB indexes verified successfully.")

    def _ensure_default_camera(self):
        """Registers the primary default camera if not already in the database."""
        self.register_camera(
            camera_id=DEFAULT_CAMERA_ID,
            camera_name=DEFAULT_CAMERA_NAME,
            location=DEFAULT_CAMERA_LOCATION,
            status="ACTIVE",
        )

    # -------------------------------------------------------------------------
    # Camera Operations
    # -------------------------------------------------------------------------
    def register_camera(self, camera_id, camera_name, location, status="ACTIVE"):
        """Inserts or updates camera information in the 'cameras' collection."""
        if not self._connected:
            return None

        doc = {
            "camera_id": camera_id,
            "camera_name": camera_name,
            "location": location,
            "status": status,
            "updated_at": datetime.now(),
        }
        res = self.db.cameras.update_one(
            {"camera_id": camera_id},
            {"$set": doc, "$setOnInsert": {"created_at": datetime.now()}},
            upsert=True,
        )
        return res

    def get_cameras(self):
        """Returns a list of all registered cameras."""
        if not self._connected:
            return []
        return list(self.db.cameras.find({}, {"_id": 0}))

    # -------------------------------------------------------------------------
    # Detection Operations
    # -------------------------------------------------------------------------
    def log_detection(
        self,
        camera_id,
        track_id,
        class_id,
        class_name,
        confidence,
        bounding_box,
        timestamp=None,
    ):
        """
        Logs a single bounding box detection into the 'detections' collection.
        """
        if not self._connected:
            return None

        ts = timestamp or datetime.now()
        doc = {
            "camera_id": camera_id,
            "track_id": int(track_id),
            "class_id": int(class_id),
            "class_name": class_name,
            "confidence": float(confidence),
            "bounding_box": bounding_box,  # [x1, y1, x2, y2]
            "timestamp": ts,
        }
        res = self.db.detections.insert_one(doc)
        return str(res.inserted_id)

    # -------------------------------------------------------------------------
    # Tracking Events Operations (With Smart Deduplication & Throttling)
    # -------------------------------------------------------------------------
    def log_tracking_event(
        self,
        camera_id,
        track_id,
        object_class,
        event_type="ACTIVE",
        throttle_seconds=2.0,
    ):
        """
        Records or updates an ongoing object track in 'tracking_events'.
        Prevents redundant document spam on every frame by updating 'last_seen'.
        """
        if not self._connected:
            return None

        now = datetime.now()
        cache_key = (camera_id, track_id)
        last_logged = self._tracking_cache.get(cache_key)

        # If logged recently within throttle_seconds, update last_seen only
        if last_logged and (now - last_logged).total_seconds() < throttle_seconds:
            self.db.tracking_events.update_one(
                {"camera_id": camera_id, "track_id": int(track_id)},
                {"$set": {"last_seen": now, "event_type": event_type}},
            )
            return "UPDATED"

        # Otherwise upsert track record
        self._tracking_cache[cache_key] = now
        res = self.db.tracking_events.update_one(
            {"camera_id": camera_id, "track_id": int(track_id)},
            {
                "$set": {
                    "object_class": object_class,
                    "last_seen": now,
                    "event_type": event_type,
                },
                "$setOnInsert": {
                    "camera_id": camera_id,
                    "track_id": int(track_id),
                    "first_seen": now,
                },
            },
            upsert=True,
        )
        return "INSERTED" if res.upserted_id else "UPDATED"

    # -------------------------------------------------------------------------
    # Alerts & Snapshots Operations
    # -------------------------------------------------------------------------
    def log_alert(
        self,
        camera_id,
        alert_type,
        description,
        severity="MEDIUM",
        frame=None,
        track_id=None,
    ):
        """
        Records a security alert in 'alerts' and saves an associated snapshot image.
        """
        if not self._connected:
            return None

        now = datetime.now()
        snapshot_rel_path = None

        if frame is not None:
            filename = f"alert_{camera_id}_{now.strftime('%Y%m%d_%H%M%S_%f')[:19]}.jpg"
            full_path = SNAPSHOTS_DIR / filename
            cv2.imwrite(str(full_path), frame)
            snapshot_rel_path = f"data/snapshots/{filename}"
            # Record in snapshots collection
            self.log_snapshot(
                camera_id=camera_id,
                image_path=snapshot_rel_path,
                track_id=track_id,
                timestamp=now,
            )

        doc = {
            "camera_id": camera_id,
            "alert_type": alert_type,
            "description": description,
            "severity": severity,  # LOW, MEDIUM, HIGH, CRITICAL
            "timestamp": now,
            "snapshot_path": snapshot_rel_path,
        }
        res = self.db.alerts.insert_one(doc)
        return str(res.inserted_id)

    def log_snapshot(self, camera_id, image_path, track_id=None, timestamp=None):
        """Stores image snapshot metadata in the 'snapshots' collection."""
        if not self._connected:
            return None

        doc = {
            "camera_id": camera_id,
            "image_path": str(image_path),
            "track_id": int(track_id) if track_id is not None else None,
            "timestamp": timestamp or datetime.now(),
        }
        res = self.db.snapshots.insert_one(doc)
        return str(res.inserted_id)

    def capture_snapshot(self, camera_id, frame, track_id=None, notes="Snapshot Capture"):
        """
        Saves a video frame snapshot to disk and registers metadata in
        both 'snapshots' and 'alerts' collections.
        """
        if frame is None or not self._connected:
            return None, None

        now = datetime.now()
        filename = f"snap_{camera_id}_{now.strftime('%Y%m%d_%H%M%S_%f')[:19]}.jpg"
        full_path = SNAPSHOTS_DIR / filename
        cv2.imwrite(str(full_path), frame)
        rel_path = f"data/snapshots/{filename}"

        self.log_snapshot(camera_id=camera_id, image_path=rel_path, track_id=track_id, timestamp=now)

        alert_res = self.db.alerts.insert_one({
            "camera_id": camera_id,
            "alert_type": "PHOTO_CAPTURED",
            "description": notes or f"Camera snapshot from {camera_id}",
            "severity": "LOW",
            "timestamp": now,
            "snapshot_path": rel_path,
        })
        return rel_path, str(alert_res.inserted_id)

    def get_snapshots(self, limit=60):
        """Returns list of captured snapshot records."""
        if not self._connected:
            return []
        return list(self.db.snapshots.find({}).sort("timestamp", DESCENDING).limit(limit))

    # -------------------------------------------------------------------------
    # Generic CRUD Operations
    # -------------------------------------------------------------------------
    def insert_one(self, collection_name, document):
        """Generic insert operation."""
        if not self._connected:
            return None
        return self.db[collection_name].insert_one(document)

    def find(self, collection_name, query=None, limit=100, sort_field=None, sort_order=-1):
        """Generic find operation."""
        if not self._connected:
            return []
        q = query or {}
        cursor = self.db[collection_name].find(q)
        if sort_field:
            cursor = cursor.sort(sort_field, sort_order)
        if limit:
            cursor = cursor.limit(limit)
        return list(cursor)

    def find_one(self, collection_name, query):
        """Generic find_one operation."""
        if not self._connected:
            return None
        return self.db[collection_name].find_one(query)

    def update_one(self, collection_name, query, update_values):
        """Generic update_one operation."""
        if not self._connected:
            return None
        return self.db[collection_name].update_one(query, update_values)

    def delete_one(self, collection_name, query):
        """Generic delete_one operation."""
        if not self._connected:
            return None
        return self.db[collection_name].delete_one(query)

    def delete_many(self, collection_name, query):
        """Generic delete_many operation."""
        if not self._connected:
            return None
        return self.db[collection_name].delete_many(query)

    # -------------------------------------------------------------------------
    # Backward Compatibility & Dashboard Helpers
    # -------------------------------------------------------------------------
    def log_event(
        self,
        event_type,
        track_id,
        class_name,
        confidence,
        frame=None,
        notes="",
        camera_id=DEFAULT_CAMERA_ID,
    ):
        """
        Backward-compatible event logging method.
        Maps into tracking_events, detections, and alerts collections.
        """
        if not self._connected:
            return None

        # 1. Update tracking events
        self.log_tracking_event(
            camera_id=camera_id,
            track_id=track_id,
            object_class=class_name,
            event_type=event_type,
        )

        # 2. Log detection bounding box / event
        self.log_detection(
            camera_id=camera_id,
            track_id=track_id,
            class_id=0,
            class_name=class_name,
            confidence=confidence,
            bounding_box=[],
        )

        # 3. Log alert if snapshot or notable event
        alert_id = self.log_alert(
            camera_id=camera_id,
            alert_type=event_type,
            description=notes or f"Track #{track_id} ({class_name})",
            severity="LOW",
            frame=frame,
            track_id=track_id,
        )
        return alert_id

    def get_recent_events(self, limit=50):
        """Fetches recent events in tabular format for the dashboard table."""
        if not self._connected:
            return []

        cursor = self.db.tracking_events.find({}).sort("last_seen", DESCENDING).limit(limit)
        events = []
        for doc in cursor:
            events.append({
                "id": str(doc.get("_id")),
                "camera_id": doc.get("camera_id", DEFAULT_CAMERA_ID),
                "timestamp": doc.get("last_seen", doc.get("first_seen")),
                "event_type": doc.get("event_type", "ACTIVE"),
                "track_id": doc.get("track_id"),
                "class_name": doc.get("object_class", "unknown"),
                "first_seen": doc.get("first_seen"),
                "last_seen": doc.get("last_seen"),
            })
        return events

    def get_events_df(self):
        """Returns tracking events and detections as a Pandas DataFrame for Plotly charts."""
        if not self._connected:
            return pd.DataFrame()

        cursor = self.db.detections.find({}, {"_id": 0}).sort("timestamp", ASCENDING)
        records = list(cursor)
        if not records:
            # Fallback to tracking events if detections empty
            cursor = self.db.tracking_events.find({}, {"_id": 0}).sort("last_seen", ASCENDING)
            records = list(cursor)
            if records:
                for r in records:
                    r["timestamp"] = r.get("last_seen")
                    r["class_name"] = r.get("object_class")

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    def get_summary_stats(self):
        """Returns KPI metrics for the Streamlit dashboard."""
        if not self._connected:
            return {
                "total_events": 0,
                "unique_objects": 0,
                "top_class": "N/A",
                "total_alerts": 0,
                "total_cameras": 0,
            }

        total_detections = self.db.detections.count_documents({})
        total_tracked = len(self.db.tracking_events.distinct("track_id"))
        total_alerts = self.db.alerts.count_documents({})
        total_cameras = self.db.cameras.count_documents({})

        # Find top detected class
        pipeline = [
            {"$group": {"_id": "$class_name", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 1},
        ]
        top_res = list(self.db.detections.aggregate(pipeline))
        top_class = top_res[0]["_id"] if top_res else "None"

        return {
            "total_events": total_detections if total_detections > 0 else total_tracked,
            "unique_objects": total_tracked,
            "top_class": top_class,
            "total_alerts": total_alerts,
            "total_cameras": total_cameras,
        }

    def clear_all(self, collections=None):
        """Safely cleans collections (primarily for testing and resets)."""
        if not self._connected:
            return
        target = collections or ["cameras", "detections", "tracking_events", "alerts", "snapshots"]
        for coll in target:
            self.db[coll].delete_many({})
        print(f"[INFO] Cleared collections: {target}")

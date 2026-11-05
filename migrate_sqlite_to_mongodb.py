"""
Migration Script: SQLite to MongoDB for VisionGuard AI.

Reads historical event records from SQLite (data/events.db)
and safely maps and inserts them into the MongoDB collections:
  - 'cameras'
  - 'detections'
  - 'tracking_events'
  - 'alerts'
  - 'snapshots'

Preserves the original SQLite database file as a backup.
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from src.database import DatabaseManager
from config import DB_PATH, DEFAULT_CAMERA_ID


def migrate():
    print("[MIGRATION] Starting SQLite to MongoDB migration...")

    if not DB_PATH.exists():
        print(f"[INFO] No existing SQLite database found at: {DB_PATH}. Nothing to migrate.")
        return True

    # 1. Connect to SQLite
    print(f"[INFO] Reading records from SQLite: {DB_PATH}")
    sqlite_conn = sqlite3.connect(str(DB_PATH))
    sqlite_conn.row_factory = sqlite3.Row
    cursor = sqlite_conn.cursor()

    try:
        cursor.execute("SELECT * FROM events ORDER BY id ASC")
        rows = cursor.fetchall()
    except sqlite3.OperationalError as e:
        print(f"[ERROR] Failed to query SQLite events: {e}")
        sqlite_conn.close()
        return False

    print(f"[INFO] Found {len(rows)} event records in SQLite.")

    # 2. Connect to MongoDB
    db = DatabaseManager()
    if not db.is_connected():
        print("[ERROR] Cannot migrate: Unable to connect to MongoDB server.")
        sqlite_conn.close()
        return False

    migrated_count = 0
    snapshot_count = 0

    for row in rows:
        row_dict = dict(row)
        track_id = row_dict["track_id"]
        class_name = row_dict["class_name"]
        confidence = float(row_dict["confidence"])
        event_type = row_dict["event_type"]
        snapshot_path = row_dict.get("snapshot_path")
        notes = row_dict.get("notes") or ""

        # Parse timestamp safely
        raw_ts = row_dict["timestamp"]
        try:
            ts = datetime.strptime(raw_ts, "%Y-%m-%d %H:%M:%S")
        except Exception:
            ts = datetime.now()

        # A. Upsert tracking event
        db.db.tracking_events.update_one(
            {"camera_id": DEFAULT_CAMERA_ID, "track_id": track_id},
            {
                "$set": {
                    "object_class": class_name,
                    "last_seen": ts,
                    "event_type": event_type,
                },
                "$setOnInsert": {
                    "camera_id": DEFAULT_CAMERA_ID,
                    "track_id": track_id,
                    "first_seen": ts,
                },
            },
            upsert=True,
        )

        # B. Insert detection record
        db.db.detections.insert_one({
            "camera_id": DEFAULT_CAMERA_ID,
            "track_id": track_id,
            "class_id": 0,
            "class_name": class_name,
            "confidence": confidence,
            "bounding_box": [],
            "timestamp": ts,
        })

        # C. Insert snapshot & alert if present
        if snapshot_path:
            db.db.snapshots.insert_one({
                "camera_id": DEFAULT_CAMERA_ID,
                "image_path": snapshot_path,
                "track_id": track_id,
                "timestamp": ts,
            })
            snapshot_count += 1

            db.db.alerts.insert_one({
                "camera_id": DEFAULT_CAMERA_ID,
                "alert_type": event_type,
                "description": notes or f"Migrated event: {class_name}",
                "severity": "LOW",
                "timestamp": ts,
                "snapshot_path": snapshot_path,
            })

        migrated_count += 1

    sqlite_conn.close()

    print(f"[SUCCESS] Migration complete! Successfully transferred:")
    print(f"  - {migrated_count} detection/tracking records to MongoDB")
    print(f"  - {snapshot_count} snapshot metadata records to MongoDB")
    print(f"[INFO] Original SQLite file safely preserved at: {DB_PATH}")
    return True


if __name__ == "__main__":
    migrate()


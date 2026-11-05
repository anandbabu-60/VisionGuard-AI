"""
Test script for Milestone 4: SQLite Event Logging.

Verifies:
  1. SQLite database and table auto-creation.
  2. Event insertion and snapshot saving.
  3. Querying recent events.
  4. Converting events to a Pandas DataFrame.
  5. Generating summary statistics.
"""

import sys
import cv2
import numpy as np
from pathlib import Path
from src.database import DatabaseManager
from config import DB_PATH, DATA_DIR


def run_database_tests():
    print("[TEST] Running SQLite database verification...")
    db = DatabaseManager()

    # Step 1: Check database file exists
    if not DB_PATH.exists():
        print(f"[FAIL] Database file not created at {DB_PATH}.")
        return False
    print(f"[PASS] Database file confirmed at: {DB_PATH}")

    # Step 2: Read test frame for snapshot logging
    test_frame_path = DATA_DIR / "test_frame.jpg"
    if test_frame_path.exists():
        frame = cv2.imread(str(test_frame_path))
    else:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Step 3: Insert test events
    print("[INFO] Logging test events...")
    event_id1 = db.log_event(
        event_type="OBJECT_ENTERED",
        track_id=1,
        class_name="person",
        confidence=0.88,
        frame=frame,
        notes="Person entered camera view"
    )
    event_id2 = db.log_event(
        event_type="OBJECT_DETECTED",
        track_id=2,
        class_name="cell phone",
        confidence=0.75,
        frame=None,
        notes="Device detected"
    )
    print(f"[PASS] Inserted events with IDs: {event_id1}, {event_id2}")

    # Step 4: Query recent events
    recent = db.get_recent_events(limit=5)
    print(f"[INFO] Fetched {len(recent)} recent events:")
    for ev in recent:
        print(f"  -> ID: {ev['id']} | Time: {ev['timestamp']} | Type: {ev['event_type']} | Object: {ev['class_name']} (#{ev['track_id']})")

    # Step 5: Test DataFrame conversion
    df = db.get_events_df()
    print(f"[INFO] DataFrame row count: {len(df)}, columns: {list(df.columns)}")
    assert not df.empty, "DataFrame should not be empty"

    # Step 6: Test Summary Stats
    stats = db.get_summary_stats()
    print(f"[INFO] Summary statistics: {stats}")
    assert stats["total_events"] >= 2, "Total events count mismatch"
    assert stats["unique_objects"] >= 2, "Unique objects count mismatch"

    print("[PASS] Milestone 4 verification successful!")
    return True


if __name__ == "__main__":
    success = run_database_tests()
    sys.exit(0 if success else 1)


"""
Unit & Integration Test Suite for MongoDB Database Layer in VisionGuard AI.

Verifies:
  1. Connection to MongoDB server.
  2. Database and collection initialization.
  3. Automatic index creation.
  4. CRUD Operations (Insert, Find, Update, Delete).
  5. Camera registration and retrieval.
  6. Detection logging.
  7. Tracking event logging and throttling (updating last_seen).
  8. Alert and Snapshot metadata persistence.
  9. Pandas DataFrame conversion & KPI summary statistics.
"""

import sys
from datetime import datetime
import pandas as pd
from src.database import DatabaseManager
from config import MONGODB_DATABASE, DEFAULT_CAMERA_ID


def run_tests():
    print("=" * 60)
    print(" VISIONGUARD AI - MONGODB VERIFICATION TEST SUITE ")
    print("=" * 60)

    # Use a dedicated test database to keep production collections clean
    test_db_name = f"{MONGODB_DATABASE}_test"
    print(f"[TEST 1] Initializing DatabaseManager on '{test_db_name}'...")
    db = DatabaseManager(db_name=test_db_name)

    if not db.is_connected():
        print("[FAIL] Could not establish connection to MongoDB.")
        return False
    print("[PASS] MongoDB connection established.")

    # Clean test collections before test run
    db.clear_all()

    # TEST 2: Collections & Index creation
    print("\n[TEST 2] Verifying collections and indexes...")
    db.init_database()
    collections = db.db.list_collection_names()
    print(f"[INFO] Existing collections in test database: {collections}")

    # Check indexes on detections
    det_indexes = db.db.detections.index_information()
    print(f"[INFO] Indexes on 'detections': {list(det_indexes.keys())}")
    assert "camera_id_1_timestamp_-1" in det_indexes, "Compound index on detections missing"
    print("[PASS] Collection indexes verified.")

    # TEST 3: Camera Registration
    print("\n[TEST 3] Testing camera registration and retrieval...")
    res = db.register_camera("CAM_TEST_01", "Lab Camera", "Testing Lab", status="ACTIVE")
    cameras = db.get_cameras()
    assert any(c["camera_id"] == "CAM_TEST_01" for c in cameras), "Registered camera not found"
    print(f"[PASS] Camera registered and retrieved successfully ({len(cameras)} camera(s)).")

    # TEST 4: Generic CRUD Operations
    print("\n[TEST 4] Testing generic CRUD operations...")
    # Insert
    doc = {"name": "test_marker", "val": 42, "created_at": datetime.now()}
    ins_res = db.insert_one("test_temp", doc)
    doc_id = ins_res.inserted_id
    assert doc_id is not None, "Failed to insert document"

    # Find
    found = db.find_one("test_temp", {"val": 42})
    assert found is not None and found["name"] == "test_marker", "Failed to find document"

    # Update
    db.update_one("test_temp", {"_id": doc_id}, {"$set": {"val": 100}})
    updated = db.find_one("test_temp", {"_id": doc_id})
    assert updated["val"] == 100, "Failed to update document"

    # Delete
    del_res = db.delete_one("test_temp", {"_id": doc_id})
    assert del_res.deleted_count == 1, "Failed to delete document"
    db.db.drop_collection("test_temp")
    print("[PASS] Generic CRUD operations (Insert, Find, Update, Delete) verified.")

    # TEST 5: Detection Logging
    print("\n[TEST 5] Testing detection logging...")
    det_id = db.log_detection(
        camera_id="CAM_TEST_01",
        track_id=10,
        class_id=0,
        class_name="person",
        confidence=0.92,
        bounding_box=[100, 150, 250, 400],
    )
    assert det_id is not None, "Detection insert failed"
    dets = db.find("detections", {"track_id": 10})
    assert len(dets) == 1 and dets[0]["class_name"] == "person"
    print(f"[PASS] Detection logged with ID {det_id}.")

    # TEST 6: Tracking Event Logging & Throttling
    print("\n[TEST 6] Testing tracking event deduplication & throttling...")
    status1 = db.log_tracking_event("CAM_TEST_01", 10, "person", event_type="ACTIVE", throttle_seconds=2.0)
    assert status1 == "INSERTED", f"Expected INSERTED, got {status1}"

    # Immediately call again to test throttling
    status2 = db.log_tracking_event("CAM_TEST_01", 10, "person", event_type="ACTIVE", throttle_seconds=2.0)
    assert status2 == "UPDATED", f"Expected UPDATED (throttled), got {status2}"

    # Confirm only one document exists in tracking_events
    track_docs = db.find("tracking_events", {"track_id": 10})
    assert len(track_docs) == 1, f"Expected 1 tracking doc, found {len(track_docs)}"
    print("[PASS] Tracking deduplication verified: only 1 document maintained per active track.")

    # TEST 7: Alert and Snapshot Persistence
    print("\n[TEST 7] Testing alert and snapshot metadata logging...")
    alert_id = db.log_alert(
        camera_id="CAM_TEST_01",
        alert_type="ZONE_INTRUSION",
        description="Unauthorized person in restricted area",
        severity="HIGH",
        frame=None,
        track_id=10,
    )
    assert alert_id is not None, "Alert insert failed"

    snap_id = db.log_snapshot(
        camera_id="CAM_TEST_01",
        image_path="data/snapshots/sample.jpg",
        track_id=10,
    )
    assert snap_id is not None, "Snapshot insert failed"
    print("[PASS] Alert and Snapshot metadata logged successfully.")

    # TEST 8: DataFrame Retrieval & KPI Summary Statistics
    print("\n[TEST 8] Testing DataFrame and KPI statistics...")
    df = db.get_events_df()
    assert not df.empty, "DataFrame should not be empty"
    print(f"[INFO] Retrieved DataFrame with {len(df)} rows and columns: {list(df.columns)}")

    stats = db.get_summary_stats()
    print(f"[INFO] Summary statistics: {stats}")
    assert stats["total_events"] >= 1, "Stats total_events mismatch"
    assert stats["unique_objects"] >= 1, "Stats unique_objects mismatch"
    print("[PASS] DataFrame queries and KPI statistics verified.")

    # Clean up test database
    db.client.drop_database(test_db_name)
    print(f"\n[INFO] Cleaned up temporary test database: {test_db_name}")

    print("\n" + "=" * 60)
    print(" ALL MONGODB TESTS PASSED SUCCESSFULLY! ")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)


"""
Unit and Integration Test Suite for SurveillanceEngine.

Verifies:
  1. Bidirectional virtual tripwire crossing (IN / OUT).
  2. Restricted zone intrusion detection (Point-in-polygon).
  3. Crowd density and overcrowding alarms.
  4. Loitering dwell-time detection.
  5. Visual HUD overlay rendering.
"""

import sys
import time
import cv2
import numpy as np
from pathlib import Path
from src.surveillance import SurveillanceEngine
from config import DATA_DIR


def run_tests():
    print("=" * 60)
    print(" VISIONGUARD AI - SURVEILLANCE ENGINE AUDIT & TEST SUITE ")
    print("=" * 60)

    engine = SurveillanceEngine(frame_width=640, frame_height=480, crowd_threshold=4, loitering_threshold_sec=1.5)

    # TEST 1: Bidirectional Virtual Tripwire (IN Crossing)
    print("\n[TEST 1] Testing Virtual Tripwire Entry (IN Crossing)...")
    line_y = engine.tripwire_line[0][1]

    # Frame 1: Person above line
    f1_objects = [{"track_id": 1, "class_name": "person", "center": (200, line_y - 20), "box": [180, line_y - 40, 220, line_y]}]
    engine.update_tripwire(f1_objects)

    # Frame 2: Person moves below line (Crosses downwards)
    f2_objects = [{"track_id": 1, "class_name": "person", "center": (200, line_y + 20), "box": [180, line_y, 220, line_y + 40]}]
    events_in = engine.update_tripwire(f2_objects)

    assert engine.in_count == 1, f"Expected in_count=1, got {engine.in_count}"
    assert len(events_in) == 1 and events_in[0]["direction"] == "IN", "Failed to detect IN crossing"
    print(f"[PASS] Detected ENTRY crossing: IN Count = {engine.in_count}, Net Occupancy = {engine.get_net_occupancy()}")

    # TEST 2: Bidirectional Virtual Tripwire (OUT Crossing)
    print("\n[TEST 2] Testing Virtual Tripwire Exit (OUT Crossing)...")
    # Frame 3: Car below line
    f3_objects = [{"track_id": 2, "class_name": "car", "center": (350, line_y + 30), "box": [320, line_y + 10, 380, line_y + 50]}]
    engine.update_tripwire(f3_objects)

    # Frame 4: Car moves above line (Crosses upwards)
    f4_objects = [{"track_id": 2, "class_name": "car", "center": (350, line_y - 30), "box": [320, line_y - 50, 380, line_y - 10]}]
    events_out = engine.update_tripwire(f4_objects)

    assert engine.out_count == 1, f"Expected out_count=1, got {engine.out_count}"
    assert len(events_out) == 1 and events_out[0]["direction"] == "OUT", "Failed to detect OUT crossing"
    print(f"[PASS] Detected EXIT crossing: OUT Count = {engine.out_count}, Net Occupancy = {engine.get_net_occupancy()}")

    # TEST 3: Restricted Zone Perimeter Intrusion
    print("\n[TEST 3] Testing Restricted Zone Polygon Intrusion...")
    # Point inside zone
    zx_center = int((engine.restricted_zone[0][0] + engine.restricted_zone[1][0]) / 2)
    zy_center = int((engine.restricted_zone[0][1] + engine.restricted_zone[2][1]) / 2)

    intruder_sample = [
        {"track_id": 3, "class_name": "person", "confidence": 0.95, "center": (zx_center, zy_center), "box": [zx_center - 10, zy_center - 10, zx_center + 10, zy_center + 10]},
        {"track_id": 4, "class_name": "person", "confidence": 0.88, "center": (50, 50), "box": [40, 40, 60, 60]}  # Safe point outside
    ]
    intruders = engine.check_zone_intrusion(intruder_sample)
    assert len(intruders) == 1 and intruders[0]["track_id"] == 3, f"Expected 1 intruder (ID #3), got {len(intruders)}"
    print(f"[PASS] Correctly flagged intruder ID #{intruders[0]['track_id']} inside restricted zone.")

    # TEST 4: Crowd Density & Overcrowding
    print("\n[TEST 4] Testing Crowd Density Thresholds...")
    crowd_sample = [{"track_id": i, "class_name": "person", "center": (100 + i * 20, 200)} for i in range(5)]
    crowd_res = engine.check_crowd_density(crowd_sample)
    assert crowd_res["is_crowded"] is True, "Expected crowd alert for 5 persons (threshold=4)"
    print(f"[PASS] Overcrowding alarm triggered: {crowd_res['person_count']} persons (Limit: {crowd_res['threshold']})")

    # TEST 5: Loitering (Dwell Time)
    print("\n[TEST 5] Testing Loitering / Dwell Time Detection...")
    loiter_obj = [{"track_id": 99, "class_name": "person", "center": (150, 150), "box": [130, 130, 170, 170]}]
    engine.check_loitering(loiter_obj)
    print("[INFO] Simulating 1.6s dwell time...")
    time.sleep(1.6)
    loiterers = engine.check_loitering(loiter_obj)
    assert len(loiterers) == 1 and loiterers[0]["track_id"] == 99, "Failed to flag loitering track #99"
    print(f"[PASS] Loitering detected for ID #{loiterers[0]['track_id']} (Dwell: {loiterers[0]['dwell_seconds']}s)")

    # TEST 6: Visual HUD Rendering
    print("\n[TEST 6] Testing Surveillance HUD Overlay Rendering...")
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    hud_frame = engine.draw_hud(test_img, has_intruders=True, is_crowded=True)
    assert hud_frame is not None and hud_frame.shape == (480, 640, 3)

    out_path = DATA_DIR / "test_hud_dummy.jpg"
    cv2.imwrite(str(out_path), hud_frame)
    print(f"[PASS] Rendered and saved HUD test overlay to: {out_path}")

    # TEST 7: Dynamic Coordinate Adjustment
    print("\n[TEST 7] Testing Dynamic Interactive Coordinate Adjustment...")
    engine.set_tripwire_position(y_pct=0.30, start_x_pct=0.10, end_x_pct=0.90)
    assert engine.tripwire_line[0][1] == int(480 * 0.30), "Failed to adjust tripwire Y position"
    assert engine.tripwire_line[0][0] == int(640 * 0.10), "Failed to adjust tripwire start X"
    assert engine.tripwire_line[1][0] == int(640 * 0.90), "Failed to adjust tripwire end X"

    engine.set_restricted_zone(x_pct=0.10, y_pct=0.20, w_pct=0.30, h_pct=0.40)
    assert engine.restricted_zone[0][0] == int(640 * 0.10)
    assert engine.restricted_zone[0][1] == int(480 * 0.20)
    assert engine.restricted_zone[2][0] == int(640 * 0.40)
    assert engine.restricted_zone[2][1] == int(480 * 0.60)
    print(f"[PASS] Dynamic coordinate repositioning verified (Tripwire & Restricted Zone).")

    # TEST 8: PPE Safety Compliance Evaluation
    print("\n[TEST 8] Testing PPE Safety Compliance Evaluation...")
    ppe_sample = [
        {"track_id": 10, "class_name": "person", "confidence": 0.90, "box": [100, 100, 200, 300]},
        {"track_id": 11, "class_name": "no_hard_hat", "confidence": 0.85, "box": [120, 100, 180, 150]},
        {"track_id": 12, "class_name": "person", "confidence": 0.92, "box": [300, 100, 400, 300]},
        {"track_id": 13, "class_name": "hard_hat", "confidence": 0.88, "box": [320, 100, 380, 150]},
        {"track_id": 14, "class_name": "safety_vest", "confidence": 0.89, "box": [310, 160, 390, 250]}
    ]
    ppe_report = engine.check_ppe_compliance(ppe_sample)
    assert len(ppe_report["violations"]) == 1, f"Expected 1 violation, got {len(ppe_report['violations'])}"
    assert ppe_report["violations"][0]["type"] == "NO_HARD_HAT", "Expected NO_HARD_HAT violation"
    assert ppe_report["compliance_rate_pct"] == 50.0, f"Expected 50.0% compliance, got {ppe_report['compliance_rate_pct']}"
    print(f"[PASS] PPE Compliance evaluated: Rate = {ppe_report['compliance_rate_pct']}%, Violations = {len(ppe_report['violations'])}")

    print("\n" + "=" * 60)
    print(" ALL SURVEILLANCE RULES VERIFIED & PASSED! ")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)


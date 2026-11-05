# VisionGuard AI - Intelligent AI Monitoring Dashboard

VisionGuard AI is a real-time computer vision security and surveillance dashboard built with **Python**, **OpenCV**, **YOLOv8**, **ByteTrack**, **MongoDB**, **Plotly**, and **Streamlit**.

---

## 🛠️ Technology Stack
- **Video Capture**: OpenCV
- **Object Detection**: YOLOv8 (Ultralytics)
- **Multi-Object Tracking**: ByteTrack (with `lapx` solver)
- **Primary Database**: MongoDB (`VisionGuardAI`) via PyMongo
- **Database Visualization**: MongoDB Compass
- **Analytics & Trends**: Plotly
- **Interactive Dashboard**: Streamlit

---

## 📁 Project Structure

```text
VisionGuard-AI/
├── .env                       # Local environment configuration (MongoDB URI, database)
├── .env.example               # Template environment configuration
├── .gitignore                 # Excludes .env, virtual environments, caches, and backups
├── config.py                  # Global settings, paths, and environment variable loaders
├── requirements.txt           # Python dependencies
├── README.md                  # Comprehensive documentation and setup guide
├── app.py                     # Streamlit monitoring dashboard
│
├── src/                       # Core modular application package
│   ├── __init__.py            # Package initializer
│   ├── camera.py              # Milestone 1: OpenCV video capture (Webcam & Video files)
│   ├── detector.py            # Milestone 2: YOLOv8 object detection wrapper
│   ├── tracker.py             # Milestone 3: ByteTrack tracking & motion trails
│   ├── surveillance.py        # Milestone 4+: Intelligent rules (Tripwire, Zones, Crowd, Loitering)
│   ├── notifier.py            # Email alert dispatcher (SMTP with snapshot attachments)
│   ├── database.py            # MongoDB database layer (PyMongo)
│   └── analytics.py           # Plotly visual charts and metrics
│
├── models/                    # YOLO weight files (e.g. yolov8n.pt)
├── data/                      # Local data artifacts
│   ├── events.db              # (Preserved legacy SQLite database backup)
│   └── snapshots/             # Saved security alert snapshots
│
├── test_camera.py             # Milestone 1 test suite
├── test_detector.py           # Milestone 2 test suite
├── test_tracker.py            # Milestone 3 test suite
├── test_surveillance.py       # Surveillance rules verification suite (Tripwire, Zones, Crowd)
├── test_mongodb.py            # Milestone 4 MongoDB comprehensive test suite
└── migrate_sqlite_to_mongodb.py # One-click SQLite -> MongoDB migration script
```

---

## 🍃 MongoDB Architecture

Database Name: **`VisionGuardAI`**

### Collections Schema:
1. **`cameras`**: Camera registry (`camera_id`, `camera_name`, `location`, `status`, `created_at`, `updated_at`). Unique index on `camera_id`.
2. **`detections`**: Raw bounding box detections (`camera_id`, `track_id`, `class_id`, `class_name`, `confidence`, `bounding_box`, `timestamp`). Compound index on `(camera_id, timestamp)`.
3. **`tracking_events`**: Persistent object tracking history (`camera_id`, `track_id`, `object_class`, `first_seen`, `last_seen`, `event_type`). Throttled automatically to avoid frame-by-frame document spam.
4. **`alerts`**: High-priority security triggers (`camera_id`, `alert_type`, `description`, `severity`, `timestamp`, `snapshot_path`).
5. **`snapshots`**: Metadata for captured incident frames (`camera_id`, `image_path`, `track_id`, `timestamp`).

---

## 🧭 Connecting via MongoDB Compass

1. Launch **MongoDB Compass**.
2. In the **New Connection** input box, paste:
   ```text
   mongodb://localhost:27017/
   ```
3. Click **Connect**.
4. In the left database list, open **`VisionGuardAI`**.
5. You will see all 5 collections:
   - `cameras`
   - `detections`
   - `tracking_events`
   - `alerts`
   - `snapshots`

---

## 🚀 Setup & Execution Guide

### 1. Installation
Activate your virtual environment and install requirements:
```bash
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run Database & Module Tests
Verify that every component is working:
```bash
# Verify MongoDB layer (Connection, Indexes, CRUD, Throttling)
python test_mongodb.py

# Verify Camera access
python test_camera.py --verify

# Verify YOLO Object Detection
python test_detector.py --verify

# Verify ByteTrack Tracking
python test_tracker.py --verify

# Verify Surveillance Engine (Tripwire, Restricted Zone, Crowd, Loitering)
python test_surveillance.py
```

### 3. Migrate Legacy SQLite Data (If applicable)
To safely copy historical records from SQLite to MongoDB:
```bash
python migrate_sqlite_to_mongodb.py
```

### 4. Launch the Streamlit Monitoring Dashboard
Run the dashboard in your browser:
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🚀 Deployment & Containerization
- **Streamlit Community Cloud**: Fast 1-click free cloud hosting (uses `packages.txt` & `.streamlit/config.toml`).
- **Docker Compose**: Production-ready containerized deployment bundling VisionGuard AI + MongoDB (`docker compose up -d`).
- **Complete Deployment Guide**: See [DEPLOYMENT.md](DEPLOYMENT.md) for full instructions.

---

## 🏁 Milestones Tracker
- [x] Milestone 1: OpenCV Webcam & RTSP Video Capture (`src/camera.py`)
- [x] Milestone 2: YOLOv8 Object Detection Engine (`src/detector.py`)
- [x] Milestone 3: ByteTrack Multi-Object Tracking & Motion Trails (`src/tracker.py`)
- [x] Milestone 4: MongoDB Database Layer & Migration (`src/database.py`, `migrate_sqlite_to_mongodb.py`)
- [x] Milestone 5: High-Tech Cyber Security HUD & Plotly Analytics (`src/analytics.py`, `app.py`)
- [x] Milestone 6: Intelligent Surveillance Rules: Virtual Tripwire IN/OUT, Polygon Intrusion & Crowd Density (`src/surveillance.py`)
- [x] Milestone 7: Offline Surveillance Video Upload, AI Checking, Monetization Analytics & H.264 MP4 Download (`app.py`)
- [x] Milestone 8: Real-Time Interactive Coordinate Sliders for Tripwires & Restricted Zones (`app.py`, `src/surveillance.py`)
- [x] Milestone 9: Web Audio Dual-Sweep Siren Alarms & Snapshot ZIP Export (`app.py`)
- [x] Milestone 10: Custom YOLOv8 PPE Training Pipeline for Helmets & Safety Vests (`train_ppe.py`)
- [x] Milestone 11: Executive PDF Incident Dossier Generator with Cryptographic Seals (`src/report.py`)

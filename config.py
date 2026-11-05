"""
Configuration settings for VisionGuard AI.
Keep all tunable parameters in one place for easy adjustments.
"""

import os
from pathlib import Path

# Base project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SNAPSHOTS_DIR = DATA_DIR / "snapshots"
MODELS_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
SNAPSHOTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
ASSETS_DIR.mkdir(exist_ok=True)

# Load environment configuration from .env (with zero-dependency fallback)
try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
except ImportError:
    # Fallback parser if python-dotenv is not installed in the active environment
    env_file = BASE_DIR / ".env"
    if env_file.exists():
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))

# MongoDB Database configuration
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "VisionGuardAI")

# Legacy SQLite database path (retained for data migration)
DB_PATH = DATA_DIR / "events.db"

# Camera identification & settings
DEFAULT_CAMERA_ID = os.getenv("DEFAULT_CAMERA_ID", "CAM_01")
DEFAULT_CAMERA_NAME = os.getenv("DEFAULT_CAMERA_NAME", "Primary Surveillance Camera")
DEFAULT_CAMERA_LOCATION = os.getenv("DEFAULT_CAMERA_LOCATION", "Entrance Hall")

CAMERA_INDEX = 0          # Default built-in webcam is usually 0
FRAME_WIDTH = 640         # Video capture width
FRAME_HEIGHT = 480        # Video capture height
FPS = 30                  # Target frames per second

# YOLO Detection & Tracking settings
YOLO_MODEL_NAME = "yolov8n.pt"  # Lightweight nano model
YOLO_MODEL_PATH = MODELS_DIR / YOLO_MODEL_NAME
CONFIDENCE_THRESHOLD = 0.45     # Minimum confidence to accept a detection
TRACKER_TYPE = "bytetrack.yaml" # Built-in ByteTrack configuration




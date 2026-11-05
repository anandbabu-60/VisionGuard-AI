"""
VisionGuard AI - Custom YOLOv8 Model Training Pipeline for PPE & Safety Compliance.

Supports fine-tuning YOLOv8 on custom security datasets:
  - Personal Protective Equipment (PPE): Hard-Hats (Helmets), Safety Vests, Violations
  - Custom Industrial Hazards & Unauthorized Objects

Usage:
  python train_ppe.py --verify                 # Runs automated pipeline health test
  python train_ppe.py --epochs 25 --batch 16   # Full training run
  python train_ppe.py --data data/custom.yaml  # Custom dataset training
"""

import argparse
import sys
import shutil
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

from config import BASE_DIR, DATA_DIR, MODELS_DIR

DEFAULT_DATA_YAML = DATA_DIR / "ppe_dataset.yaml"


def generate_ppe_yaml_template(output_path=DEFAULT_DATA_YAML):
    """
    Generates a standard YOLOv8 YAML dataset configuration for PPE detection.
    Classes:
      0: hard_hat (Safety Helmet)
      1: safety_vest (High-Visibility Vest)
      2: no_hard_hat (Helmet Violation)
      3: no_safety_vest (Vest Violation)
      4: person (Worker)
    """
    dataset_dir = DATA_DIR / "ppe_dataset"
    content = f"""# VisionGuard AI - PPE & Safety Gear Dataset Configuration
path: {dataset_dir.as_posix()}
train: images/train
val: images/val
test: images/test

# PPE Classes
names:
  0: hard_hat
  1: safety_vest
  2: no_hard_hat
  3: no_safety_vest
  4: person
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[INFO] Generated PPE dataset template at: {output_path}")
    return output_path


def create_synthetic_verification_dataset(target_dir: Path):
    """
    Generates a minimal 4-image synthetic dataset to verify training pipeline health.
    """
    for split in ["train", "val"]:
        img_dir = target_dir / "images" / split
        lbl_dir = target_dir / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for i in range(2):
            img_path = img_dir / f"sample_{i}.jpg"
            lbl_path = lbl_dir / f"sample_{i}.txt"

            # Create synthetic 320x320 image
            img = np.zeros((320, 320, 3), dtype=np.uint8)
            cv2.rectangle(img, (80, 40), (240, 280), (120, 120, 120), -1)  # Person body
            cv2.circle(img, (160, 90), 35, (0, 220, 255), -1)              # Yellow hard-hat
            cv2.rectangle(img, (100, 140), (220, 240), (0, 200, 0), -1)   # Green safety vest
            cv2.imwrite(str(img_path), img)

            # YOLO format: <class_id> <x_center> <y_center> <width> <height> (normalized 0-1)
            # 0: hard_hat, 1: safety_vest, 4: person
            annotations = [
                "4 0.50 0.50 0.50 0.75",  # person
                "0 0.50 0.28 0.22 0.22",  # hard_hat
                "1 0.50 0.59 0.38 0.31",  # safety_vest
            ]
            with open(lbl_path, "w", encoding="utf-8") as f:
                f.write("\n".join(annotations) + "\n")

    yaml_path = target_dir / "verify_ppe.yaml"
    content = f"""path: {target_dir.as_posix()}
train: images/train
val: images/val
names:
  0: hard_hat
  1: safety_vest
  2: no_hard_hat
  3: no_safety_vest
  4: person
"""
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(content)
    return yaml_path


def train_model(
    data_yaml_path,
    base_model="models/yolov8n.pt",
    epochs=15,
    imgsz=640,
    batch=8,
    device="cpu",
    output_model_name="ppe_yolov8n.pt"
):
    """
    Executes fine-tuning of YOLOv8 model on the specified dataset.
    """
    base_model_path = Path(base_model)
    if not base_model_path.exists():
        base_model_path = MODELS_DIR / "yolov8n.pt"

    print("=" * 60)
    print(" VISIONGUARD AI - MODEL TRAINING PIPELINE ")
    print("=" * 60)
    print(f"[INFO] Base Model: {base_model_path}")
    print(f"[INFO] Dataset Config: {data_yaml_path}")
    print(f"[INFO] Target Epochs: {epochs} | Batch: {batch} | ImgSz: {imgsz} | Device: {device}")

    model = YOLO(str(base_model_path))

    results = model.train(
        data=str(data_yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=0,  # Windows safe multiprocessing
        verbose=True,
        project=str(DATA_DIR / "runs"),
        name="ppe_train",
        exist_ok=True,
    )

    # Save output model weights to models/ directory
    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    if not best_weights.exists():
        best_weights = Path(results.save_dir) / "weights" / "last.pt"

    dest_model_path = MODELS_DIR / output_model_name
    if best_weights.exists():
        shutil.copy2(str(best_weights), str(dest_model_path))
        print(f"[SUCCESS] Best trained model weights saved to: {dest_model_path}")
    else:
        print(f"[WARN] Trained weights not located at {best_weights}, saving current model checkpoint.")
        model.save(str(dest_model_path))

    return dest_model_path


def verify_pipeline():
    """
    Executes an end-to-end dry-run verification of the training pipeline.
    """
    print("=" * 60)
    print(" VISIONGUARD AI - TRAINING PIPELINE HEALTH VERIFICATION ")
    print("=" * 60)

    test_dir = DATA_DIR / "test_ppe_verify"
    if test_dir.exists():
        shutil.rmtree(str(test_dir), ignore_errors=True)

    print("[STEP 1] Generating synthetic verification dataset...")
    test_yaml = create_synthetic_verification_dataset(test_dir)
    assert test_yaml.exists(), "Verification YAML creation failed"
    print(f"[PASS 1] Dataset structure created: {test_yaml}")

    print("\n[STEP 2] Running 1-epoch test fine-tune...")
    trained_model = train_model(
        data_yaml_path=test_yaml,
        epochs=1,
        imgsz=320,
        batch=2,
        device="cpu",
        output_model_name="ppe_test_verify.pt"
    )

    assert trained_model.exists() and trained_model.stat().st_size > 0, "Model weight export failed"
    print(f"[PASS 2] Model trained and exported successfully: {trained_model} ({trained_model.stat().st_size} bytes)")

    print("\n[STEP 3] Testing inference with newly fine-tuned model...")
    test_yolo = YOLO(str(trained_model))
    sample_img = cv2.imread(str(test_dir / "images" / "train" / "sample_0.jpg"))
    preds = test_yolo(sample_img, verbose=False)
    assert preds is not None, "Inference on newly trained model failed"
    print(f"[PASS 3] Inference executed successfully! Detected classes: {test_yolo.names}")

    # Clean up verification sandbox
    if test_dir.exists():
        shutil.rmtree(str(test_dir), ignore_errors=True)
    if trained_model.exists():
        trained_model.unlink(missing_ok=True)

    print("\n" + "=" * 60)
    print(" ALL TRAINING PIPELINE CHECKS PASSED! PIPELINE IS PRODUCTION READY. ")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VisionGuard AI - Custom YOLOv8 Training Pipeline")
    parser.add_argument("--verify", action="store_true", help="Runs automated pipeline health verification test")
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA_YAML), help="Path to dataset YAML file")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--device", type=str, default="cpu", help="Device: 'cpu' or '0' (for GPU)")
    parser.add_argument("--model", type=str, default="models/yolov8n.pt", help="Base model weights")
    parser.add_argument("--out", type=str, default="ppe_yolov8n.pt", help="Output model filename")

    args = parser.parse_args()

    # Always ensure default YAML template exists
    generate_ppe_yaml_template()

    if args.verify:
        verify_pipeline()
    else:
        yaml_p = Path(args.data)
        if not yaml_p.exists():
            print(f"[ERROR] Dataset configuration not found: {yaml_p}")
            print(f"[TIP] A template has been generated for you at: {DEFAULT_DATA_YAML}")
            sys.exit(1)

        train_model(
            data_yaml_path=yaml_p,
            base_model=args.model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            output_model_name=args.out
        )


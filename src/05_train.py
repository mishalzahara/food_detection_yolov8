"""
05_train.py
-----------
Train YOLOv8 models for food object detection.

Trains two models sequentially:
  • YOLOv8n  (nano)   – fast baseline
  • YOLOv8m  (medium) – tuned model with more epochs

Both models are trained on the dataset described in configs/food.yaml.

Usage:
    python src/05_train.py
    # or, to train only one model:
    python src/05_train.py --model nano
    python src/05_train.py --model medium
"""

import argparse

from ultralytics import YOLO


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
YAML_PATH = "/content/drive/MyDrive/Wajahat_Project/food.yaml"

NANO_CONFIG = dict(
    weights   = "yolov8n.pt",
    epochs    = 50,
    imgsz     = 640,
    batch     = 16,
    name      = "food_detection_yolov8n",
)

MEDIUM_CONFIG = dict(
    weights   = "yolov8m.pt",
    epochs    = 150,
    imgsz     = 640,
    batch     = 8,
    name      = "food_detection_yolov8m_tuned",
    # Uncomment to tune learning rates:
    # lr0     = 0.01,
    # lrf     = 0.001,
    # patience= 50,     # early stopping
)


# ─────────────────────────────────────────────
# Training helpers
# ─────────────────────────────────────────────
def train_model(config: dict, data_yaml: str) -> YOLO:
    """
    Instantiate and train a YOLOv8 model.

    Parameters
    ----------
    config    : dict with keys: weights, epochs, imgsz, batch, name, and
                any additional ultralytics train() keyword arguments.
    data_yaml : str  path to the dataset YAML config.

    Returns
    -------
    Trained YOLO model instance.
    """
    weights = config.pop("weights")
    print(f"\n{'='*60}")
    print(f"Training  : {weights}  →  run name: {config.get('name')}")
    print(f"Epochs    : {config['epochs']}  |  Batch: {config['batch']}")
    print(f"{'='*60}\n")

    model = YOLO(weights)
    model.train(data=data_yaml, **config)
    return model


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main(which: str = "both") -> None:
    if which in ("nano", "both"):
        train_model(dict(NANO_CONFIG), YAML_PATH)

    if which in ("medium", "both"):
        train_model(dict(MEDIUM_CONFIG), YAML_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 food detection models.")
    parser.add_argument(
        "--model",
        choices=["nano", "medium", "both"],
        default="both",
        help="Which model to train (default: both)",
    )
    args = parser.parse_args()
    main(args.model)

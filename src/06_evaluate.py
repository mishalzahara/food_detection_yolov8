"""
06_evaluate.py
--------------
Load trained YOLOv8 models, run validation, print performance metrics,
and display evaluation plots (F1, Precision, Recall curves + Confusion Matrix).

Usage:
    python src/06_evaluate.py
    python src/06_evaluate.py --model_path /path/to/best.pt --val_dir /path/to/val/results
"""

import argparse
import os

import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO


# ─────────────────────────────────────────────
# Default paths  (override via CLI or env vars)
# ─────────────────────────────────────────────
YAML_PATH         = "/content/drive/MyDrive/Wajahat_Project/food.yaml"
NANO_WEIGHTS      = "/content/runs/detect/food_detection_yolov8n/weights/best.pt"
MEDIUM_WEIGHTS    = "/content/runs/detect/food_detection_yolov8m_tuned2/weights/best.pt"
NANO_VAL_DIR      = "/content/runs/detect/val"
MEDIUM_VAL_DIR    = "/content/runs/detect/val2"


# ─────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────
def evaluate_model(model_path: str, data_yaml: str) -> dict:
    """
    Load a model, run val(), and return a metrics summary dict.

    Returns
    -------
    {
        "map50"       : float,
        "map50_95"    : float,
        "class_names" : list[str],
        "ap50"        : list[float],
    }
    """
    print(f"\nLoading model: {model_path}")
    model   = YOLO(model_path)
    metrics = model.val(data=data_yaml)

    summary = {
        "map50"       : metrics.box.map50,
        "map50_95"    : metrics.box.map,
        "class_names" : list(metrics.names.values()),
        "ap50"        : list(metrics.box.ap),
    }
    return summary


def print_metrics(summary: dict, label: str = "Model") -> None:
    """Pretty-print overall and class-wise metrics."""
    print(f"\n{'─'*50}")
    print(f"  {label}  –  Validation Metrics")
    print(f"{'─'*50}")
    print(f"  mAP50      : {summary['map50']:.4f}")
    print(f"  mAP50-95   : {summary['map50_95']:.4f}")
    print(f"\n  Class-wise AP50:")
    for name, ap in zip(summary["class_names"], summary["ap50"]):
        print(f"    {name:<18} {ap:.4f}")


def compare_models(nano: dict, medium: dict) -> None:
    """Print a side-by-side comparison table."""
    print(f"\n{'='*65}")
    print(f"  {'Class':<18} {'YOLOv8n AP50':>14} {'YOLOv8m AP50':>14}")
    print(f"{'='*65}")
    for name, n_ap, m_ap in zip(nano["class_names"], nano["ap50"], medium["ap50"]):
        print(f"  {name:<18} {n_ap:>14.4f} {m_ap:>14.4f}")
    print(f"{'─'*65}")
    print(f"  {'mAP50':<18} {nano['map50']:>14.4f} {medium['map50']:>14.4f}")
    print(f"  {'mAP50-95':<18} {nano['map50_95']:>14.4f} {medium['map50_95']:>14.4f}")
    print(f"{'='*65}\n")


# ─────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────
_PLOTS = {
    "F1 Curve"         : "BoxF1_curve.png",
    "Precision Curve"  : "BoxP_curve.png",
    "Recall Curve"     : "BoxR_curve.png",
    "Confusion Matrix" : "confusion_matrix.png",
}


def display_plots(val_dir: str, label: str = "") -> None:
    """Read and display all standard YOLO evaluation plots from *val_dir*."""
    for title, filename in _PLOTS.items():
        path = os.path.join(val_dir, filename)
        if not os.path.exists(path):
            print(f"  [WARN] Not found: {path}")
            continue
        img_rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10, 7))
        plt.imshow(img_rgb)
        plt.title(f"{label}  –  {title}" if label else title)
        plt.axis("off")
        plt.show()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main(
    nano_weights: str   = NANO_WEIGHTS,
    medium_weights: str = MEDIUM_WEIGHTS,
    data_yaml: str      = YAML_PATH,
    nano_val_dir: str   = NANO_VAL_DIR,
    medium_val_dir: str = MEDIUM_VAL_DIR,
) -> None:
    nano_metrics   = evaluate_model(nano_weights,   data_yaml)
    medium_metrics = evaluate_model(medium_weights, data_yaml)

    print_metrics(nano_metrics,   "YOLOv8n  (baseline)")
    print_metrics(medium_metrics, "YOLOv8m  (tuned)")

    compare_models(nano_metrics, medium_metrics)

    print("\n── YOLOv8n Evaluation Plots ──")
    display_plots(nano_val_dir,   label="YOLOv8n")

    print("\n── YOLOv8m Evaluation Plots ──")
    display_plots(medium_val_dir, label="YOLOv8m (tuned)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained YOLOv8 food detection models.")
    parser.add_argument("--nano_weights",   default=NANO_WEIGHTS)
    parser.add_argument("--medium_weights", default=MEDIUM_WEIGHTS)
    parser.add_argument("--data_yaml",      default=YAML_PATH)
    parser.add_argument("--nano_val_dir",   default=NANO_VAL_DIR)
    parser.add_argument("--medium_val_dir", default=MEDIUM_VAL_DIR)
    args = parser.parse_args()

    main(
        nano_weights   = args.nano_weights,
        medium_weights = args.medium_weights,
        data_yaml      = args.data_yaml,
        nano_val_dir   = args.nano_val_dir,
        medium_val_dir = args.medium_val_dir,
    )

"""
02_visualization.py
-------------------
Visualizes YOLO bounding boxes on raw images and saves annotated samples
to an output directory.

Usage:
    python src/02_visualization.py  (after running 01_data_loading.py)
"""

import os
import cv2
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def load_yolo_annotation(txt_path: str) -> list[tuple]:
    """
    Parse a YOLO-format annotation file.

    Returns
    -------
    list of (class_id, cx, cy, w, h) in normalised [0-1] coordinates.
    """
    boxes = []
    with open(txt_path, "r") as fh:
        for line in fh:
            parts = line.strip().split()
            cls, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:])
            boxes.append((cls, cx, cy, bw, bh))
    return boxes


def draw_boxes(img_bgr, boxes: list[tuple]) -> "np.ndarray":
    """Draw bounding boxes on a BGR image and return it in RGB."""
    import numpy as np
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W, _ = img_rgb.shape
    for cls, cx, cy, bw, bh in boxes:
        x1 = int((cx - bw / 2) * W)
        y1 = int((cy - bh / 2) * H)
        x2 = int((cx + bw / 2) * W)
        y2 = int((cy + bh / 2) * H)
        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(
            img_rgb, str(cls), (x1, max(y1 - 5, 0)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2,
        )
    return img_rgb


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────
def visualize(
    img_name: str,
    raw_img_dir: str,
    ann_dir: str,
) -> None:
    """Display a single annotated image in a matplotlib figure."""
    img_path = os.path.join(raw_img_dir, img_name)
    txt_path = os.path.join(ann_dir, os.path.splitext(img_name)[0] + ".txt")

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    boxes = load_yolo_annotation(txt_path)
    img_rgb = draw_boxes(img_bgr, boxes)

    plt.figure(figsize=(8, 8))
    plt.imshow(img_rgb)
    plt.axis("off")
    plt.title(img_name)
    plt.show()


def save_samples(
    imgs: list[str],
    raw_img_dir: str,
    ann_dir: str,
    output_dir: str,
    n: int = 10,
) -> None:
    """Save the first *n* annotated images to *output_dir*."""
    os.makedirs(output_dir, exist_ok=True)
    for img_name in imgs[:n]:
        img_path = os.path.join(raw_img_dir, img_name)
        txt_path = os.path.join(ann_dir, os.path.splitext(img_name)[0] + ".txt")

        if not os.path.exists(txt_path):
            print(f"  Skipping {img_name} – no annotation file.")
            continue

        img_bgr = cv2.imread(img_path)
        boxes = load_yolo_annotation(txt_path)
        img_rgb = draw_boxes(img_bgr, boxes)

        save_path = os.path.join(output_dir, img_name)
        cv2.imwrite(save_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

    print(f"Saved {min(n, len(imgs))} annotated samples to: {output_dir}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    BASE         = "/content/drive/MyDrive/Wajahat_Project"
    RAW_IMG_DIR  = "/content/raw_dataset/Raw_Dataset"
    ANN_DIR      = "/content/sample_data/Annotations"
    OUTPUT_DIR   = f"{BASE}/Output Sample"

    img_exts = (".jpg", ".jpeg", ".png")
    imgs = [f for f in os.listdir(RAW_IMG_DIR) if f.lower().endswith(img_exts)]

    # Show one sample
    visualize(imgs[4], RAW_IMG_DIR, ANN_DIR)

    # Save first 10 samples
    save_samples(imgs, RAW_IMG_DIR, ANN_DIR, OUTPUT_DIR)

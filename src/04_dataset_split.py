"""
04_dataset_split.py
-------------------
Splits the preprocessed dataset into train / test subsets (80 / 20 by default)
and copies image–annotation pairs to the expected YOLOv8 directory layout:

    Dataset_Split/
    ├── train/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/

Usage:
    python src/04_dataset_split.py
"""

import os
import random
import shutil

import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
DATASET_DIR    = "/content/drive/MyDrive/Wajahat_Project/Preprocessed_dataset"
BASE_SPLIT_DIR = "/content/drive/MyDrive/Wajahat_Project/Dataset_Split"
SPLIT_RATIO    = 0.80   # fraction used for training
RANDOM_SEED    = 42


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def make_split_dirs(base: str) -> tuple[str, str, str, str]:
    train_img = os.path.join(base, "train", "images")
    train_lbl = os.path.join(base, "train", "labels")
    test_img  = os.path.join(base, "test",  "images")
    test_lbl  = os.path.join(base, "test",  "labels")
    for d in (train_img, train_lbl, test_img, test_lbl):
        os.makedirs(d, exist_ok=True)
    return train_img, train_lbl, test_img, test_lbl


def copy_pairs(
    img_list: list[str],
    src_dir: str,
    img_dest: str,
    lbl_dest: str,
) -> None:
    """Copy each image and its annotation (if present) to the destination dirs."""
    for img_name in tqdm(img_list, desc=f"→ {os.path.basename(img_dest)}"):
        src_img = os.path.join(src_dir, img_name)
        src_lbl = os.path.join(src_dir, img_name.replace(".jpg", ".txt"))
        shutil.copy(src_img, img_dest)
        if os.path.exists(src_lbl):
            shutil.copy(src_lbl, lbl_dest)


def split_dataset(
    dataset_dir: str,
    base_split_dir: str,
    split_ratio: float = SPLIT_RATIO,
    seed: int = RANDOM_SEED,
) -> tuple[list[str], list[str]]:
    """
    Randomly split images into train and test sets and copy them.

    Returns
    -------
    (train_images, test_images) : tuple of filename lists.
    """
    random.seed(seed)
    images = [f for f in os.listdir(dataset_dir) if f.endswith(".jpg")]
    random.shuffle(images)

    split_idx     = int(len(images) * split_ratio)
    train_images  = images[:split_idx]
    test_images   = images[split_idx:]

    train_img_dir, train_lbl_dir, test_img_dir, test_lbl_dir = make_split_dirs(base_split_dir)

    copy_pairs(train_images, dataset_dir, train_img_dir, train_lbl_dir)
    copy_pairs(test_images,  dataset_dir, test_img_dir,  test_lbl_dir)

    print("\nDataset split complete!")
    print(f"  Training images : {len(train_images)}")
    print(f"  Testing  images : {len(test_images)}")

    return train_images, test_images


# ─────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────
def preview_training_samples(
    train_images: list[str],
    train_img_dir: str,
    n: int = 9,
) -> None:
    """Display a grid of the first *n* training images."""
    cols = 3
    rows = (n + cols - 1) // cols
    plt.figure(figsize=(10, 10))
    for i, img_name in enumerate(train_images[:n]):
        img_path = os.path.join(train_img_dir, img_name)
        img_bgr  = cv2.imread(img_path)
        img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img_rgb)
        plt.axis("off")
    plt.suptitle("Sample Augmented Training Images", fontsize=16)
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    train_images, test_images = split_dataset(DATASET_DIR, BASE_SPLIT_DIR)

    TRAIN_IMG_DIR = os.path.join(BASE_SPLIT_DIR, "train", "images")
    preview_training_samples(train_images, TRAIN_IMG_DIR)

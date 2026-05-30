"""
03_preprocessing.py
-------------------
Full preprocessing pipeline:
  1. Remove duplicate images (MD5 hash-based)
  2. Detect and report blurry / low-quality images (Laplacian variance)
  3. Resize all images to 640 × 640 and normalise pixel values
  4. Data augmentation (rotation, horizontal flip, brightness shift)

Usage:
    python src/03_preprocessing.py
"""

import hashlib
import os
import random
import shutil

import cv2
import numpy as np
from tqdm import tqdm


# ─────────────────────────────────────────────
# Configuration  (edit as needed)
# ─────────────────────────────────────────────
RAW_DIR = "/content/raw_dataset/Raw_Dataset"
ANN_DIR = "/content/sample_data/Annotations"
OUT_DIR = "/content/drive/MyDrive/Wajahat_Project/Preprocessed_dataset"
AUG_DIR = os.path.join(OUT_DIR, "augmented")

IMG_SIZE       = 640   # Target spatial resolution (pixels)
BLUR_THRESHOLD = 60    # Laplacian variance below this → blurry


# ─────────────────────────────────────────────
# 1.  Remove duplicate images
# ─────────────────────────────────────────────
def compute_md5(filepath: str) -> str:
    hasher = hashlib.md5()
    with open(filepath, "rb") as fh:
        hasher.update(fh.read())
    return hasher.hexdigest()


def remove_duplicates(raw_dir: str) -> list[str]:
    """
    Detect duplicate images by MD5 hash and delete them in-place.

    Returns the list of removed filenames.
    """
    hashes: dict[str, str] = {}
    duplicates: list[str] = []

    for img in tqdm(os.listdir(raw_dir), desc="Checking duplicates"):
        path = os.path.join(raw_dir, img)
        h = compute_md5(path)
        if h in hashes:
            duplicates.append(img)
        else:
            hashes[h] = img

    for dup in duplicates:
        os.remove(os.path.join(raw_dir, dup))

    print(f"Removed {len(duplicates)} duplicate(s): {duplicates or 'none'}")
    return duplicates


# ─────────────────────────────────────────────
# 2.  Detect blurry images
# ─────────────────────────────────────────────
def is_blurry(image_path: str, threshold: int = BLUR_THRESHOLD) -> tuple[bool, float]:
    """Return (is_blurry, laplacian_variance) for a grayscale image."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    variance = cv2.Laplacian(img, cv2.CV_64F).var()
    return variance < threshold, variance


def flag_blurry_images(raw_dir: str) -> list[str]:
    """List all blurry images; does NOT delete them (manual review recommended)."""
    bad: list[str] = []
    for img in tqdm(os.listdir(raw_dir), desc="Blur detection"):
        path = os.path.join(raw_dir, img)
        blurry, score = is_blurry(path)
        if blurry:
            bad.append(img)
    print(f"Blurry / low-quality images ({len(bad)}): {bad or 'none'}")
    return bad


# ─────────────────────────────────────────────
# 3.  Resize and normalise
# ─────────────────────────────────────────────
def resize_and_normalize(img_bgr, size: int = IMG_SIZE):
    """Resize to (size × size) and normalise to uint8 [0-255]."""
    resized = cv2.resize(img_bgr, (size, size))
    normalised = (resized / 255.0 * 255).astype(np.uint8)
    return normalised


def preprocess_dataset(raw_dir: str, ann_dir: str, out_dir: str) -> None:
    """Resize all images and copy matching annotations to *out_dir*."""
    os.makedirs(out_dir, exist_ok=True)
    for img_name in tqdm(os.listdir(raw_dir), desc="Resizing"):
        if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        img_bgr = cv2.imread(os.path.join(raw_dir, img_name))
        if img_bgr is None:
            continue
        resized = resize_and_normalize(img_bgr)
        cv2.imwrite(os.path.join(out_dir, img_name), resized)

        # Copy annotation if it exists
        base = os.path.splitext(img_name)[0]
        ann_src  = os.path.join(ann_dir, f"{base}.txt")
        ann_dest = os.path.join(out_dir, f"{base}.txt")
        if os.path.exists(ann_src):
            shutil.copy(ann_src, ann_dest)

    print(f"Preprocessed dataset saved to: {out_dir}")


# ─────────────────────────────────────────────
# 4.  Data augmentation
# ─────────────────────────────────────────────
def augment_image(
    img_path: str,
    ann_path: str,
    img_name: str,
    aug_dir: str,
) -> None:
    """
    Apply rotation (±15°), horizontal flip, and brightness shifts (±40)
    to a single image and write all variants + annotations to *aug_dir*.

    Note: YOLO bounding boxes are centre-based and symmetric, so they are
    preserved exactly for centre-locked rotations and brightness changes.
    For horizontal flips the x-coordinate is mirrored: cx_new = 1 - cx.
    """
    img = cv2.imread(img_path)
    H, W = img.shape[:2]

    with open(ann_path, "r") as fh:
        lines = fh.readlines()

    base = os.path.splitext(img_name)[0]

    # -- Rotation -------------------------------------------------------
    for angle in (-15, 15):
        M = cv2.getRotationMatrix2D((W // 2, H // 2), angle, 1.0)
        rotated = cv2.warpAffine(img, M, (W, H))
        suffix = f"_rot{angle:+d}"
        out_img = os.path.join(aug_dir, f"{base}{suffix}.jpg")
        out_ann = os.path.join(aug_dir, f"{base}{suffix}.txt")
        cv2.imwrite(out_img, rotated)
        with open(out_ann, "w") as fh:
            fh.writelines(lines)

    # -- Horizontal flip ------------------------------------------------
    flipped = cv2.flip(img, 1)
    out_img = os.path.join(aug_dir, f"{base}_flip.jpg")
    out_ann = os.path.join(aug_dir, f"{base}_flip.txt")
    cv2.imwrite(out_img, flipped)
    # Mirror cx for each annotation line
    flipped_lines = []
    for line in lines:
        parts = line.strip().split()
        cls, cx, cy, bw, bh = parts[0], float(parts[1]), *map(float, parts[2:])
        flipped_lines.append(f"{cls} {1.0 - cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
    with open(out_ann, "w") as fh:
        fh.writelines(flipped_lines)

    # -- Brightness shift -----------------------------------------------
    for delta in (40, -40):
        bright = np.clip(img.astype(np.int16) + delta, 0, 255).astype(np.uint8)
        out_img = os.path.join(aug_dir, f"{base}_b{delta:+d}.jpg")
        out_ann = os.path.join(aug_dir, f"{base}_b{delta:+d}.txt")
        cv2.imwrite(out_img, bright)
        with open(out_ann, "w") as fh:
            fh.writelines(lines)


def run_augmentation(out_dir: str, aug_dir: str) -> None:
    """Run augmentation on every preprocessed image that has an annotation."""
    os.makedirs(aug_dir, exist_ok=True)
    imgs = [f for f in os.listdir(out_dir) if f.endswith(".jpg")]
    for img_name in tqdm(imgs, desc="Augmenting"):
        img_path = os.path.join(out_dir, img_name)
        ann_path = os.path.join(out_dir, img_name.replace(".jpg", ".txt"))
        if not os.path.exists(ann_path):
            continue
        augment_image(img_path, ann_path, img_name, aug_dir)
    print(f"Augmented images saved to: {aug_dir}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    remove_duplicates(RAW_DIR)
    flag_blurry_images(RAW_DIR)
    preprocess_dataset(RAW_DIR, ANN_DIR, OUT_DIR)
    run_augmentation(OUT_DIR, AUG_DIR)

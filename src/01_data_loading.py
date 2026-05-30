"""
01_data_loading.py
------------------
Mounts Google Drive, extracts raw dataset and annotations from .rar archives,
counts image/annotation pairs, and flags any images with missing annotations.

Usage (Google Colab):
    Run each section sequentially, or import functions into a notebook.
"""

import os
import rarfile
from google.colab import drive


# ─────────────────────────────────────────────
# 1.  Mount Google Drive
# ─────────────────────────────────────────────
def mount_drive():
    drive.mount("/content/drive")
    print("Drive mounted successfully.")


# ─────────────────────────────────────────────
# 2.  Extract archives
# ─────────────────────────────────────────────
def extract_archives(
    raw_zip: str,
    ann_zip: str,
    raw_out: str = "/content/raw_dataset",
    ann_out: str = "/content/sample_data",
) -> tuple[str, str]:
    """
    Extract raw images and annotation archives.

    Parameters
    ----------
    raw_zip : str   Full path to the raw dataset .rar file.
    ann_zip : str   Full path to the annotations .rar file.
    raw_out : str   Destination directory for raw images.
    ann_out : str   Destination directory for annotations.

    Returns
    -------
    (raw_out, ann_out) : tuple[str, str]
    """
    os.makedirs(raw_out, exist_ok=True)
    os.makedirs(ann_out, exist_ok=True)

    for archive, destination in [(raw_zip, raw_out), (ann_zip, ann_out)]:
        try:
            with rarfile.RarFile(archive, "r") as z:
                z.extractall(destination)
            print(f"Extracted {archive} → {destination}")
        except rarfile.BadRarFile as exc:
            print(f"ERROR extracting {archive}: {exc}")

    print("Extraction complete.")
    return raw_out, ann_out


# ─────────────────────────────────────────────
# 3.  Count images and annotations
# ─────────────────────────────────────────────
def count_files(raw_img_dir: str, ann_dir: str) -> None:
    """Print image/annotation counts and list any missing annotation files."""
    img_exts = (".jpg", ".jpeg", ".png")
    imgs = [f for f in os.listdir(raw_img_dir) if f.lower().endswith(img_exts)]
    anns = [f for f in os.listdir(ann_dir) if f.endswith(".txt")]

    print(f"Total Images           : {len(imgs)}")
    print(f"Total Annotation Files : {len(anns)}")

    missing = [
        img for img in imgs
        if f"{os.path.splitext(img)[0]}.txt" not in anns
    ]
    if missing:
        print(f"\nImages missing annotations ({len(missing)}):")
        for m in missing:
            print(f"  {m}")
    else:
        print("\nAll images have matching annotation files. ✓")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    mount_drive()

    BASE = "/content/drive/MyDrive/Wajahat_Project"
    RAW_ZIP = f"{BASE}/01_Raw_Dataset/Raw_Dataset.rar"
    ANN_ZIP = f"{BASE}/02_Annotations/Annotations.rar"

    raw_out, ann_out = extract_archives(RAW_ZIP, ANN_ZIP)

    RAW_IMG_DIR = os.path.join(raw_out, "Raw_Dataset")
    ANN_DIR     = os.path.join(ann_out, "Annotations")

    count_files(RAW_IMG_DIR, ANN_DIR)

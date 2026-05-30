# 🍕 Food Object Detection with YOLOv8

A complete pipeline for detecting 10 food categories using YOLOv8. Trained and evaluated on Google Colab with data stored in Google Drive.

---

## 📋 Table of Contents
- [Project Structure](#project-structure)
- [Classes](#classes)
- [Results](#results)
- [Getting Started](#getting-started)
- [Pipeline Steps](#pipeline-steps)
- [Configuration](#configuration)
- [Model Comparison](#model-comparison)

---

## 📁 Project Structure

```
food_detection_yolov8/
├── configs/
│   └── food.yaml              # Dataset config for YOLOv8
├── notebooks/
│   └── food_detection_pipeline.ipynb   # Full end-to-end Colab notebook
├── src/
│   ├── 01_data_loading.py     # Mount Drive, extract archives, count files
│   ├── 02_visualization.py    # Draw & save YOLO bounding boxes
│   ├── 03_preprocessing.py    # Dedup, blur detection, resize, augmentation
│   ├── 04_dataset_split.py    # 80/20 train-test split
│   ├── 05_train.py            # Train YOLOv8n and YOLOv8m
│   └── 06_evaluate.py         # Validate models, print metrics, show plots
├── docs/
│   └── (add any extra docs here)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🏷️ Classes

| ID | Class         |
|----|---------------|
| 0  | Breads        |
| 1  | Curry         |
| 2  | Pizza         |
| 3  | Rice          |
| 4  | Roti_Chappati |
| 5  | Cake          |
| 6  | Drink         |
| 7  | Egg           |
| 8  | Fruit         |
| 9  | Salad         |

---

## 📊 Results

### Model Comparison (Test Set)

| Metric     | YOLOv8n (baseline) | YOLOv8m (tuned) |
|------------|-------------------|-----------------|
| mAP50      | 0.1409            | **0.3154**      |
| mAP50-95   | 0.0656            | **0.1293**      |

### Class-wise AP50

| Class         | YOLOv8n | YOLOv8m |
|---------------|---------|---------|
| Breads        | 0.1400  | 0.1030  |
| Curry         | 0.1348  | 0.2130  |
| Pizza         | 0.0435  | 0.1553  |
| Rice          | 0.1133  | 0.1910  |
| Roti_Chappati | 0.0121  | 0.2260  |
| Cake          | 0.0436  | 0.0917  |
| Drink         | 0.1444  | 0.1804  |
| Egg           | 0.0028  | 0.0839  |
| Fruit         | 0.0188  | 0.0230  |
| Salad         | 0.0024  | 0.0256  |

> **Key finding:** YOLOv8m (150 epochs) more than doubles the mAP50 of YOLOv8n (50 epochs). Most classes improve significantly; `Breads` is the exception and warrants further investigation.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/food-detection-yolov8.git
cd food-detection-yolov8
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Google Colab** — all dependencies are installed automatically in the first cell of the notebook.

### 3. Prepare your data

Place your data in Google Drive following this structure:

```
MyDrive/
└── Wajahat_Project/
    ├── 01_Raw_Dataset/
    │   └── Raw_Dataset.rar
    └── 02_Annotations/
        └── Annotations.rar
```

### 4. Run the pipeline

**Option A — Notebook (recommended for Colab)**

Open `notebooks/food_detection_pipeline.ipynb` in Google Colab and run all cells sequentially.

**Option B — Python scripts**

```bash
python src/01_data_loading.py
python src/02_visualization.py
python src/03_preprocessing.py
python src/04_dataset_split.py
python src/05_train.py            # trains both models by default
python src/05_train.py --model nano     # train only the baseline
python src/05_train.py --model medium   # train only the tuned model
python src/06_evaluate.py
```

---

## 🔧 Pipeline Steps

| Step | Script | Description |
|------|--------|-------------|
| 1 | `01_data_loading.py`   | Mount Drive, extract `.rar` archives, count image-annotation pairs |
| 2 | `02_visualization.py`  | Draw bounding boxes on raw images; save 10 sample outputs |
| 3 | `03_preprocessing.py`  | Remove duplicates (MD5), flag blurry images (Laplacian variance < 60), resize to 640×640, augment |
| 4 | `04_dataset_split.py`  | Shuffle and split 80/20 into train/test; preview sample images |
| 5 | `05_train.py`          | Train YOLOv8n (50 epochs, batch 16) and YOLOv8m (150 epochs, batch 8) |
| 6 | `06_evaluate.py`       | Validate both models; print and compare metrics; display evaluation plots |

---

## ⚙️ Configuration

Edit `configs/food.yaml` to change dataset paths, class names, or class count:

```yaml
path: /content/drive/MyDrive/Wajahat_Project/Dataset_Split
train: train/images
val:   test/images

nc: 10
names:
  - Breads
  - Curry
  - Pizza
  - Rice
  - Roti_Chappati
  - Cake
  - Drink
  - Egg
  - Fruit
  - Salad
```

---

## 🔍 Augmentation Details

Each image generates 5 augmented variants:
- **Rotation** at −15° and +15°
- **Horizontal flip** (YOLO cx coordinate mirrored: `cx_new = 1 − cx`)
- **Brightness shift** at +40 and −40 (pixel values clamped to [0, 255])

---

## 📌 Next Steps

- Investigate the `Breads` class performance regression in YOLOv8m.
- Collect more examples for `Fruit` (AP50 = 0.023) and `Salad` (AP50 = 0.026).
- Experiment with class-specific data augmentation strategies.
- Try YOLOv8l or YOLOv8x for higher accuracy if compute allows.
- Export to ONNX or TensorRT for production deployment.

---

## 📄 License

This project is for academic/research purposes.

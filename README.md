# ♻️ Recyclable Garbage Classification

Binary classification model that distinguishes **Recyclable** vs **Non-Recyclable** waste using Transfer Learning (MobileNetV2).

---

## Overview

| Item | Detail |
|------|--------|
| Dataset | [Garbage Classification (12 classes)](https://www.kaggle.com/datasets/mostafaabla/garbage-classification) — Kaggle |
| Task | Binary classification (Recyclable / Non-Recyclable) |
| Model | MobileNetV2 (Transfer Learning, ImageNet pretrained) |
| Framework | TensorFlow / Keras |

**Key idea:** The original dataset has 12 fine-grained classes (paper, plastic, metal, glass, etc.). Re-labeled them into 2 categories — Recyclable and Non-Recyclable — to build a practical waste sorting classifier.

---

## Dataset

- Total: 15,150 images across 12 classes
- Re-labeled into 2 categories:

| Category | Classes |
|----------|---------|
| Recyclable | cardboard, paper, plastic, metal, green-glass, brown-glass, white-glass |
| Non-Recyclable | battery, biological, clothes, shoes, trash |

---

## Results

### ✅ Final Model (MobileNetV2 + Transfer Learning)

| Metric | Score |
|--------|-------|
| Accuracy | **89%** |
| Precision (Recyclable) | 92% |
| Recall (Non-Recyclable) | 95% |
| F1-Score (Weighted Avg) | **0.89** |

### ❌ Baseline (Simple CNN — Overfitting)

| Metric | Train | Validation |
|--------|-------|------------|
| Accuracy | 98.64% | 56.03% |
| Loss | 0.032 | 2.376 |

Train accuracy 98% vs Validation 56% — classic overfitting.  
**Solution:** Switched to MobileNetV2 with frozen base layers + Data Augmentation + Dropout(0.5).

---

## Model Architecture

```
MobileNetV2 (pretrained on ImageNet, frozen)
    ↓
GlobalAveragePooling2D
    ↓
Dense(128, relu)
    ↓
Dropout(0.5)
    ↓
Dense(1, sigmoid)  →  Recyclable / Non-Recyclable
```

---

## Key Implementation Details

- **Preprocessing:** MobileNetV2-specific `preprocess_input` (scales to [-1, 1])
- **Data Augmentation:** rotation, shift, shear, zoom, horizontal flip
- **Optimizer:** Adam (lr=0.0001)
- **Loss:** Binary Crossentropy
- **Early Stopping:** patience=5, monitor=val_loss
- **Input size:** 160×160

---

## Project Structure

```
├── 재활용쓰레기분류.py   # Full training pipeline
└── README.md
```

---

## How to Run

1. Download dataset from [Kaggle](https://www.kaggle.com/datasets/mostafaabla/garbage-classification)
2. Organize into `Recyclable/` and `Non_Recyclable/` folders
3. Run on Google Colab or local environment

```python
# Set data directory
data_dir = 'path/to/garbage_classification'

# Then run 재활용쓰레기분류.py
```

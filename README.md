# Visual Grounding with PyTorch

This repository contains PyTorch implementations for **visual grounding** — the task of localizing an object in an image based on a natural language expression.

---

## Table of Contents

1. [Overview](#overview)  
2. [Dataset](#dataset)  
3. [Installation](#installation)  
4. [Model Architectures](#model-architectures)  
5. [Training & Evaluation](#training--evaluation)  
6. [Functions](#functions)  
7. [Acknowledgements](#acknowledgements)

---

## Overview

Visual grounding involves predicting the bounding box of an object in an image, given a referring expression (caption).

This repository includes:

- A **baseline model**: `VisualGroundingModel`
- A **developing model**: `ReferringGroundingModel`

---

## Dataset

The models are trained and evaluated on the **RefCOCO** dataset.

You need the following:

- `train2014/` — COCO training images  
- `val2014/` — COCO validation images  
- `instances.json` — COCO annotation file (train)  
- `refs(unc).p` — RefCOCO expressions (train)  
- `instances_val.json` — COCO annotation file (val)  
- `refs(unc)_val.p` — RefCOCO expressions (val)

### Download Links

- **COCO Images**: [https://cocodataset.org/#download](https://cocodataset.org/#download)  
- **RefCOCO**: [https://github.com/lichengunc/refer](https://github.com/lichengunc/refer)

---

## Installation

Install dependencies with:

```bash
pip install -r requirements.txt
```

---

## Model Architectures

### 🔹 Baseline — `VisualGroundingModel` (`model_visual_grounding.py`)

- **Image Encoder**: MobileNetV3 (pretrained)  
- **Text Encoder**: BERT (`bert-base-uncased`)  
- **Fusion**: Transformer decoder (text queries, image keys/values)  
- **Output**: `[x1, y1, x2, y2]` bounding box  

### 🔸 Developing — `ReferringGroundingModel` (`model_referring_grounding.py`)

- **Image Encoder**: MobileNetV2  
- **Text Encoder**: DistilBERT  
- **Fusion**: Concatenation + heatmap  
- **Prediction**:
  - Heatmap to find object center  
  - Regression head for width/height  
  - Final output: `[x1, y1, x2, y2]` box

---

## Training & Evaluation

### 🏋️‍♂️ Training (`scripts/train.py`)

- Loads data from `train2014/`
- Splits a small portion for internal validation
- Saves model to `models/visual_grounding_model.pth`

**Set paths:**

```python
image_dir = 'data/train2014'
annotation_file = 'data/instances.json'
refexp_file = 'data/refs(unc).p'
```

---

### 🧪 Evaluation (`scripts/evaluate.py`)

- Uses `val2014` as a completely separate evaluation dataset
- Computes IoU, Precision, Recall, mAP

**Set paths:**

```python
image_dir = 'data/val2014'
annotation_file = 'data/instances_val.json'
refexp_file = 'data/refs(unc)_val.p'
```

---



---

## Acknowledgements

- [RefCOCO Dataset](https://github.com/lichengunc/refer)  
- [COCO Dataset](https://cocodataset.org)  
- [HuggingFace Transformers](https://huggingface.co/transformers/)  
- [Torchvision Models](https://pytorch.org/vision/stable/models.html)
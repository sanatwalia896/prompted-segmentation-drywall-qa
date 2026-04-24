# 🧠 Prompted Segmentation for Drywall QA

## 📌 Overview
This project implements a **text-conditioned image segmentation system** using CLIPSeg.

Given an image and a natural-language prompt, the model generates a binary segmentation mask.

### 🎯 Tasks
- Segment drywall taping areas  
- Segment surface cracks  

---

## 🧠 Model
- **CLIPSeg (CIDAS/clipseg-rd64-refined)**
- Frozen CLIP encoder  
- Fine-tuned segmentation decoder  

---

## 📂 Datasets

| Dataset | Type | Description |
|--------|------|------------|
| Drywall Join | YOLO → Mask | Taping areas |
| Crack Dataset | COCO Segmentation | Surface cracks |

### ⚙️ Preprocessing
- YOLO bounding boxes → converted to masks  
- COCO polygons → converted to masks  
- All images resized to **352×352**

---

## 🧪 Training

- Loss: **BCE + Dice Loss**
- Optimizer: **AdamW**
- Learning Rate: `3e-4`
- Scheduler: Cosine Annealing
- Epochs: 15
- Early stopping (patience = 3)

---

## 📊 Results

| Task | mIoU | Dice |
|------|------|------|
| Drywall | 0.621 | 0.756 |
| Crack   | 0.511 | 0.661 |
| **Overall** | **0.566** | **0.708** |

---

## ⚡ Runtime

- Inference: **~22 ms/image**
- Training time: ~52 min
- Trainable params: ~1.1M

---

## 🖼 Sample Outputs

| Input | Ground Truth | Prediction |
|------|-------------|-----------|
| ✔ | ✔ | ✔ |

*(See `outputs/` folder)*

---

## 🚀 Streamlit Demo

Run locally:

```bash
streamlit run app/app.py
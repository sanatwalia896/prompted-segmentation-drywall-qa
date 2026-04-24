
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

## 📦 Model Weights

This project uses **two components**:

### 1️⃣ Base Model (Auto Downloaded)
The base CLIPSeg model is automatically downloaded from Hugging Face:

```

CIDAS/clipseg-rd64-refined (~600MB)

```

This happens automatically the first time you run the app.

---

### 2️⃣ Fine-tuned Weights (This Project)

Your trained model weights are stored as:

```

models/best_clipseg.pth

```

These weights are loaded on top of the base model to produce the final segmentation model.

---

### 🔁 Loading Flow

```

Download base model → Load architecture → Apply fine-tuned weights

````

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
````

### Example prompts:

* `segment crack`
* `segment taping area`

---

## 📁 Project Structure

```
├── app/
│   └── app.py
├── models/
│   └── best_clipseg.pth
├── notebooks/
│   └── training.ipynb
├── outputs/
│   └── samples.png
├── requirements.txt
└── README.md
```

---

## 🧠 Key Insights

* Model learns **semantic structures beyond labels**
* Drywall performs better due to structured geometry
* Crack segmentation is harder due to fine details
* YOLO → mask conversion introduces label noise

---

## ⚠️ Limitations

* Coarse masks for drywall dataset
* Slight over-segmentation
* Thin cracks sometimes missed

---

## 🔮 Future Work

* Use true segmentation labels
* Improve prompt engineering
* Add post-processing

---
## 📦 Model Weights

Due to file size limitations, the fine-tuned model is hosted externally.

### 🔽 Download Model
Download from Google Drive:
👉 [Download best_clipseg.pth](https://drive.google.com/file/d/1s4vOjPfi1dC9n-0jXVSDQyoZeiXO6rLU/view?usp=share_link)

After downloading, place it in:

## 👨‍💻 Author

Sanat Walia


---
import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import os

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_NAME = "CIDAS/clipseg-rd64-refined"
MODEL_PATH = "models/best_clipseg.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    processor = CLIPSegProcessor.from_pretrained(MODEL_NAME)
    model = CLIPSegForImageSegmentation.from_pretrained(MODEL_NAME)

    if os.path.exists(MODEL_PATH):
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
    else:
        st.warning("⚠️ Custom model not found. Using base CLIPSeg.")

    model.to(DEVICE)
    model.eval()

    return processor, model


processor, model = load_model()


# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
st.title("🧠 Prompted Segmentation Demo")
st.write("Enter a prompt like:")
st.write("👉 segment crack")
st.write("👉 segment taping area")

uploaded = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
prompt = st.text_input("Enter Prompt", "segment crack")


# ─────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    image_np = np.array(image)

    st.image(image, caption="Input Image", use_container_width=True)

    if st.button("Run Segmentation"):

        # Encode input
        enc = processor(
            text=prompt,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=20
        )

        pixel_values = enc["pixel_values"].to(DEVICE)
        input_ids    = enc["input_ids"].to(DEVICE)
        attn_mask    = enc["attention_mask"].to(DEVICE)

        # Forward pass
        with torch.no_grad():
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attn_mask,
            )

        logits = outputs.logits
        if logits.dim() == 4:
            logits = logits.squeeze(1)

        mask = torch.sigmoid(logits).cpu().numpy()[0]

        # ───────── FIXED RESIZE (CRITICAL) ─────────
        h, w = image_np.shape[:2]
        mask = cv2.resize(mask, (w, h))

        # Binary mask
        mask_bin = (mask > 0.5).astype(np.uint8) * 255

        # ───────── OVERLAY ─────────
        overlay = image_np.copy()
        overlay[mask_bin > 0] = [255, 0, 0]

        blended = cv2.addWeighted(image_np, 0.7, overlay, 0.3, 0)

        # ───────── DISPLAY ─────────
        st.subheader("Results")

        col1, col2 = st.columns(2)

        with col1:
            st.image(mask_bin, caption="Predicted Mask")

        with col2:
            st.image(blended, caption="Overlay")

        # ───────── DOWNLOAD ─────────
        mask_pil = Image.fromarray(mask_bin)
        st.download_button(
            label="Download Mask",
            data=mask_pil.tobytes(),
            file_name="mask.png",
            mime="image/png"
        )
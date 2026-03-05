# app.py
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import os

st.set_page_config(page_title="Pneumonia Detector (Demo)", layout="centered")

st.title("🩺 Pneumonia Detector — Demo")
st.write("Upload a chest X-ray image. If you have a trained model file named `best_pneumonia_model.h5` (or `pneumonia_model.h5`) in this folder, the app will use it. Otherwise it will simulate a prediction.")

MODEL_PATHS = ["best_pneumonia_model.h5", "pneumonia_model.h5"]
model = None
for p in MODEL_PATHS:
    if os.path.exists(p):
        try:
            model = tf.keras.models.load_model(p)
            st.success(f"Loaded model: {p}")
            break
        except Exception as e:
            st.warning(f"Found {p} but failed to load it: {e}")
            model = None

def predict_from_model(img_arr):
    # img_arr: HxW x3 float32 in [0,1], resize to model expected size
    # Try common sizes; default resize to 150x150
    try_sizes = [(150,150), (128,128), (224,224)]
    for s in try_sizes:
        try:
            x = cv2.resize(img_arr, s)
            x = np.expand_dims(x, axis=0).astype(np.float32)
            preds = model.predict(x)
            # model may output single sigmoid neuron or 2-class softmax
            if preds.ndim == 2 and preds.shape[1] == 1:
                score = float(preds[0][0])
            elif preds.ndim == 2 and preds.shape[1] == 2:
                # assume [prob_normal, prob_pneumonia]
                score = float(preds[0][1])
            else:
                score = float(preds.ravel()[-1])
            return score
        except Exception:
            continue
    # fallback
    return None

def map_severity(conf_pct):
    # conf_pct is pneumonia confidence in 0..100
    if conf_pct < 60:
        return "Mild"
    elif conf_pct < 75:
        return "Mild"
    elif conf_pct < 90:
        return "Moderate"
    else:
        return "Severe"

uploaded_file = st.file_uploader("Choose a chest X-ray image (jpg/png)", type=["jpg","jpeg","png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-ray", use_column_width=True)

    # prepare image array
    img = np.array(image)
    img_proc = img.astype("float32") / 255.0

    # If model available, use it; otherwise random demo
    if model is not None:
        score = predict_from_model(img_proc)
        if score is None:
            st.error("Model couldn't make a prediction for this image size. Please provide a typical chest X-ray (150x150/224x224).")
        else:
            pneumonia_conf = score * 100.0
            normal_conf = 100.0 - pneumonia_conf
            if pneumonia_conf >= 50:
                severity = map_severity(pneumonia_conf)
                st.error(f"🔴 Pneumonia detected — {severity}")
                st.write(f"Confidence (Pneumonia): **{pneumonia_conf:.2f}%**")
            else:
                st.success("🟢 Normal")
                st.write(f"Confidence (Normal): **{normal_conf:.2f}%**")
    else:
        # dummy behaviour: use simple image brightness heuristic + randomness
        mean_int = img_proc.mean()
        # brighter images we'll treat as normal more often (this is arbitrary demo logic)
        base = (0.5 - (mean_int - 0.5))  # near 0.5 -> base 0.5
        rand = np.random.RandomState(int(mean_int*1e6) % 100000)
        pneumonia_conf = np.clip((rand.rand() * 0.4 + (1 - base)*0.5) * 100, 25, 98)
        if pneumonia_conf >= 50:
            severity = map_severity(pneumonia_conf)
            st.error(f"🔴 Pneumonia detected — {severity}")
            st.write(f"Confidence (Pneumonia): **{pneumonia_conf:.2f}%**  _(demo)_")
        else:
            st.success("🟢 Normal")
            st.write(f"Confidence (Normal): **{100.0 - pneumonia_conf:.2f}%**  _(demo)_")

    st.markdown("---")
    st.info("To use a real model: put your trained model file `best_pneumonia_model.h5` in this same folder. The app will automatically load it on start.")

    # optional: save prediction image locally
    if st.button("Save labeled image"):
        import datetime
        label = "PNEUMONIA" if (model is not None and pneumonia_conf>=50) or (model is None and pneumonia_conf>=50) else "NORMAL"
        out = image.copy()
        import PIL.ImageDraw as ImageDraw, PIL.ImageFont as ImageFont
        draw = ImageDraw.Draw(out)
        txt = f"{label} - {pneumonia_conf:.2f}%" if label=="PNEUMONIA" else f"{label} - {100.0-pneumonia_conf:.2f}%"
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        draw.text((10,10), txt, fill=(255,0,0), font=font)
        fname = f"pred_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        out.save(fname)
        st.success(f"Saved as {fname}")

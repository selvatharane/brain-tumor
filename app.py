import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# --- Load Models ---
ct_model_path = "ct_small_cnn.keras"
mri_model_path = "mri_effb3_finetuned.keras"

ct_model = tf.keras.models.load_model(ct_model_path)
mri_model = tf.keras.models.load_model(mri_model_path)

st.title("🧠 Brain Scan Tumor Detection App")

# --- Upload Image ---
uploaded_file = st.file_uploader("Upload an MRI or CT Scan Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Always ensure RGB
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    # --- Detect Scan Type ---
    mean_intensity = np.mean(img_array)
    scan_type = "CT" if mean_intensity < 100 else "MRI"
    st.write(f"🩺 Detected Scan Type: **{scan_type}**")

    # --- Preprocess Image ---
    # Resize according to model input
    if scan_type == "CT":
        target_size = (128, 128)  # Adjust to your CT model input
    else:
        target_size = (300, 300)  # Adjust to your MRI model input (EfficientNet usually 300x300)

    img_resized = cv2.resize(img_array, target_size)

    # Ensure 3 channels
    if len(img_resized.shape) == 2:
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)

    img_resized = img_resized / 255.0
    img_input = np.expand_dims(img_resized, axis=0)

    # --- Prediction ---
    if scan_type == "CT":
        pred = ct_model.predict(img_input)
        result = "🧠 Tumor Detected" if pred[0][0] > 0.5 else "✅ No Tumor"
        st.subheader(f"CT Result: {result}")

    else:
        pred = mri_model.predict(img_input)
        classes = ["Meningioma", "Glioma", "Pituitary", "No Tumor"]
        result = classes[np.argmax(pred)]
        st.subheader(f"MRI Result: {result}")

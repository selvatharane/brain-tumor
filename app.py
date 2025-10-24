import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# Model paths
ct_model_path = "ct_small_cnn.keras"
mri_model_path = "mri_effb3_finetuned.keras"

# Load models
ct_model = tf.keras.models.load_model(ct_model_path)
mri_model = tf.keras.models.load_model(mri_model_path)

st.title("🧠 Brain Scan Tumor Detection App")

uploaded_file = st.file_uploader("Upload MRI or CT image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")  # Force 3 channels
    img_array = np.array(image)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Detect if MRI or CT (simple brightness-based heuristic)
    mean_intensity = np.mean(img_array)
    scan_type = "CT" if mean_intensity < 100 else "MRI"

    st.write(f"🩺 Detected Scan Type: **{scan_type}**")

    # Preprocess
    img_resized = cv2.resize(img_array, (300, 300))
    img_resized = img_resized / 255.0
    img_input = np.expand_dims(img_resized, axis=0)

    if scan_type == "CT":
        pred = ct_model.predict(img_input)
        result = "Tumor Detected" if pred[0][0] > 0.5 else "No Tumor"
        st.write(f"🧬 **CT Result:** {result}")
    else:
        pred = mri_model.predict(img_input)
        classes = ["Meningioma", "Glioma", "Pituitary", "No Tumor"]
        result = classes[np.argmax(pred)]
        st.write(f"🧬 **MRI Result:** {result}")

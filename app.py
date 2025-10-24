import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

st.set_option('deprecation.showfileUploaderEncoding', False)

# --- Load models once and cache ---
@st.cache_resource
def load_models():
    ct_model = tf.keras.models.load_model("ct_small_cnn.keras")
    mri_model = tf.keras.models.load_model("mri_effb3_finetuned.keras")
    return ct_model, mri_model

ct_model, mri_model = load_models()

# --- App Title ---
st.title("🧠 Brain Scan Tumor Detection App")
st.write("Upload an MRI or CT scan — the app detects the type and predicts tumor results.")

# --- Upload Image ---
uploaded_file = st.file_uploader("📤 Upload MRI or CT Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Open and convert to RGB
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.write(f"🧩 Original image shape: {img_array.shape}")

    # --- Detect scan type ---
    mean_intensity = np.mean(img_array)
    scan_type = "CT" if mean_intensity < 100 else "MRI"
    st.info(f"Detected scan type: **{scan_type}**")

    # --- Resize ---
    target_size = (128, 128) if scan_type == "CT" else (300, 300)
    img_resized = cv2.resize(img_array, target_size, interpolation=cv2.INTER_AREA)

    # --- Ensure 3 channels ---
    if len(img_resized.shape) == 2:
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
    elif img_resized.shape[-1] == 1:
        img_resized = np.repeat(img_resized, 3, axis=-1)

    st.write(f"✅ Final image shape before prediction: {img_resized.shape}")

    # --- Normalize & expand dims ---
    img_input = np.expand_dims(img_resized.astype(np.float32)/255.0, axis=0)

    # --- Predict ---
    if scan_type == "CT":
        pred = ct_model.predict(img_input)
        result = "🧠 Tumor Detected" if pred[0][0] > 0.5 else "✅ No Tumor"
        st.subheader(f"CT Result: {result}")
    else:
        pred = mri_model.predict(img_input)
        classes = ["Meningioma", "Glioma", "Pituitary", "No Tumor"]
        result = classes[np.argmax(pred)]
        st.subheader(f"MRI Result: {result}")

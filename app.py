import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# Cache models so they load once
@st.cache_resource
def load_models():
    ct_model = tf.keras.models.load_model("ct_small_cnn.keras")
    mri_model = tf.keras.models.load_model("mri_effb3_finetuned.keras")
    return ct_model, mri_model

ct_model, mri_model = load_models()

st.title("🧠 Brain Scan Tumor Detection App")
st.write("Upload either an MRI or CT scan image below. The app will automatically detect the scan type and classify accordingly.")

uploaded_file = st.file_uploader("📤 Upload MRI or CT image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Always open image as RGB (3 channels)
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Simple heuristic to detect MRI or CT
    mean_intensity = np.mean(img_array)
    scan_type = "CT" if mean_intensity < 100 else "MRI"
    st.info(f"Detected scan type: **{scan_type}**")

    # Resize image according to model requirement
    if scan_type == "CT":
        target_size = (128, 128)  # Adjust if your CT model expects a different size
    else:
        target_size = (300, 300)  # EfficientNet default size

    img_resized = cv2.resize(img_array, target_size)

    # Ensure image has 3 channels (convert if grayscale)
    if len(img_resized.shape) == 2 or img_resized.shape[-1] == 1:
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)

    # Normalize and prepare
    img_resized = img_resized / 255.0
    img_input = np.expand_dims(img_resized, axis=0)

    # --- Make Prediction ---
    if scan_type == "CT":
        pred = ct_model.predict(img_input)
        result = "🧠 Tumor Detected" if pred[0][0] > 0.5 else "✅ No Tumor"
        st.subheader(f"CT Result: {result}")

    else:
        pred = mri_model.predict(img_input)
        classes = ["Meningioma", "Glioma", "Pituitary", "No Tumor"]
        result = classes[np.argmax(pred)]
        st.subheader(f"MRI Result: {result}")

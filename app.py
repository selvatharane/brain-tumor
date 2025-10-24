import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image

# Load models once
@st.cache_resource
def load_models():
    ct_model = tf.keras.models.load_model("ct_small_cnn.keras")
    mri_model_top = tf.keras.models.load_model("mri_effb3_top.keras")
    mri_model_finetuned = tf.keras.models.load_model("mri_effb3_finetuned.keras")
    return ct_model, mri_model_top, mri_model_finetuned

ct_model, mri_model_top, mri_model_finetuned = load_models()

st.title("🧠 Brain Tumor Detection System")
st.write("Upload a **CT or MRI** scan, and the system will automatically detect the scan type and predict the result.")

uploaded_file = st.file_uploader("Upload a brain scan image", type=["jpg", "jpeg", "png"])

# Function to preprocess the image
def preprocess_image(image, target_size):
    image = image.resize(target_size)
    image = np.array(image)
    if len(image.shape) == 2:  # grayscale → RGB
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Simple scan type detection (based on brightness and texture)
def detect_scan_type(image):
    img_gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    mean_intensity = np.mean(img_gray)
    # Heuristic: CT scans tend to be higher contrast & lighter
    if mean_intensity > 100:
        return "CT"
    else:
        return "MRI"

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    scan_type = detect_scan_type(image)
    st.info(f"Detected Scan Type: **{scan_type} Scan**")

    if st.button("Analyze"):
        if scan_type == "CT":
            img_processed = preprocess_image(image, target_size=(128, 128))
            prediction = ct_model.predict(img_processed)
            label = "Tumor" if prediction[0][0] > 0.5 else "No Tumor"
            st.success(f"🩻 **CT Scan Result:** {label}")

        elif scan_type == "MRI":
            img_processed = preprocess_image(image, target_size=(224, 224))
            # Combine both MRI models for better accuracy (optional)
            preds_top = mri_model_top.predict(img_processed)
            preds_fine = mri_model_finetuned.predict(img_processed)
            preds_avg = (preds_top + preds_fine) / 2

            classes = ["Meningioma", "Glioma", "Pituitary", "No Tumor"]
            label = classes[np.argmax(preds_avg)]
            st.success(f"🧬 **MRI Scan Result:** {label}")

        st.balloons()

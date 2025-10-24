import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import io

# --- Load Models ---
ct_model = tf.keras.models.load_model("ct_small_cnn.keras")
mri_model = tf.keras.models.load_model("mri_effb3_finetuned.keras")

# --- Class Labels ---
ct_classes = ['No Tumor', 'Tumor']
mri_classes = ['Glioma', 'Meningioma', 'Pituitary', 'No Tumor']

# --- Function to Preprocess Image ---
def preprocess_image(img, target_size=(224, 224)):
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    if img_array.shape[-1] == 4:  # Handle RGBA
        img_array = img_array[..., :3]
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# --- Function to Detect Scan Type (CT or MRI) ---
def detect_scan_type(image):
    """
    Simple heuristic: MRI scans tend to be darker with circular brain regions,
    while CT scans are brighter and have a wider intensity distribution.
    You can refine this with a small classifier if needed.
    """
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    mean_intensity = np.mean(gray)
    if mean_intensity < 100:
        return "MRI"
    else:
        return "CT"

# --- Streamlit UI ---
st.set_page_config(page_title="Brain Scan Classifier", layout="centered")
st.title("🧠 Brain Tumor Classification App")

st.write("Upload a **CT** or **MRI** brain scan image, and the model will automatically:")
st.markdown("""
- Detect the scan type (CT / MRI)
- Classify:
  - **CT:** Tumor / No Tumor  
  - **MRI:** Glioma / Meningioma / Pituitary / No Tumor
""")

uploaded_file = st.file_uploader("📤 Upload a Brain Scan Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Scan", use_container_width=True)

    # Detect scan type
    scan_type = detect_scan_type(image)
    st.info(f"Detected Scan Type: **{scan_type}**")

    # Preprocess image
    img_array = preprocess_image(image, target_size=(224, 224))

    # Predict based on scan type
    if scan_type == "CT":
        preds = ct_model.predict(img_array)
        result = ct_classes[np.argmax(preds)]
        confidence = np.max(preds) * 100
        st.success(f"**CT Result:** {result} ({confidence:.2f}% confidence)")

    elif scan_type == "MRI":
        preds = mri_model.predict(img_array)
        result = mri_classes[np.argmax(preds)]
        confidence = np.max(preds) * 100
        st.success(f"**MRI Result:** {result} ({confidence:.2f}% confidence)")

    # Optional: show probability chart
    with st.expander("🔍 View Model Probabilities"):
        if scan_type == "CT":
            for label, prob in zip(ct_classes, preds[0]):
                st.write(f"{label}: {prob*100:.2f}%")
        else:
            for label, prob in zip(mri_classes, preds[0]):
                st.write(f"{label}: {prob*100:.2f}%")

else:
    st.warning("Please upload a CT or MRI scan image to continue.")

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -------------------------------
# Function: Load models safely
# -------------------------------
@st.cache_resource
def load_models():
    try:
        ct_model = tf.keras.models.load_model("ct_model.h5")
        st.success("✅ CT model loaded successfully.")
    except Exception as e:
        ct_model = None
        st.error(f"⚠️ Failed to load CT model: {e}")

    try:
        # safe_mode=False allows loading even if minor shape mismatches exist
        mri_model = tf.keras.models.load_model("mri_model.h5", safe_mode=False)
        st.success("✅ MRI model loaded successfully.")
    except Exception as e:
        mri_model = None
        st.error(f"⚠️ Failed to load MRI model: {e}")

    return ct_model, mri_model

# Load both models
ct_model, mri_model = load_models()

# -------------------------------
# Function: Preprocess the image
# -------------------------------
def preprocess_image(image, target_size):
    # Ensure RGB mode (3 channels)
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# -------------------------------
# Function: Predict image
# -------------------------------
def predict(model, image_array):
    prediction = model.predict(image_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    return predicted_class

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Brain Scan Classifier", page_icon="🧠", layout="centered")

st.title("🧠 Brain Scan Classification (CT & MRI)")

st.write("Upload a **CT** or **MRI** brain scan image to classify.")

option = st.radio("Select Scan Type:", ["CT", "MRI"])
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

# Show model input shapes (for debugging)
if ct_model:
    st.write("📏 CT Model Input Shape:", ct_model.input_shape)
if mri_model:
    st.write("📏 MRI Model Input Shape:", mri_model.input_shape)

# -------------------------------
# Main Prediction Section
# -------------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="🩻 Uploaded Image", use_column_width=True)

    if st.button("🔍 Classify"):
        if option == "CT" and ct_model is not None:
            target_size = ct_model.input_shape[1:3]
            image_array = preprocess_image(image, target_size)
            predicted_class = predict(ct_model, image_array)
            st.success(f"✅ CT Prediction: Class {predicted_class}")

        elif option == "MRI" and mri_model is not None:
            target_size = mri_model.input_shape[1:3]
            image_array = preprocess_image(image, target_size)
            predicted_class = predict(mri_model, image_array)
            st.success(f"✅ MRI Prediction: Class {predicted_class}")

        else:
            st.error("⚠️ Model not loaded. Please check your model files.")

st.write("---")
st.caption("Developed by Selva Dharani | Powered by Streamlit + TensorFlow")

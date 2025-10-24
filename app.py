import os
import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image

# Helper function to build the correct path
def get_model_path(filename):
    return os.path.join(os.path.dirname(__file__), filename)

# Load Keras models
ct_model_path = get_model_path("ct_small_cnn (1).keras")
mri_finetuned_path = get_model_path("mri_effb3_finetuned (1).keras")
mri_top_path = get_model_path("mri_effb3_top (2).keras")

ct_model = tf.keras.models.load_model(ct_model_path)
mri_finetuned_model = tf.keras.models.load_model(mri_finetuned_path)
mri_top_model = tf.keras.models.load_model(mri_top_path)

st.title("Brain Tumor Classification App")
st.write("Models loaded successfully!")

# Example: Display uploaded image
uploaded_file = st.file_uploader("Upload a brain scan image", type=["png", "jpg", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)



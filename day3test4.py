import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import requests
import tempfile
import os

# Download model from Google Drive
@st.cache_resource
def load_model_from_drive():
    url = "https://drive.google.com/uc?export=download&id=16LjFwzLXKLey7p_zuC4-ZLDmFpUuckT3"
    response = requests.get(url)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".h5")
    temp_file.write(response.content)
    temp_file.close()
    return load_model(temp_file.name)

# Load model
model = load_model_from_drive()

# Class labels
labels = {
    0: "Cloudy",
    1: "Desert",
    2: "Green_Area",
    3: "Water"
}

# Streamlit UI
st.set_page_config(page_title="🌍 Satellite Image Classifier", layout="centered")
st.title("🛰️ Satellite Image Classifier")
st.markdown("Upload a satellite image and let AI predict the scene type!")

# Upload section
uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocessing
    img = image.resize((256, 256))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    predicted_class = labels[np.argmax(prediction)]

    st.success(f"🌟 Prediction: **{predicted_class}**")
    st.bar_chart(prediction[0])
else:
    st.info("Please upload a satellite image to get started.")

st.markdown("---")
st.caption("🔍 Model loaded directly from Google Drive.")

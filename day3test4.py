import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

# Load model
model = load_model("Modelenv.v1.h5")

# Class labels
labels = {
    0: "Cloudy",
    1: "Desert",
    2: "Green_Area",
    3: "Water"
}

# App title and layout
st.set_page_config(page_title="🌍 Satellite Image Classifier", layout="centered")
st.title("🌈 Satellite Image Classification")
st.markdown("Upload a satellite image and let AI guess the type!")

# Upload image
uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image.resize((256, 256))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = labels[np.argmax(prediction)]

    st.success(f"🌟 Prediction: **{predicted_class}**")
    st.bar_chart(prediction[0])
else:
    st.info("Please upload a satellite image to begin.")

st.markdown("---")
st.caption("🛰️ Powered by a CNN model trained on satellite image data.")

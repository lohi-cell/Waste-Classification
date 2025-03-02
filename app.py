import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("waste_classification_model.h5")

model = load_model()

# Define class labels
class_labels = ["♻️ Organic Waste", "🔄 Recyclable Waste"]

# Function to preprocess image
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to predict waste category
def predict_waste(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction)
    return class_labels[predicted_class], prediction[0][predicted_class] * 100

# Streamlit UI
st.set_page_config(page_title="Waste Classifier", page_icon="♻️", layout="wide")

# Title with emoji
st.markdown("<h1 style='text-align: center; color: green;'>♻️ Waste Classification using CNN</h1>", unsafe_allow_html=True)
st.write("<h4 style='text-align: center;'>Upload an image to classify it as Organic or Recyclable Waste.</h4>", unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("📂 **Choose an image...**", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(image, caption="🖼 Uploaded Image", use_column_width=True)
    
    with col2:
        st.write("⏳ **Analyzing... Please wait**")
        progress_bar = st.progress(0)

        for percent in range(100):
            progress_bar.progress(percent + 1)
        
        # Perform prediction
        label, confidence = predict_waste(image)

        # Display results
        st.markdown(f"<h3>🏷 Prediction: {label}</h3>", unsafe_allow_html=True)
        st.markdown(f"<h4>🔍 Confidence: {confidence:.2f}%</h4>", unsafe_allow_html=True)

        # Add a progress visualization
        st.progress(confidence / 100)


import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown

# Google Drive file ID (replace with your file ID)
file_id = "1NGgnO4nTFRRk55z7fgm_eFl6kgeLXr-1"
model_path = "waste_classification_model.h5"

# Download the model if it doesn’t exist
@st.cache_resource
def load_model():
    gdown.download(f"https://drive.google.com/uc?id={file_id}", model_path, quiet=False)
    return tf.keras.models.load_model(model_path)

# Load the trained model
model = load_model()

# Define class labels (modify based on your dataset)
class_labels = ["Organic Waste", "Recyclable Waste"]

def preprocess_image(image):
    """Preprocess the image for prediction."""
    image = image.resize((224, 224))  # Resize to match model input
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def predict_waste(image):
    """Predict the waste category."""
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction)
    return class_labels[predicted_class], prediction[0][predicted_class] * 100

# Streamlit UI
st.title("♻️ Waste Classification using CNN")
st.write("Upload an image to classify it as **Organic** or **Recyclable** Waste.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Perform prediction
    label, confidence = predict_waste(image)
    
    st.success(f"### 🏷 Prediction: **{label}**")
    st.info(f"### 🔍 Confidence: **{confidence:.2f}%**")

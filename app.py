import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import time  # For loading animation

# Set Streamlit page config
st.set_page_config(page_title="Waste Classification", page_icon="♻️", layout="wide")

# Sidebar - About the Model
st.sidebar.title("🔍 About the Model")
st.sidebar.info(
    "This AI model classifies waste into:\n"
    "- **Organic Waste** 🍃\n"
    "- **Recyclable Waste** 🔄\n\n"
    "**Upload an image to see the classification!**"
)

# Google Drive file ID (replace with your file ID)
file_id = "1NGgnO4nTFRRk55z7fgm_eFl6kgeLXr-1"
model_path = "waste_classification_model.h5"

# Download and cache the model
@st.cache_resource
def load_model():
    with st.spinner("🔄 Loading model..."):
        gdown.download(f"https://drive.google.com/uc?id={file_id}", model_path, quiet=False)
        return tf.keras.models.load_model(model_path)

# Load model
model = load_model()

# Class labels
class_labels = ["Organic Waste 🍃", "Recyclable Waste 🔄"]

def preprocess_image(image):
    """Preprocess image for prediction."""
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def predict_waste(image):
    """Predict the waste category."""
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction)
    return class_labels[predicted_class], prediction[0][predicted_class] * 100

# Main UI
st.markdown("<h1 style='text-align: center; color: #2E8B57;'>♻️ Waste Classification using CNN</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload an image to classify it as <b>Organic</b> or <b>Recyclable</b> Waste.</p>", unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("📤 Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # Display uploaded image
    st.image(image, caption="📸 Uploaded Image", use_column_width=True)

    # Add a button for prediction
    if st.button("🔍 Predict Waste Type"):
        with st.spinner("⏳ Analyzing image..."):
            time.sleep(2)  # Simulating processing delay
            label, confidence = predict_waste(image)
        
        # Display results
        st.success(f"🏷 **Prediction:** {label}")
        st.info(f"🔍 **Confidence:** {confidence:.2f}%")

        # Progress bar
        progress_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.02)
            progress_bar.progress(percent_complete + 1)

# Footer
st.markdown(
    "<hr><p style='text-align: center; color: gray;'>🚀 Developed with ❤️ using TensorFlow & Streamlit</p>",
    unsafe_allow_html=True
)


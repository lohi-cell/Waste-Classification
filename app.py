import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('model.pkl')

st.title("ML Model Prediction Web App 🚀")
st.write("Provide input values and get predictions instantly!")

# Example input fields (Modify based on your features)
feature1 = st.number_input('Enter Feature 1:', value=0.0)
feature2 = st.number_input('Enter Feature 2:', value=0.0)
feature3 = st.number_input('Enter Feature 3:', value=0.0)

# Predict button
if st.button('Predict'):
    prediction = model.predict(np.array([[feature1, feature2, feature3]]))
    st.success(f"The predicted result is: {prediction[0]}")

import streamlit as st
import numpy as np
import pickle

# Load the trained model
with open('plant_disease_detection_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("🌿 Plant Disease Predictor")
st.subheader("Predict if a plant is likely to have a disease based on environmental factors")

# Input features
temperature = st.slider("Temperature (°C)", min_value=0.0, max_value=50.0, step=0.1)
humidity = st.slider("Humidity (%)", min_value=0.0, max_value=100.0, step=0.1)
rainfall = st.slider("Rainfall (mm)", min_value=0.0, max_value=500.0, step=0.1)
soil_pH = st.slider("Soil pH", min_value=3.0, max_value=10.0, step=0.1)

# Predict button
if st.button("Predict"):
    input_data = np.array([[temperature, humidity, rainfall, soil_pH]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("🚨 The plant is likely to have a disease.")
    else:
        st.success("✅ The plant is healthy.")

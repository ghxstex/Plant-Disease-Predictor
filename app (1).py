import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load trained model
with open('plant_disease_detection_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Streamlit UI
st.set_page_config(page_title="🌾 Plant Disease Predictor", layout="centered")

st.title("🌱 Plant Disease Prediction App")
st.markdown("Enter environmental conditions to predict if the plant is **diseased** or **healthy**.")

# Input fields
temperature = st.number_input("🌡️ Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0, step=0.1)
humidity = st.number_input("💧 Humidity (%)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
rainfall = st.number_input("🌧️ Rainfall (mm)", min_value=0.0, max_value=500.0, value=20.0, step=0.1)
soil_pH = st.number_input("🧪 Soil pH", min_value=0.0, max_value=14.0, value=6.5, step=0.1)

# Predict button
if st.button("🔍 Predict"):
    input_data = np.array([[temperature, humidity, rainfall, soil_pH]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        # red text
        st.error("🚨 The plant is likely to be **Diseased**.")
    else:
        # green text
        st.success("✅ The plant is likely to be **Healthy**.")



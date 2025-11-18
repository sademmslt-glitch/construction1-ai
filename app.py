import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page setup
st.set_page_config(page_title="AI Construction Predictor", page_icon="🏗️", layout="centered")

st.title("🏗️ AI-based Construction Project Prediction System")
st.write("""
This web app predicts:
- **Final Project Cost (SAR)** 💰
- **Probability of Delay (%)** ⏱️

Enter your project details below to get predictions.
""")

# User Inputs
size = st.number_input("Project Size (m²)", min_value=50, max_value=10000, value=1000)
workers = st.number_input("Number of Workers", min_value=1, max_value=500, value=20)
budget = st.number_input("Estimated Budget (SAR)", min_value=50000, max_value=50000000, value=500000)
duration = st.number_input("Expected Duration (months)", min_value=1, max_value=60, value=12)

# Load trained models
@st.cache_resource
def load_models():
    reg = joblib.load("regression_model.pkl")      # final cost model
    clf = joblib.load("classification_model.pkl")  # delay model
    return reg, clf

# Try loading the models
try:
    reg_model, clf_model = load_models()

    # Create input dataframe
    input_df = pd.DataFrame({
        "Project_Size": [size],
        "Num_Workers": [workers],
        "Budget": [budget],
        "Duration": [duration]
    })

    # Predictions
    predicted_cost = reg_model.predict(input_df)[0]
    delay_probability = clf_model.predict_proba(input_df)[0][1] * 100  # %

    # Output
    st.subheader("🔍 Prediction Results")
    st.metric("Predicted Final Cost (SAR)", f"{predicted_cost:,.0f}")
    st.metric("Delay Probability (%)", f"{delay_probability:.1f}%")

    st.info("Note: This model is trained on simulated data for academic purposes.")

except Exception as e:
    st.error("❌ Error loading models. Make sure the .pkl files are uploaded correctly.")
    st.write(e)
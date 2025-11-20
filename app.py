import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import datetime
import sklearn

# Page setup
st.set_page_config(page_title="AI Construction Predictor", page_icon="🏗️", layout="centered")

st.title("🏗️ AI-based Construction Project Prediction System")
st.write("""
This web app predicts:
- **Final Project Cost (SAR)** 💰
- **Probability of Delay (%)** ⏱️

Enter your project details below to get predictions.
""")

# ---------- Utility / helpers ----------
def validate_inputs(size, workers, budget, duration):
    errors = []
    if size <= 0: errors.append("Project Size must be > 0.")
    if workers <= 0: errors.append("Number of Workers must be > 0.")
    if budget <= 0: errors.append("Estimated Budget must be > 0.")
    if duration <= 0: errors.append("Expected Duration must be > 0.")
    return errors

def color_for_prob(p):
    # returns a color hex for probability p in [0,100]
    if p < 20:
        return "#2ecc71"  # green
    if p < 50:
        return "#f1c40f"  # yellow
    return "#e74c3c"      # red

def save_prediction(row, fname="predictions_log.csv"):
    df = pd.DataFrame([row])
    if os.path.exists(fname):
        df.to_csv(fname, mode="a", header=False, index=False)
    else:
        df.to_csv(fname, index=False)

# ---------- Sidebar / info ----------
st.sidebar.header("Model & Info")
st.sidebar.write(f"scikit-learn: {sklearn.__version__}")
st.sidebar.write(f"numpy: {np.__version__}")
st.sidebar.markdown("---")
st.sidebar.write("Tips:")
st.sidebar.write("- Use realistic budget and duration values.")
st.sidebar.write("- Results are based on the trained models (simulated data).")

# ---------- User Inputs ----------
st.subheader("Project Inputs")
col1, col2 = st.columns(2)
with col1:
    size = st.number_input("Project Size (m²)", min_value=1, max_value=100000, value=1000, step=1)
    budget = st.number_input("Estimated Budget (SAR)", min_value=1, max_value=10_000_000_000, value=500000, step=1000)
with col2:
    workers = st.number_input("Number of Workers", min_value=1, max_value=2000, value=20, step=1)
    duration = st.number_input("Expected Duration (months)", min_value=1, max_value=120, value=12, step=1)

# Sample presets using session_state (safer than experimental_rerun)
if "size" not in st.session_state:
    st.session_state["size"] = 1000
if "workers" not in st.session_state:
    st.session_state["workers"] = 20
if "budget" not in st.session_state:
    st.session_state["budget"] = 500000
if "duration" not in st.session_state:
    st.session_state["duration"] = 12

if st.button("Use example project"):
    st.session_state["size"] = 1200
    st.session_state["workers"] = 25
    st.session_state["budget"] = 600000
    st.session_state["duration"] = 14

# Then use session_state values as defaults for inputs:
size = st.number_input("Project Size (m²)", min_value=1, max_value=100000, value=st.session_state.get("size", 1000), step=1)
budget = st.number_input("Estimated Budget (SAR)", min_value=1, max_value=10_000_000_000, value=st.session_state.get("budget", 500000), step=1000)
workers = st.number_input("Number of Workers", min_value=1, max_value=2000, value=st.session_state.get("workers", 20), step=1)
duration = st.number_input("Expected Duration (months)", min_value=1, max_value=120, value=st.session_state.get("duration", 12), step=1)

# validation
errors = validate_inputs(size, workers, budget, duration)
if errors:
    for e in errors:
        st.error(e)
    st.stop()

# ---------- Load models ----------
@st.cache_resource
def load_models():
    reg_path = "regression_model.pkl"
    clf_path = "classification_model.pkl"
    reg = joblib.load(reg_path)
    clf = joblib.load(clf_path)
    return reg, clf, reg_path, clf_path

try:
    reg_model, clf_model, reg_path, clf_path = load_models()
except Exception as e:
    st.error("❌ Error loading models. Make sure regression_model.pkl and classification_model.pkl are uploaded in the app folder.")
    st.exception(e)
    st.stop()

# ---------- Prediction ----------
input_df = pd.DataFrame({
    "Project_Size": [size],
    "Num_Workers": [workers],
    "Budget": [budget],
    "Duration": [duration]
})

try:
    predicted_cost = float(reg_model.predict(input_df)[0])
except Exception as e:
    st.error("Error predicting cost.")
    st.exception(e)
    st.stop()

# probability
try:
    if hasattr(clf_model, "predict_proba"):
        delay_probability = float(clf_model.predict_proba(input_df)[0][1] * 100)
    else:
        # fallback to predict (0/1)
        delay_probability = float(clf_model.predict(input_df)[0] * 100)
except Exception as e:
    st.error("Error predicting delay probability.")
    st.exception(e)
    st.stop()

# ---------- Results Display ----------
st.subheader("🔍 Prediction Results")

# Show predicted cost with formatting and small note about expected error (editable)
mae_note = st.number_input("Expected MAE (SAR) — optional (enter model MAE if available)", min_value=0, value=0, step=1)
st.metric("Predicted Final Cost (SAR)", f"{predicted_cost:,.0f}")
if mae_note > 0:
    st.write(f"Estimated error (MAE): ±{int(mae_note):,} SAR")

# Delay probability as text + colored bar
st.metric("Delay Probability (%)", f"{delay_probability:.1f}%")
col = color_for_prob(delay_probability)
st.markdown(f"<div style='background:{col};height:12px;border-radius:6px'></div>", unsafe_allow_html=True)

# ---------- Save to log ----------
if st.button("Save this prediction"):
    row = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "Project_Size": size,
        "Num_Workers": workers,
        "Budget": budget,
        "Duration": duration,
        "Predicted_Cost": predicted_cost,
        "Delay_Probability": delay_probability
    }
    save_prediction(row)
    st.success("Prediction saved to predictions_log.csv (in app folder)")

# ---------- Show recent predictions ----------
if os.path.exists("predictions_log.csv"):
    st.subheader("Recent predictions (log)")
    logs = pd.read_csv("predictions_log.csv").sort_values("timestamp", ascending=False).head(10)
    st.dataframe(logs)

# ---------- Footer ----------
st.write("---")
st.caption("Note: This application is for academic/demo use. Replace models and data with real project records for production.")

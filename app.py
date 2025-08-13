import streamlit as st
import pandas as pd
import numpy as np
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fallback check
def check_dependencies():
    try:
        import torch
        st.success(f"PyTorch {torch.__version__} detected")
        return True
    except ImportError:
        st.error("PyTorch not installed! Using mock data mode")
        return False

# Mock data generator
def generate_mock_data(n_samples, condition):
    data = {
        "PatientID": range(1, n_samples+1),
        "Age": np.random.randint(18, 90, n_samples),
        "Condition": condition,
        "BloodPressure": np.random.randint(80, 180, n_samples),
        "Glucose": np.random.randint(70, 300, n_samples)
    }
    return pd.DataFrame(data)

# Main app
st.set_page_config(page_title="Medical GAN", layout="wide")
st.title("🏥 Synthetic Medical Records Generator")

# UI Controls
with st.sidebar:
    st.header("Settings")
    n_samples = st.slider("Records to generate", 1, 1000, 50)
    condition = st.selectbox("Condition", ["Diabetes", "Hypertension", "Asthma"])
    generate_btn = st.button("Generate")

# Core Logic
if generate_btn:
    if check_dependencies():
        try:
            # Real model would go here
            st.warning("Model loading not implemented - using mock data")
            df = generate_mock_data(n_samples, condition)
        except Exception as e:
            logger.error(str(e))
            df = generate_mock_data(n_samples, condition)
    else:
        df = generate_mock_data(n_samples, condition)
    
    st.dataframe(df)
    st.download_button("Download CSV", df.to_csv(), "records.csv")
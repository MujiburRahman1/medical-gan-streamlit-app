import streamlit as st
import pandas as pd
import torch
from main import Generator, get_private_discriminator
from validation import validate_synthetic

# --- UI Config ---
st.set_page_config(page_title="Medical GAN", layout="wide")
st.title("🔒 Privacy-Preserving Medical Record Generator")

# --- Sidebar Controls ---
with st.sidebar:
    st.header("Controls")
    n_samples = st.slider("Records to generate", 1, 1000, 50)
    condition = st.selectbox("Medical Condition", ["Diabetes", "Hypertension", "Asthma"])
    generate_btn = st.button("Generate Data")

# --- Load Model ---
@st.cache_resource
def load_model():
    generator = Generator(latent_dim=64, data_dim=42, n_classes=3)  # Update dims
    generator.load_state_dict(torch.load("generator.pth", map_location="cpu"))
    return generator

# --- Main Logic ---
if generate_btn:
    generator = load_model()
    condition_code = {"Diabetes": 0, "Hypertension": 1, "Asthma": 2}[condition]
    
    # Generate with labels for cGAN
    z = torch.randn(n_samples, 64)
    labels = torch.full((n_samples,), condition_code, dtype=torch.long)
    synthetic = generator(z, labels).detach().numpy()
    
    # Convert to DataFrame (add your inverse-transform logic)
    df = pd.DataFrame(synthetic, columns=[f"Feature_{i}" for i in range(synthetic.shape[1])])
    
    # Show results
    st.dataframe(df.head(10))
    st.download_button("Download CSV", df.to_csv(), "synthetic_records.csv")
    
    # Validation
    with st.spinner("Validating privacy..."):
        real_data = pd.read_csv("dataset.csv").values  # Load your real data
        score = validate_synthetic(real_data, synthetic)
        st.success(f"Privacy Score: {score:.3f} (0.5 = ideal)")
import streamlit as st
import pandas as pd
import torch
import numpy as np
import logging
import os
from torch import nn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="Medical GAN",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("🏥 Synthetic Medical Records Generator")
st.markdown("""
Generate privacy-preserving synthetic patient records using GANs.
""")

# --- Constants ---
LATENT_DIM = 64
DATA_DIM = 42  # Update based on your dataset
N_CLASSES = 3  # Diabetes, Hypertension, Asthma

# --- Model Definition ---
class Generator(nn.Module):
    def __init__(self, latent_dim, data_dim, n_classes):
        super().__init__()
        self.label_embed = nn.Embedding(n_classes, latent_dim)
        self.model = nn.Sequential(
            nn.Linear(latent_dim * 2, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, data_dim),
            nn.Tanh()
        )

    def forward(self, z, labels=None):
        if labels is not None:
            label_embed = self.label_embed(labels)
            z = torch.cat([z, label_embed], dim=1)
        return self.model(z)

# --- UI Components ---
def sidebar_controls():
    with st.sidebar:
        st.header("⚙️ Controls")
        n_samples = st.slider("Records to generate", 1, 1000, 50)
        condition = st.selectbox(
            "Medical Condition", 
            ["Diabetes", "Hypertension", "Asthma"]
        )
        generate_btn = st.button("Generate", type="primary")
        st.markdown("---")
        st.info("Note: This is a demo using mock data.")
    return n_samples, condition, generate_btn

# --- Mock Data Generation ---
def generate_mock_data(n_samples, condition):
    """Fallback function if model fails to load"""
    data = {
        "Age": np.random.randint(20, 80, n_samples),
        "BloodPressure": np.random.randint(80, 180, n_samples),
        "Condition": [condition] * n_samples
    }
    return pd.DataFrame(data)

# --- Main App ---
def main():
    n_samples, condition, generate_btn = sidebar_controls()
    
    if generate_btn:
        try:
            # Try loading real model
            generator = Generator(LATENT_DIM, DATA_DIM, N_CLASSES)
            generator.load_state_dict(
                torch.load("generator.pth", map_location="cpu")
            )
            
            # Generate synthetic data
            z = torch.randn(n_samples, LATENT_DIM)
            labels = torch.tensor(
                ["Diabetes", "Hypertension", "Asthma"].index(condition)
            ).repeat(n_samples)
            synthetic = generator(z, labels).detach().numpy()
            
            df = pd.DataFrame(synthetic)
            st.success("✅ Generated synthetic records!")
            
        except Exception as e:
            logger.error(f"Model failed: {str(e)}")
            st.warning("⚠️ Using mock data (model not found)")
            df = generate_mock_data(n_samples, condition)
        
        st.dataframe(df.head(10))
        st.download_button(
            "Download CSV",
            df.to_csv(index=False),
            f"synthetic_{condition}.csv"
        )

if __name__ == "__main__":
    main()
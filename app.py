import streamlit as st
import pandas as pd
import torch
import numpy as np
import os
import logging
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

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
This tool helps researchers work with realistic data without compromising patient privacy.
""")

# --- Constants ---
LATENT_DIM = 64
DATA_DIM = 42  # Update this based on your actual dataset features
N_CLASSES = 3  # Number of medical conditions

# --- File Path Handling ---
@st.cache_data
def get_file_path(filename):
    """Handle paths for both local and Streamlit Cloud deployment"""
    if os.path.exists(os.path.join(os.getcwd(), filename)):
        return os.path.join(os.getcwd(), filename)
    return os.path.join("/app", filename)  # Streamlit Cloud path

# --- GAN Model Definition ---
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

# --- Load Model ---
@st.cache_resource
def load_model():
    try:
        generator = Generator(LATENT_DIM, DATA_DIM, N_CLASSES)
        model_path = get_file_path("generator.pth")
        generator.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        generator.eval()
        return generator
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        st.error("Failed to load model. Please check if 'generator.pth' exists.")
        st.stop()

# --- Data Processing ---
@st.cache_data
def load_real_data():
    try:
        data_path = get_file_path("dataset.csv")
        return pd.read_csv(data_path)
    except Exception as e:
        logger.error(f"Data loading failed: {str(e)}")
        st.error("Failed to load dataset.csv")
        st.stop()

def preprocess_data(df):
    """Example preprocessing - adapt to your dataset"""
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = df.select_dtypes(include=["object"]).columns
    
    scaler = MinMaxScaler(feature_range=(-1, 1))
    encoder = OneHotEncoder(sparse_output=False)
    
    num_scaled = scaler.fit_transform(df[num_cols]) if len(num_cols) > 0 else np.empty((len(df), 0))
    cat_encoded = encoder.fit_transform(df[cat_cols]) if len(cat_cols) > 0 else np.empty((len(df), 0))
    
    return np.hstack((num_scaled, cat_encoded)).astype(np.float32), scaler, encoder

# --- UI Components ---
def sidebar_controls():
    with st.sidebar:
        st.header("⚙️ Controls")
        n_samples = st.slider("Number of records", 1, 1000, 50)
        condition = st.selectbox(
            "Medical condition",
            ["Diabetes", "Hypertension", "Asthma"],
            index=0
        )
        generate_btn = st.button("Generate Records", type="primary")
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("This app generates synthetic medical records using a GAN trained with differential privacy.")
    
    return n_samples, condition, generate_btn

# --- Main App Logic ---
def main():
    # Load dependencies
    generator = load_model()
    real_data = load_real_data()
    
    # Sidebar controls
    n_samples, condition, generate_btn = sidebar_controls()
    
    if generate_btn:
        with st.spinner("Generating synthetic records..."):
            try:
                # Convert condition to label
                condition_map = {"Diabetes": 0, "Hypertension": 1, "Asthma": 2}
                condition_code = condition_map[condition]
                
                # Generate synthetic data
                z = torch.randn(n_samples, LATENT_DIM)
                labels = torch.full((n_samples,), condition_code, dtype=torch.long)
                synthetic = generator(z, labels).detach().numpy()
                
                # Post-processing (example - adapt to your preprocessing)
                synthetic_df = pd.DataFrame(synthetic, columns=[f"feature_{i}" for i in range(synthetic.shape[1])])
                
                # Display results
                st.success(f"Generated {n_samples} {condition} records!")
                st.dataframe(synthetic_df.head(10))
                
                # Download button
                st.download_button(
                    label="Download as CSV",
                    data=synthetic_df.to_csv(index=False),
                    file_name=f"synthetic_{condition.lower()}_records.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                logger.error(f"Generation failed: {str(e)}")
                st.error("Generation failed. Please check the logs.")

if __name__ == "__main__":
    main()
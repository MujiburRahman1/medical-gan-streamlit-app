import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import torch
import torch.nn as nn
from opacus import PrivacyEngine  # <-- New

# --- Conditional GAN Generator ---
class Generator(nn.Module):
    def __init__(self, latent_dim, data_dim, n_classes):
        super().__init__()
        self.label_embed = nn.Embedding(n_classes, latent_dim)  # <-- New
        self.model = nn.Sequential(
            nn.Linear(latent_dim * 2, 128),  # Double input for noise + label
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, data_dim),
            nn.Tanh()
        )
    def forward(self, z, labels=None):  # <-- Modified
        if labels is not None:
            label_embed = self.label_embed(labels)
            z = torch.cat([z, label_embed], dim=1)
        return self.model(z)

# --- DP-Enabled Discriminator ---
def get_private_discriminator(input_dim, lr=0.0002):
    model = nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.LeakyReLU(0.2),
        nn.Linear(256, 128),
        nn.LeakyReLU(0.2),
        nn.Linear(128, 1),
        nn.Sigmoid()
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    privacy_engine = PrivacyEngine()
    return privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        noise_multiplier=1.0,
        max_grad_norm=1.0
    )

# --- Training Loop with Validation ---
def train_gan(data_processed, epochs=2000):
    # (Keep your existing preprocessing)
    generator = Generator(latent_dim=64, data_dim=data_processed.shape[1], n_classes=3)
    discriminator, optim_D = get_private_discriminator(data_processed.shape[1])
    
    for epoch in range(epochs):
        # ... (your existing training loop) ...
        if epoch % 200 == 0:
            synthetic = generate_samples(generator, n_samples=100)
            score = validate_synthetic(data_processed, synthetic)
            print(f"Epoch {epoch}: Privacy Score={score:.3f}")  # ~0.5 is ideal

    torch.save(generator.state_dict(), "generator.pth")  # Save for Streamlit
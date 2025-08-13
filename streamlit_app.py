"""
Streamlit App for Medical Conditional GAN with Differential Privacy

This app provides a user-friendly interface for training and using the cGAN
to generate synthetic medical records with privacy guarantees.
"""

import streamlit as st
import pandas as pd
import numpy as np

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
import base64
from datetime import datetime
import time

# Import our cGAN implementation
from medical_cgan_dp import MedicalConditionalGAN, MedicalDataset
from torch.utils.data import DataLoader

# Page configuration
st.set_page_config(
    page_title="Medical cGAN with Differential Privacy",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .privacy-info {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff7f0e;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🏥 Medical Conditional GAN with Differential Privacy</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model parameters
        st.subheader("Model Settings")
        noise_dim = st.slider("Noise Dimension", 50, 200, 100, 10)
        learning_rate = st.selectbox("Learning Rate", [0.0001, 0.0002, 0.0005, 0.001], index=1)
        batch_size = st.selectbox("Batch Size", [4, 8, 16, 32], index=1)
        epochs = st.slider("Training Epochs", 10, 200, 50, 10)
        
        # Privacy parameters
        st.subheader("🔒 Privacy Settings")
        epsilon = st.slider("Privacy Budget (ε)", 0.1, 10.0, 1.0, 0.1)
        delta = st.selectbox("Privacy Parameter (δ)", [1e-6, 1e-5, 1e-4], index=1)
        max_grad_norm = st.slider("Max Gradient Norm", 0.1, 2.0, 1.0, 0.1)
        
        # Generation parameters
        st.subheader("🎯 Generation Settings")
        num_samples_per_condition = st.slider("Samples per Condition", 10, 100, 50, 10)
        
        st.markdown("---")
        st.markdown("**Note**: Lower ε values provide stronger privacy but may reduce data quality.")
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Overview", "🚀 Training", "🎲 Generation", "📈 Results"])
    
    with tab1:
        show_data_overview()
    
    with tab2:
        show_training_interface(noise_dim, learning_rate, batch_size, epochs, epsilon, delta, max_grad_norm)
    
    with tab3:
        show_generation_interface(num_samples_per_condition)
    
    with tab4:
        show_results()

def show_data_overview():
    """Display dataset overview and statistics."""
    st.header("📊 Dataset Overview")
    
    try:
        # Load and display dataset
        dataset = MedicalDataset('dataset.csv')
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Samples", len(dataset))
        
        with col2:
            st.metric("Features", dataset.features.shape[1])
        
        with col3:
            st.metric("Conditions", dataset.conditions.shape[1])
        
        # Display sample data
        st.subheader("Sample Data")
        sample_df = pd.read_csv('dataset.csv').head(10)
        st.dataframe(sample_df, use_container_width=True)
        
        # Display condition distribution
        st.subheader("Condition Distribution")
        condition_df = pd.read_csv('dataset.csv')[['neuropathy', 'retinopathy', 'hypoglycemia', 'uti']]
        
        col1, col2 = st.columns(2)
        
        with col1:
            condition_counts = condition_df.sum()
            fig = px.bar(x=condition_counts.index, y=condition_counts.values,
                        title="Condition Counts", color=condition_counts.values,
                        color_continuous_scale='viridis')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Feature correlation heatmap
            numerical_cols = ['age_years', 'weight_kg', 'bmi', 'systolic_bp_mmHg', 'diastolic_bp_mmHg']
            corr_matrix = sample_df[numerical_cols].corr()
            fig = px.imshow(corr_matrix, title="Feature Correlation", 
                           color_continuous_scale='RdBu', aspect="auto")
            st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        st.info("Please ensure 'dataset.csv' is in the current directory.")

def show_training_interface(noise_dim, learning_rate, batch_size, epochs, epsilon, delta, max_grad_norm):
    """Show the training interface."""
    st.header("🚀 Model Training")
    
    if st.button("🚀 Start Training", type="primary", use_container_width=True):
        with st.spinner("Training in progress... This may take several minutes."):
            try:
                # Load dataset
                dataset = MedicalDataset('dataset.csv')
                
                # Split dataset
                train_size = int(0.8 * len(dataset))
                test_size = len(dataset) - train_size
                train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
                
                # Create data loader
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=0
                )
                
                # Configuration
                config = {
                    'noise_dim': noise_dim,
                    'condition_dim': dataset.conditions.shape[1],
                    'feature_dim': dataset.features.shape[1],
                    'generator_hidden_dims': [256, 512, 256],
                    'discriminator_hidden_dims': [256, 512, 256],
                    'learning_rate': learning_rate,
                    'batch_size': batch_size,
                    'epochs': epochs,
                    'epsilon': epsilon,
                    'delta': delta,
                    'max_grad_norm': max_grad_norm
                }
                
                # Initialize cGAN
                cgan = MedicalConditionalGAN(config)
                
                # Setup differential privacy
                cgan.setup_differential_privacy(
                    train_loader=train_loader,
                    epsilon=epsilon,
                    delta=delta,
                    max_grad_norm=max_grad_norm
                )
                
                # Training progress
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Custom training loop with progress updates
                cgan.train_loader = train_loader
                cgan.train_with_progress(epochs, progress_bar, status_text)
                
                # Save the trained model
                torch.save({
                    'generator_state_dict': cgan.generator.state_dict(),
                    'discriminator_state_dict': cgan.discriminator.state_dict(),
                    'config': config,
                    'training_history': {
                        'g_losses': cgan.g_losses,
                        'd_losses': cgan.d_losses
                    }
                }, 'trained_cgan.pth')
                
                st.success("✅ Training completed successfully!")
                st.session_state['model_trained'] = True
                st.session_state['cgan'] = cgan
                st.session_state['config'] = config
                
                # Display training results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Final Generator Loss", f"{cgan.g_losses[-1]:.4f}")
                
                with col2:
                    st.metric("Final Discriminator Loss", f"{cgan.d_losses[-1]:.4f}")
                
                # Training loss plot
                fig = make_subplots(rows=1, cols=2, subplot_titles=("Training Losses", "Generator Loss"))
                
                fig.add_trace(
                    go.Scatter(y=cgan.g_losses, name="Generator Loss", line=dict(color='blue')),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(y=cgan.d_losses, name="Discriminator Loss", line=dict(color='red')),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(y=cgan.g_losses, name="Generator Loss", line=dict(color='blue')),
                    row=1, col=2
                )
                
                fig.update_layout(height=400, showlegend=True)
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Training failed: {e}")
                st.exception(e)

def show_generation_interface(num_samples_per_condition):
    """Show the data generation interface."""
    st.header("🎲 Synthetic Data Generation")
    
    if not st.session_state.get('model_trained', False):
        st.warning("⚠️ Please train the model first in the Training tab.")
        return
    
    st.success("✅ Model is ready for generation!")
    
    # Condition selection
    st.subheader("Select Conditions for Generation")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        neuropathy = st.checkbox("Neuropathy", value=False)
    with col2:
        retinopathy = st.checkbox("Retinopathy", value=False)
    with col3:
        hypoglycemia = st.checkbox("Hypoglycemia", value=False)
    with col4:
        uti = st.checkbox("UTI", value=False)
    
    # Custom condition combinations
    st.subheader("Or Use Predefined Combinations")
    condition_combinations = {
        "Single Conditions": [
            ([1, 0, 0, 0], "Only Neuropathy"),
            ([0, 1, 0, 0], "Only Retinopathy"),
            ([0, 0, 1, 0], "Only Hypoglycemia"),
            ([0, 0, 0, 1], "Only UTI")
        ],
        "Combined Conditions": [
            ([1, 1, 0, 0], "Neuropathy + Retinopathy"),
            ([0, 0, 1, 1], "Hypoglycemia + UTI"),
            ([1, 1, 1, 1], "All Conditions"),
            ([0, 0, 0, 0], "No Conditions")
        ]
    }
    
    selected_combinations = []
    
    for category, combinations in condition_combinations.items():
        st.write(f"**{category}:**")
        cols = st.columns(len(combinations))
        for i, (condition, description) in enumerate(combinations):
            with cols[i]:
                if st.checkbox(description, key=f"combo_{i}"):
                    selected_combinations.append((condition, description))
    
    # Generation button
    if st.button("🎲 Generate Synthetic Data", type="primary", use_container_width=True):
        if not selected_combinations:
            st.warning("Please select at least one condition combination.")
            return
        
        with st.spinner("Generating synthetic data..."):
            try:
                cgan = st.session_state['cgan']
                
                synthetic_data = []
                synthetic_conditions = []
                
                for condition, description in selected_combinations:
                    st.write(f"Generating {description}...")
                    
                    conditions_tensor = torch.tensor([condition] * num_samples_per_condition, dtype=torch.float32)
                    synthetic_features = cgan.generate_synthetic_data(conditions_tensor, num_samples_per_condition)
                    
                    synthetic_data.append(synthetic_features.cpu().numpy())
                    synthetic_conditions.append(conditions_tensor.cpu().numpy())
                
                # Combine all synthetic data
                all_synthetic_features = np.vstack(synthetic_data)
                all_synthetic_conditions = np.vstack(synthetic_conditions)
                
                # Create DataFrame
                feature_columns = [f'feature_{i}' for i in range(all_synthetic_features.shape[1])]
                condition_columns = ['neuropathy', 'retinopathy', 'hypoglycemia', 'uti']
                
                synthetic_df = pd.DataFrame(
                    all_synthetic_features,
                    columns=feature_columns
                )
                
                # Add condition columns
                for i, col in enumerate(condition_columns):
                    synthetic_df[col] = all_synthetic_conditions[:, i]
                
                # Add patient IDs
                synthetic_df['patient_id'] = [f'SYNTH-{i:04d}' for i in range(len(synthetic_df))]
                
                # Save to session state
                st.session_state['synthetic_data'] = synthetic_df
                
                st.success(f"✅ Generated {len(synthetic_df)} synthetic medical records!")
                
                # Display summary
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Records", len(synthetic_df))
                
                with col2:
                    st.metric("Records per Combination", num_samples_per_condition)
                
                with col3:
                    st.metric("Combinations", len(selected_combinations))
                
                # Show sample data
                st.subheader("Sample Generated Data")
                st.dataframe(synthetic_df.head(10), use_container_width=True)
                
                # Download button
                csv = synthetic_df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="synthetic_medical_records.csv">📥 Download CSV</a>'
                st.markdown(href, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Generation failed: {e}")
                st.exception(e)

def show_results():
    """Show the results and analysis."""
    st.header("📈 Results & Analysis")
    
    if not st.session_state.get('synthetic_data', None):
        st.info("No synthetic data generated yet. Please use the Generation tab first.")
        return
    
    synthetic_df = st.session_state['synthetic_data']
    
    # Data overview
    st.subheader("Generated Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", len(synthetic_df))
    
    with col2:
        st.metric("Features", len([col for col in synthetic_df.columns if col.startswith('feature_')]))
    
    with col3:
        st.metric("Conditions", len([col for col in synthetic_df.columns if col in ['neuropathy', 'retinopathy', 'hypoglycemia', 'uti']]))
    
    with col4:
        st.metric("Unique Patients", synthetic_df['patient_id'].nunique())
    
    # Condition distribution
    st.subheader("Condition Distribution")
    
    condition_cols = ['neuropathy', 'retinopathy', 'hypoglycemia', 'uti']
    condition_counts = synthetic_df[condition_cols].sum()
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(x=condition_counts.index, y=condition_counts.values,
                    title="Condition Counts in Synthetic Data",
                    color=condition_counts.values,
                    color_continuous_scale='viridis')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Pie chart
        fig = px.pie(values=condition_counts.values, names=condition_counts.index,
                     title="Condition Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature analysis
    st.subheader("Feature Analysis")
    
    feature_cols = [col for col in synthetic_df.columns if col.startswith('feature_')]
    
    if feature_cols:
        # Feature statistics
        feature_stats = synthetic_df[feature_cols].describe()
        st.write("**Feature Statistics:**")
        st.dataframe(feature_stats, use_container_width=True)
        
        # Feature correlation heatmap
        st.write("**Feature Correlation Matrix:**")
        corr_matrix = synthetic_df[feature_cols].corr()
        fig = px.imshow(corr_matrix, 
                       title="Feature Correlation Heatmap",
                       color_continuous_scale='RdBu',
                       aspect="auto")
        st.plotly_chart(fig, use_container_width=True)
    
    # Privacy metrics
    st.subheader("🔒 Privacy Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Privacy Budget (ε)", st.session_state.get('config', {}).get('epsilon', 'N/A'))
        st.metric("Privacy Parameter (δ)", st.session_state.get('config', {}).get('delta', 'N/A'))
    
    with col2:
        st.metric("Max Gradient Norm", st.session_state.get('config', {}).get('max_grad_norm', 'N/A'))
        
        # Privacy info box
        st.markdown("""
        <div class="privacy-info">
        <strong>Privacy Guarantees:</strong><br>
        • Individual patient records are protected during training<br>
        • Generated data maintains statistical properties<br>
        • Configurable privacy-utility trade-off
        </div>
        """, unsafe_allow_html=True)

# Add the missing method to MedicalConditionalGAN class
def add_progress_training_method():
    """Add progress tracking method to the cGAN class."""
    def train_with_progress(self, epochs, progress_bar, status_text):
        """Train with progress updates for Streamlit."""
        self.train_loader = self.train_loader
        
        for epoch in range(epochs):
            g_loss_epoch = 0
            d_loss_epoch = 0
            
            for batch_idx, (real_features, real_conditions) in enumerate(self.train_loader):
                real_features = real_features.to(self.device)
                real_conditions = real_conditions.to(self.device)
                batch_size = real_features.size(0)
                
                # Create labels
                real_labels = torch.ones(batch_size, 1).to(self.device)
                fake_labels = torch.zeros(batch_size, 1).to(self.device)
                
                # Train Discriminator
                self.d_optimizer.zero_grad()
                
                # Real data
                real_outputs = self.discriminator(real_features, real_conditions)
                d_real_loss = self.criterion(real_outputs, real_labels)
                
                # Fake data
                noise = torch.randn(batch_size, self.config['noise_dim']).to(self.device)
                fake_features = self.generator(noise, real_conditions)
                fake_outputs = self.discriminator(fake_features.detach(), real_conditions)
                d_fake_loss = self.criterion(fake_outputs, fake_labels)
                
                d_loss = d_real_loss + d_fake_loss
                d_loss.backward()
                
                # Apply gradient clipping for basic privacy protection
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.config['max_grad_norm'])
                
                self.d_optimizer.step()
                
                # Train Generator
                self.g_optimizer.zero_grad()
                
                fake_outputs = self.discriminator(fake_features, real_conditions)
                g_loss = self.criterion(fake_outputs, real_labels)
                
                g_loss.backward()
                self.g_optimizer.step()
                
                # Record losses
                g_loss_epoch += g_loss.item()
                d_loss_epoch += d_loss.item()
            
            # Record epoch losses
            avg_g_loss = g_loss_epoch / len(self.train_loader)
            avg_d_loss = d_loss_epoch / len(self.train_loader)
            self.g_losses.append(avg_g_loss)
            self.d_losses.append(avg_d_loss)
            
            # Update progress
            progress = (epoch + 1) / epochs
            progress_bar.progress(progress)
            status_text.text(f"Epoch {epoch + 1}/{epochs} - G_Loss: {avg_g_loss:.4f}, D_Loss: {avg_d_loss:.4f}")
            
            # Small delay for UI responsiveness
            time.sleep(0.1)
    
    # Add the method to the class
    MedicalConditionalGAN.train_with_progress = train_with_progress

# Initialize the app
if __name__ == "__main__":
    add_progress_training_method()
    main()

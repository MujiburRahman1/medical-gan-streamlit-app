# Medical Conditional GAN with Differential Privacy

A complete PyTorch implementation of a Conditional Generative Adversarial Network (cGAN) for generating synthetic medical records with differential privacy guarantees.

## 🏥 Overview

This implementation provides a privacy-preserving solution for generating synthetic medical records that can be used for research, development, and testing without compromising patient confidentiality. The system uses differential privacy through the Opacus library to ensure that training does not leak identifiable patient information.

## ✨ Features

- **Conditional Generation**: Generate synthetic medical records conditioned on specific disease types and demographics
- **Differential Privacy**: Built-in privacy protection using DP-SGD with configurable privacy parameters
- **Tabular Data Support**: Handles both numerical and categorical medical features
- **Medical Dataset Compatibility**: Works with MIMIC-III, Synthea, or similar structured medical datasets
- **Modular Architecture**: Clean, well-documented code with separate components for Generator, Discriminator, and training pipeline
- **Privacy Guarantees**: Configurable epsilon and delta parameters for differential privacy

## 🚀 Quick Start

### Prerequisites

The required dependencies are already included in `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Running the Implementation

Simply run the main script:

```bash
python medical_cgan_dp.py
```

The script will:
1. Load and preprocess the medical dataset
2. Train the conditional GAN with differential privacy
3. Generate synthetic medical records for different condition combinations
4. Save the results to `synthetic_medical_records.csv`
5. Display training progress and privacy budget

## 📊 Dataset Structure

The implementation expects medical records in CSV format with the following structure:

### Required Columns:
- **Numerical Features**: Age, weight, BMI, blood pressure, lab values, etc.
- **Categorical Features**: Medications, diagnosis codes, etc.
- **Condition Labels**: Binary indicators for diseases (neuropathy, retinopathy, hypoglycemia, UTI)

### Example Dataset Format:
```csv
patient_id,age_years,weight_kg,bmi,systolic_bp_mmHg,diastolic_bp_mmHg,heart_rate_bpm,body_temp_C,fasting_glucose_mg_dL,hba1c_percent,medications,neuropathy,retinopathy,hypoglycemia,uti
P-2025-001,52,83.7,28.3,138,86,80,36.8,137,8.73,"Metformin 1000 mg BID",0,0,0,0
P-2025-002,45,70.2,24.1,125,78,72,36.9,95,6.2,"Insulin NPH",1,0,1,0
```

## 🏗️ Architecture

### Generator Network
- **Input**: Random noise + condition labels
- **Architecture**: Fully connected layers with BatchNorm, LeakyReLU, and Dropout
- **Output**: Synthetic medical features

### Discriminator Network
- **Input**: Medical features + condition labels
- **Architecture**: Fully connected layers with LeakyReLU and Dropout
- **Output**: Probability of being real (0-1)

### Differential Privacy Integration
- Uses Opacus library for DP-SGD
- Gradient clipping and noise addition
- Configurable privacy budget (ε, δ)

## ⚙️ Configuration

The main configuration parameters can be modified in the `main()` function:

```python
config = {
    'noise_dim': 100,                    # Dimension of noise vector
    'condition_dim': 4,                  # Number of condition labels
    'feature_dim': 27,                   # Number of medical features
    'generator_hidden_dims': [256, 512, 256],
    'discriminator_hidden_dims': [256, 512, 256],
    'learning_rate': 0.0002,
    'batch_size': 32,
    'epochs': 100,
    'epsilon': 1.0,                      # Privacy parameter ε
    'delta': 1e-5,                       # Privacy parameter δ
    'max_grad_norm': 1.0                 # Gradient clipping norm
}
```

## 🔒 Privacy Features

### Differential Privacy Parameters
- **Epsilon (ε)**: Controls privacy level (lower = more private)
- **Delta (δ)**: Probability of privacy failure (typically 1e-5)
- **Max Gradient Norm**: Clips gradients to prevent information leakage

### Privacy Guarantees
- Training does not leak individual patient information
- Generated data maintains statistical properties of original data
- Configurable privacy-utility trade-off

## 📈 Output

The implementation generates:

1. **Synthetic Medical Records**: CSV file with generated patient data
2. **Training Progress**: Real-time loss monitoring and privacy budget tracking
3. **Visualization**: Training loss plots
4. **Condition Distribution**: Statistics on generated conditions

### Sample Output Structure:
```csv
patient_id,age_years,weight_kg,bmi,systolic_bp_mmHg,diastolic_bp_mmHg,heart_rate_bpm,body_temp_C,fasting_glucose_mg_dL,hba1c_percent,medication_encoded,neuropathy,retinopathy,hypoglycemia,uti
SYNTH-0000,48.2,75.8,26.1,132.4,84.7,78.3,36.9,112.6,7.2,3,1,0,0,0
SYNTH-0001,51.7,82.1,27.8,138.9,87.2,81.1,37.1,125.3,8.1,2,0,1,0,0
```

## 🎯 Use Cases

### Research Applications
- Clinical trial simulation
- Medical algorithm development
- Healthcare system testing
- Privacy-preserving data sharing

### Development Scenarios
- Testing medical software
- Training healthcare AI models
- Validating clinical protocols
- Educational purposes

## 🔧 Customization

### Adding New Features
1. Modify the `MedicalDataset` class to include new columns
2. Update the `feature_dim` in configuration
3. Adjust the feature column lists in the main function

### Changing Conditions
1. Modify `condition_columns` in `MedicalDataset`
2. Update `condition_dim` in configuration
3. Adjust condition combinations in the generation loop

### Privacy Tuning
1. Adjust `epsilon` for privacy level (lower = more private)
2. Modify `max_grad_norm` for gradient clipping
3. Change `delta` for privacy failure probability

## 📚 Technical Details

### Model Architecture
- **Generator**: 3 hidden layers (256→512→256) with BatchNorm and Dropout
- **Discriminator**: 3 hidden layers (256→512→256) with Dropout
- **Activation**: LeakyReLU (0.2) for hidden layers, Tanh for generator output, Sigmoid for discriminator

### Training Process
1. **Discriminator Training**: Real vs fake classification with condition labels
2. **Generator Training**: Adversarial training to fool discriminator
3. **Privacy Protection**: DP-SGD with gradient clipping and noise addition

### Data Preprocessing
- **Numerical Features**: StandardScaler normalization
- **Categorical Features**: LabelEncoder encoding
- **Missing Values**: Zero-filling
- **Condition Labels**: Binary encoding

## 🛡️ Privacy Considerations

### What This Implementation Protects
- Individual patient records during training
- Sensitive medical information
- Patient identities and demographics

### What It Doesn't Protect
- Statistical patterns in the data
- General medical knowledge
- Population-level insights

### Best Practices
- Use appropriate epsilon values (1-10 for most applications)
- Regularly audit privacy budget consumption
- Validate synthetic data quality
- Consider additional privacy measures for production use

## 📄 License

This implementation is provided for educational and research purposes. Please ensure compliance with relevant privacy regulations (HIPAA, GDPR, etc.) when using with real medical data.

## 🤝 Contributing

Contributions are welcome! Please ensure:
- Code follows the existing style
- New features include documentation
- Privacy considerations are addressed
- Tests are added for new functionality

## 📞 Support

For questions or issues:
1. Check the configuration parameters
2. Verify dataset format compatibility
3. Review privacy parameter settings
4. Ensure all dependencies are installed

---

**Note**: This implementation is designed for research and development purposes. Always consult with privacy experts and legal professionals when working with real medical data in production environments.

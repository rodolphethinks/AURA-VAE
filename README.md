# AURA-VAE: Acoustic Unsupervised Recognition of Anomalies

An unsupervised in-vehicle acoustic anomaly detection system using Variational Autoencoders (VAE).

## Project Overview

AURA-VAE is designed to detect acoustic anomalies in vehicle environments using unsupervised learning. The system:
- Trains on normal driving sounds only
- Detects anomalies based on reconstruction error
- Runs fully offline on Android 11 devices
- Optimized for 1800×720 landscape displays

## Algorithm Design

### VAE-based Acoustic Anomaly Detection

The approach is based on the DCASE (Detection and Classification of Acoustic Scenes and Events) challenge methodologies for unsupervised anomalous sound detection.

**Key References:**
- DCASE 2023 Task 2: First-shot Unsupervised Anomalous Sound Detection
- "First-shot anomaly sound detection for machine condition monitoring" (Harada et al., 2023)
- Autoencoder-based approaches with reconstruction error scoring

### Architecture Overview

```
Input Audio → Preprocessing → Log-Mel Spectrogram → VAE Encoder → Latent Space → VAE Decoder → Reconstruction → Anomaly Score
```

**VAE Components:**
1. **Encoder**: Convolutional layers that compress mel spectrograms to latent representation
2. **Latent Space**: Low-dimensional representation with KL regularization
3. **Decoder**: Transposed convolutions to reconstruct the spectrogram
4. **Anomaly Score**: Mean Squared Error between input and reconstruction

**Why VAE for Anomaly Detection:**
- Trained only on normal data, learns the "normal" distribution
- Anomalies produce high reconstruction error (model hasn't seen them)
- KL divergence regularizes latent space for better generalization
- Compact model suitable for mobile deployment

## Results

![Analysis of Filante Sound Acquisition 3](experiments/legacy_results_v1/inference/Filante%20Sound%20Acquisition%203_analysis_v2.png)

## Project Structure

```
Aura/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── python/
│   ├── config.py               # Configuration parameters
│   ├── preprocessing.py        # Audio preprocessing utilities
│   ├── dataset.py              # Dataset loading and management
│   ├── model.py                # VAE model architecture
│   ├── train.py                # Training script
│   ├── evaluate.py             # Evaluation and visualization
│   ├── convert_tflite.py       # TFLite conversion script
│   └── utils.py                # Utility functions
├── models/                      # Saved models
│   ├── vae_model.h5            # Trained Keras model
│   ├── vae_model.tflite        # TFLite converted model
│   └── normalization_params.json # Feature normalization parameters
├── data/
│   ├── raw/                    # Raw audio files
│   ├── processed/              # Preprocessed features
│   └── synthetic/              # Synthetic anomaly samples
├── results/
│   ├── plots/                  # Training plots and visualizations
│   └── metrics/                # Evaluation metrics
└── android/
    └── AuraVAE/                # Android Studio project
```

## Audio Preprocessing Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Sample Rate | 16,000 Hz | Standard for speech/audio ML |
| Window Length | 1.0 s | Audio segment duration |
| Hop Length | 0.5 s | 50% overlap |
| FFT Size | 1024 | Frequency resolution |
| Hop Size (STFT) | 512 | Time resolution |
| Mel Bins | 64 | Mel frequency bins |
| Frequency Range | 50-8000 Hz | Relevant frequency range |

## Quick Start

### 1. Setup Python Environment

```bash
cd c:\Users\rodol\Documents\Neos\Aura
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run Pipeline (Train, Evaluate, Convert)

To run the full pipeline (preprocessing, training, evaluation, conversion):

```bash
python run_pipeline.py --retrain
```

Or for individual steps:

```bash
python python/preprocessing.py
python python/train.py
python python/evaluate.py
python python/convert_tflite.py
```

### 3. Build Android App

1. Models are automatically copied to `android/AuraVAE/app/src/main/assets`
2. Open `android/AuraVAE` in Android Studio
3. Build and run on Android 11 device

## Model Specifications

- **Input Shape**: (64, 32, 1) - 64 mel bins × 32 time frames
- **Latent Dimension**: 32
- **Model Size**: ~2.5 MB (TFLite)
- **Inference Time**: < 50ms per segment

## Anomaly Detection Logic

```python
anomaly_score = mean_squared_error(input_spectrogram, reconstructed_spectrogram)
threshold = mean(training_scores) + k * std(training_scores)  # k = 3
is_anomaly = anomaly_score > threshold
```

## Android App Features

- **Recording**: 30-60 second audio capture
- **Real-time Status**: Recording and analysis indicators
- **Results**: Normal/Anomaly classification with confidence score
- **Offline**: Fully on-device inference

## System Requirements

### Python Development
- Python 3.8+
- TensorFlow 2.x
- librosa, numpy, scipy

### Android Device
- Android 11 (API Level 30)
- Screen: 1800×720 landscape
- Microphone access
- Storage: ~50 MB

## License

MIT License - See LICENSE file

## Acknowledgments

- DCASE Challenge for anomaly detection methodologies
- TensorFlow Lite for mobile inference
- librosa for audio processing

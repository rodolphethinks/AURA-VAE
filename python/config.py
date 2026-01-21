"""
AURA-VAE Configuration Parameters

This file contains all configuration parameters for the audio preprocessing,
model architecture, and training pipeline. These parameters are designed to
be consistent between Python training and Android inference.

Reference: DCASE Challenge methodologies for acoustic anomaly detection
"""

import os

# =============================================================================
# PATH CONFIGURATION
# =============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
SYNTHETIC_DATA_DIR = os.path.join(DATA_DIR, "synthetic")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")

# Source audio file
SOURCE_AUDIO_FILE = os.path.join(BASE_DIR, "Filante Sound Acquisition.m4a")

# =============================================================================
# AUDIO PREPROCESSING PARAMETERS
# These MUST match Android implementation exactly
# =============================================================================

# Sample rate for all processing
SAMPLE_RATE = 16000  # 16 kHz - standard for audio ML

# Audio segmentation
SEGMENT_DURATION = 1.0  # seconds per segment
SEGMENT_HOP = 0.5  # 50% overlap between segments
SEGMENT_SAMPLES = int(SAMPLE_RATE * SEGMENT_DURATION)  # 16000 samples
HOP_SAMPLES = int(SAMPLE_RATE * SEGMENT_HOP)  # 8000 samples

# STFT parameters
N_FFT = 1024  # FFT window size
HOP_LENGTH = 512  # STFT hop length (32 frames per 1s segment)
WIN_LENGTH = 1024  # Window length

# Mel spectrogram parameters
N_MELS = 64  # Number of mel frequency bins
F_MIN = 50  # Minimum frequency (Hz)
F_MAX = 8000  # Maximum frequency (Hz) - Nyquist/2 for 16kHz

# Derived dimensions
# Number of time frames = (SEGMENT_SAMPLES - N_FFT) / HOP_LENGTH + 1
# For 16000 samples, 1024 FFT, 512 hop: (16000 - 1024) / 512 + 1 ≈ 30-32
N_TIME_FRAMES = 32  # We'll pad/trim to this exact size

# Spectrogram normalization
SPEC_REF = 1.0  # Reference for dB conversion
SPEC_AMIN = 1e-10  # Minimum amplitude for log
SPEC_TOP_DB = 80.0  # Maximum dynamic range

# =============================================================================
# MODEL ARCHITECTURE PARAMETERS
# =============================================================================

# Input shape for VAE (mel_bins, time_frames, channels)
INPUT_SHAPE = (N_MELS, N_TIME_FRAMES, 1)

# Latent space dimension
LATENT_DIM = 16  # Compact latent representation

# Encoder architecture (filters for each conv layer)
ENCODER_FILTERS = [32, 64, 128]
ENCODER_KERNELS = [(3, 3), (3, 3), (3, 3)]
ENCODER_STRIDES = [(2, 2), (2, 2), (2, 2)]

# Dense layer after conv layers
ENCODER_DENSE = 128

# Decoder mirrors encoder
DECODER_DENSE = 128
DECODER_FILTERS = [128, 64, 32]
DECODER_KERNELS = [(3, 3), (3, 3), (3, 3)]
DECODER_STRIDES = [(2, 2), (2, 2), (2, 2)]

# =============================================================================
# TRAINING PARAMETERS
# =============================================================================

# Data split
TRAIN_SPLIT = 0.7  # 70% for training
TEST_SPLIT = 0.3   # 30% for testing

# Training hyperparameters
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-3

# VAE loss weights
RECONSTRUCTION_WEIGHT = 1.0
KL_WEIGHT = 0.001  # Beta-VAE weight (lower = more reconstruction focus)

# Early stopping
EARLY_STOPPING_PATIENCE = 15
MIN_DELTA = 1e-4

# Learning rate scheduler
LR_REDUCE_FACTOR = 0.5
LR_REDUCE_PATIENCE = 5
LR_MIN = 1e-6

# =============================================================================
# ANOMALY DETECTION PARAMETERS
# =============================================================================

# Threshold calculation (mean + k * std)
ANOMALY_THRESHOLD_K = 3.0

# Percentile-based threshold (alternative)
ANOMALY_THRESHOLD_PERCENTILE = 95

# =============================================================================
# ANDROID-SPECIFIC PARAMETERS
# =============================================================================

# Recording parameters
ANDROID_SAMPLE_RATE = 16000
ANDROID_CHANNEL_CONFIG = "MONO"
ANDROID_AUDIO_FORMAT = "PCM_16BIT"
ANDROID_RECORD_DURATION_MIN = 30  # seconds
ANDROID_RECORD_DURATION_MAX = 60  # seconds

# Display
ANDROID_SCREEN_WIDTH = 1800
ANDROID_SCREEN_HEIGHT = 720

# =============================================================================
# FILE NAMES
# =============================================================================

MODEL_FILENAME = "vae_model.weights.h5"
TFLITE_FILENAME = "vae_model.tflite"
NORM_PARAMS_FILENAME = "normalization_params.json"
TRAINING_HISTORY_FILENAME = "training_history.json"

# =============================================================================
# UTILITY FUNCTION TO CREATE DIRECTORIES
# =============================================================================

def create_directories():
    """Create all necessary directories for the project."""
    dirs = [
        DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, SYNTHETIC_DATA_DIR,
        MODELS_DIR, RESULTS_DIR, PLOTS_DIR, METRICS_DIR
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print(f"Created project directories in: {BASE_DIR}")

if __name__ == "__main__":
    create_directories()
    print("\nConfiguration Summary:")
    print(f"  Sample Rate: {SAMPLE_RATE} Hz")
    print(f"  Segment Duration: {SEGMENT_DURATION}s")
    print(f"  Mel Bins: {N_MELS}")
    print(f"  Time Frames: {N_TIME_FRAMES}")
    print(f"  Input Shape: {INPUT_SHAPE}")
    print(f"  Latent Dimension: {LATENT_DIM}")

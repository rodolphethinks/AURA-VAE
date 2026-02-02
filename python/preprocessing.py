"""
AURA-VAE Audio Preprocessing Module

Handles all audio preprocessing including:
- Audio loading and format conversion
- Resampling to target sample rate
- Mono conversion
- Amplitude normalization
- Segmentation with overlap
- Log-mel spectrogram extraction

All preprocessing steps are designed to be exactly replicable on Android.
"""

import os
import numpy as np
import librosa
import soundfile as sf
from scipy import signal
from tqdm import tqdm
import json

from config import (
    SAMPLE_RATE, SEGMENT_DURATION, SEGMENT_HOP, SEGMENT_SAMPLES, HOP_SAMPLES,
    N_FFT, HOP_LENGTH, WIN_LENGTH, N_MELS, F_MIN, F_MAX, N_TIME_FRAMES,
    SPEC_REF, SPEC_AMIN, SPEC_TOP_DB,
    SOURCE_AUDIO_FILE, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR,
    NORM_PARAMS_FILENAME, create_directories
)


def load_audio(file_path: str, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Load audio file and convert to mono at target sample rate.
    
    Supports: WAV, M4A, MP3, FLAC, OGG, etc. (via librosa/audioread)
    
    Args:
        file_path: Path to audio file
        target_sr: Target sample rate (default: 16000)
    
    Returns:
        Audio samples as numpy array (float32, normalized to [-1, 1])
    """
    print(f"Loading audio: {file_path}")
    
    # Load with librosa (handles resampling and mono conversion)
    audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
    
    print(f"  Original duration: {len(audio) / sr:.2f}s")
    print(f"  Sample rate: {sr} Hz")
    print(f"  Samples: {len(audio)}")
    
    return audio.astype(np.float32)


def normalize_audio(audio: np.ndarray, target_db: float = -3.0) -> np.ndarray:
    """
    Normalize audio amplitude to target dB level.
    
    Args:
        audio: Input audio samples
        target_db: Target peak dB level (default: -3 dB)
    
    Returns:
        Normalized audio samples
    """
    # Calculate current peak
    peak = np.max(np.abs(audio))
    
    if peak > 0:
        # Normalize to [-1, 1] then scale to target dB
        target_linear = 10 ** (target_db / 20)
        audio = audio / peak * target_linear
    
    return audio


def segment_audio(audio: np.ndarray, 
                  segment_samples: int = SEGMENT_SAMPLES,
                  hop_samples: int = HOP_SAMPLES) -> np.ndarray:
    """
    Segment audio into fixed-length overlapping windows.
    
    Args:
        audio: Input audio samples
        segment_samples: Samples per segment
        hop_samples: Hop between segments (overlap = segment - hop)
    
    Returns:
        Array of shape (n_segments, segment_samples)
    """
    segments = []
    
    # Calculate number of complete segments
    n_segments = max(1, (len(audio) - segment_samples) // hop_samples + 1)
    
    for i in range(n_segments):
        start = i * hop_samples
        end = start + segment_samples
        
        if end <= len(audio):
            segment = audio[start:end]
        else:
            # Pad last segment if needed
            segment = np.zeros(segment_samples, dtype=audio.dtype)
            segment[:len(audio) - start] = audio[start:]
        
        segments.append(segment)
    
    return np.array(segments)


def compute_mel_spectrogram(audio_segment: np.ndarray,
                            sr: int = SAMPLE_RATE,
                            n_fft: int = N_FFT,
                            hop_length: int = HOP_LENGTH,
                            n_mels: int = N_MELS,
                            f_min: float = F_MIN,
                            f_max: float = F_MAX) -> np.ndarray:
    """
    Compute log-mel spectrogram from audio segment.
    
    This implementation matches what will be used on Android.
    
    Args:
        audio_segment: Audio samples (1D array)
        sr: Sample rate
        n_fft: FFT window size
        hop_length: STFT hop length
        n_mels: Number of mel bins
        f_min: Minimum frequency
        f_max: Maximum frequency
    
    Returns:
        Log-mel spectrogram of shape (n_mels, n_time_frames)
    """
    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio_segment,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        n_mels=n_mels,
        fmin=f_min,
        fmax=f_max,
        power=2.0  # Power spectrogram
    )
    
    # Convert to log scale (dB)
    # Use ref=np.max to normalize each spectrogram relative to its maximum
    # This is consistent with run_pipeline.py training preprocessing
    log_mel_spec = librosa.power_to_db(
        mel_spec,
        ref=np.max,
        amin=SPEC_AMIN,
        top_db=SPEC_TOP_DB
    )
    
    # Ensure consistent time dimension
    if log_mel_spec.shape[1] < N_TIME_FRAMES:
        # Pad with minimum value
        pad_width = N_TIME_FRAMES - log_mel_spec.shape[1]
        log_mel_spec = np.pad(
            log_mel_spec, 
            ((0, 0), (0, pad_width)),
            mode='constant',
            constant_values=-SPEC_TOP_DB
        )
    elif log_mel_spec.shape[1] > N_TIME_FRAMES:
        # Trim to exact size
        log_mel_spec = log_mel_spec[:, :N_TIME_FRAMES]
    
    return log_mel_spec.astype(np.float32)


def extract_features_from_audio(audio: np.ndarray) -> np.ndarray:
    """
    Extract mel spectrogram features from raw audio.
    
    Full pipeline: segmentation -> mel spectrogram extraction
    
    Args:
        audio: Raw audio samples (1D array)
    
    Returns:
        Features array of shape (n_segments, n_mels, n_time_frames)
    """
    # Segment the audio
    segments = segment_audio(audio)
    print(f"  Extracted {len(segments)} segments")
    
    # Extract mel spectrograms
    features = []
    for segment in tqdm(segments, desc="  Extracting mel spectrograms"):
        mel_spec = compute_mel_spectrogram(segment)
        features.append(mel_spec)
    
    return np.array(features)


class FeatureNormalizer:
    """
    Feature normalizer for mel spectrograms.
    
    Normalizes features to zero mean and unit variance.
    Parameters are saved and loaded for consistency between training and inference.
    """
    
    def __init__(self):
        self.mean = None
        self.std = None
        self.fitted = False
    
    def fit(self, features: np.ndarray):
        """
        Fit normalizer on training features.
        
        Args:
            features: Training features of shape (n_samples, n_mels, n_time_frames)
        """
        # Compute global mean and std across all samples
        self.mean = np.mean(features)
        self.std = np.std(features)
        
        # Avoid division by zero
        if self.std < 1e-8:
            self.std = 1.0
        
        self.fitted = True
        print(f"  Normalizer fitted: mean={self.mean:.4f}, std={self.std:.4f}")
    
    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        Transform features using fitted parameters.
        
        Args:
            features: Features to normalize
        
        Returns:
            Normalized features
        """
        if not self.fitted:
            raise ValueError("Normalizer must be fitted before transform")
        
        return (features - self.mean) / self.std
    
    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(features)
        return self.transform(features)
    
    def inverse_transform(self, features: np.ndarray) -> np.ndarray:
        """Inverse transform normalized features."""
        if not self.fitted:
            raise ValueError("Normalizer must be fitted before inverse_transform")
        
        return features * self.std + self.mean
    
    def save(self, file_path: str):
        """Save normalization parameters to JSON file."""
        params = {
            'mean': float(self.mean),
            'std': float(self.std),
            'fitted': self.fitted
        }
        with open(file_path, 'w') as f:
            json.dump(params, f, indent=2)
        print(f"  Saved normalization parameters to: {file_path}")
    
    def load(self, file_path: str):
        """Load normalization parameters from JSON file."""
        with open(file_path, 'r') as f:
            params = json.load(f)
        self.mean = params['mean']
        self.std = params['std']
        self.fitted = params['fitted']
        print(f"  Loaded normalization parameters: mean={self.mean:.4f}, std={self.std:.4f}")


def preprocess_source_audio():
    """
    Main preprocessing pipeline for the source audio file.
    
    Steps:
    1. Load and convert audio
    2. Normalize amplitude
    3. Segment with overlap
    4. Extract mel spectrograms
    5. Save processed features
    """
    create_directories()
    
    print("\n" + "="*60)
    print("AURA-VAE Audio Preprocessing Pipeline")
    print("="*60)
    
    # Check if source file exists
    if not os.path.exists(SOURCE_AUDIO_FILE):
        raise FileNotFoundError(f"Source audio not found: {SOURCE_AUDIO_FILE}")
    
    print(f"\n[1/4] Loading audio file...")
    audio = load_audio(SOURCE_AUDIO_FILE)
    
    print(f"\n[2/4] Normalizing audio...")
    audio = normalize_audio(audio)
    print(f"  Peak amplitude: {np.max(np.abs(audio)):.4f}")
    
    print(f"\n[3/4] Extracting features...")
    features = extract_features_from_audio(audio)
    print(f"  Feature shape: {features.shape}")
    print(f"  (samples, mel_bins, time_frames)")
    
    print(f"\n[4/4] Saving processed features...")
    features_path = os.path.join(PROCESSED_DATA_DIR, "normal_features.npy")
    np.save(features_path, features)
    print(f"  Saved to: {features_path}")
    
    # Also save raw segments for reference
    segments = segment_audio(audio)
    segments_path = os.path.join(PROCESSED_DATA_DIR, "normal_segments.npy")
    np.save(segments_path, segments)
    print(f"  Segments saved to: {segments_path}")
    
    print("\n" + "="*60)
    print("Preprocessing Complete!")
    print("="*60)
    print(f"\nSummary:")
    print(f"  Total audio duration: {len(audio) / SAMPLE_RATE:.2f}s")
    print(f"  Number of segments: {len(features)}")
    print(f"  Feature shape per segment: {features[0].shape}")
    print(f"  Total features size: {features.nbytes / 1024 / 1024:.2f} MB")
    
    return features


def generate_synthetic_anomalies(n_samples: int = 100) -> np.ndarray:
    """
    Generate synthetic anomaly samples for evaluation.
    
    Types of synthetic anomalies:
    1. White noise
    2. Pink noise
    3. Sine waves (various frequencies)
    4. Impulse sounds
    5. Random frequency sweeps
    
    Args:
        n_samples: Number of synthetic samples to generate per type
    
    Returns:
        Synthetic anomaly features
    """
    print("\nGenerating synthetic anomalies...")
    
    all_anomalies = []
    samples_per_type = n_samples // 5
    
    # 1. White noise
    print("  Generating white noise...")
    for _ in range(samples_per_type):
        noise = np.random.randn(SEGMENT_SAMPLES).astype(np.float32) * 0.5
        mel_spec = compute_mel_spectrogram(noise)
        all_anomalies.append(mel_spec)
    
    # 2. Pink noise (1/f noise)
    print("  Generating pink noise...")
    for _ in range(samples_per_type):
        white = np.random.randn(SEGMENT_SAMPLES)
        # Simple pink noise approximation
        b, a = signal.butter(1, 0.1)
        pink = signal.filtfilt(b, a, white).astype(np.float32) * 0.5
        mel_spec = compute_mel_spectrogram(pink)
        all_anomalies.append(mel_spec)
    
    # 3. Sine waves (various frequencies)
    print("  Generating sine waves...")
    for _ in range(samples_per_type):
        freq = np.random.uniform(200, 4000)
        t = np.linspace(0, SEGMENT_DURATION, SEGMENT_SAMPLES)
        sine = (np.sin(2 * np.pi * freq * t) * 0.5).astype(np.float32)
        mel_spec = compute_mel_spectrogram(sine)
        all_anomalies.append(mel_spec)
    
    # 4. Impulse sounds
    print("  Generating impulse sounds...")
    for _ in range(samples_per_type):
        impulse = np.zeros(SEGMENT_SAMPLES, dtype=np.float32)
        n_impulses = np.random.randint(5, 20)
        positions = np.random.randint(0, SEGMENT_SAMPLES, n_impulses)
        impulse[positions] = np.random.uniform(0.3, 1.0, n_impulses)
        # Add some decay
        for pos in positions:
            decay_len = min(500, SEGMENT_SAMPLES - pos)
            decay = np.exp(-np.linspace(0, 5, decay_len))
            impulse[pos:pos+decay_len] *= decay[:len(impulse[pos:pos+decay_len])]
        mel_spec = compute_mel_spectrogram(impulse)
        all_anomalies.append(mel_spec)
    
    # 5. Frequency sweeps
    print("  Generating frequency sweeps...")
    for _ in range(samples_per_type):
        t = np.linspace(0, SEGMENT_DURATION, SEGMENT_SAMPLES)
        f0, f1 = np.random.uniform(100, 1000), np.random.uniform(2000, 7000)
        sweep = signal.chirp(t, f0, SEGMENT_DURATION, f1).astype(np.float32) * 0.5
        mel_spec = compute_mel_spectrogram(sweep)
        all_anomalies.append(mel_spec)
    
    anomalies = np.array(all_anomalies)
    print(f"  Generated {len(anomalies)} synthetic anomaly samples")
    
    return anomalies


if __name__ == "__main__":
    # Run preprocessing pipeline
    features = preprocess_source_audio()
    
    # Generate synthetic anomalies
    anomalies = generate_synthetic_anomalies(n_samples=200)
    anomalies_path = os.path.join(PROCESSED_DATA_DIR, "synthetic_anomalies.npy")
    np.save(anomalies_path, anomalies)
    print(f"\nSaved synthetic anomalies to: {anomalies_path}")

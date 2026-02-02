"""
Test mel spectrogram preprocessing functionality.

Verifies:
- Consistent mel spectrogram extraction
- Proper normalization
- Shape consistency
- Power-to-dB conversion consistency
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python.config import (
    SAMPLE_RATE, SEGMENT_SAMPLES, N_FFT, HOP_LENGTH, N_MELS, 
    F_MIN, F_MAX, N_TIME_FRAMES, SPEC_REF, SPEC_AMIN, SPEC_TOP_DB
)
from python.preprocessing import (
    segment_audio, compute_mel_spectrogram, FeatureNormalizer,
    extract_features_from_audio
)


class TestMelSpectrogram:
    """Test mel spectrogram extraction."""
    
    def test_mel_spectrogram_shape(self):
        """Test that mel spectrogram has correct shape."""
        # Create 1 second of audio at 16kHz
        audio_segment = np.random.randn(SEGMENT_SAMPLES).astype(np.float32)
        
        mel_spec = compute_mel_spectrogram(audio_segment)
        
        assert mel_spec.shape == (N_MELS, N_TIME_FRAMES), \
            f"Expected shape ({N_MELS}, {N_TIME_FRAMES}), got {mel_spec.shape}"
    
    def test_mel_spectrogram_dtype(self):
        """Test that mel spectrogram is float32."""
        audio_segment = np.random.randn(SEGMENT_SAMPLES).astype(np.float32)
        mel_spec = compute_mel_spectrogram(audio_segment)
        
        assert mel_spec.dtype == np.float32, \
            f"Expected float32, got {mel_spec.dtype}"
    
    def test_mel_spectrogram_range(self):
        """Test that mel spectrogram values are in expected dB range."""
        audio_segment = np.random.randn(SEGMENT_SAMPLES).astype(np.float32) * 0.5
        mel_spec = compute_mel_spectrogram(audio_segment)
        
        # Values should be in reasonable dB range
        # With ref=1.0, random noise can produce positive dB values
        assert mel_spec.min() >= -SPEC_TOP_DB - 1, \
            f"Min value {mel_spec.min()} below expected range"
        # Allow positive values up to ~20 dB (random noise can be loud)
        assert mel_spec.max() <= 20, \
            f"Max value {mel_spec.max()} above expected range"
    
    def test_mel_spectrogram_silent_audio(self):
        """Test mel spectrogram with silent audio.
        
        With ref=np.max normalization, silent audio (all zeros) will have
        np.max return 0, resulting in 0 dB output (or very low values).
        The key is that the spectrogram should have consistent low energy.
        """
        silent_audio = np.zeros(SEGMENT_SAMPLES, dtype=np.float32)
        mel_spec = compute_mel_spectrogram(silent_audio)
        
        # Should produce consistent low/zero energy values
        assert mel_spec.shape == (N_MELS, N_TIME_FRAMES)
        # With ref=np.max, silent audio produces 0 dB (max of 0 -> ref=0)
        # All values should be the same (uniform silence)
        assert mel_spec.std() < 0.01, \
            f"Silent audio should have uniform low energy, got std={mel_spec.std()}"
    
    def test_mel_spectrogram_sine_wave(self):
        """Test mel spectrogram with known sine wave."""
        # 1kHz sine wave
        t = np.linspace(0, 1, SEGMENT_SAMPLES, dtype=np.float32)
        sine_wave = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
        
        mel_spec = compute_mel_spectrogram(sine_wave)
        
        assert mel_spec.shape == (N_MELS, N_TIME_FRAMES)
        # Should have some energy around the 1kHz bin
        assert mel_spec.max() > -60, \
            f"1kHz sine wave produced too low energy: {mel_spec.max()}"
    
    def test_mel_spectrogram_consistency(self):
        """Test that same input produces same output."""
        audio_segment = np.random.randn(SEGMENT_SAMPLES).astype(np.float32)
        
        mel_spec1 = compute_mel_spectrogram(audio_segment)
        mel_spec2 = compute_mel_spectrogram(audio_segment)
        
        np.testing.assert_array_equal(mel_spec1, mel_spec2, 
            err_msg="Same input produced different outputs")


class TestSegmentation:
    """Test audio segmentation."""
    
    def test_segment_audio_basic(self):
        """Test basic audio segmentation."""
        # 3 seconds of audio should produce multiple segments
        audio = np.random.randn(SAMPLE_RATE * 3).astype(np.float32)
        
        segments = segment_audio(audio)
        
        # With 1s segments and 0.5s hop, 3s audio should give ~5 segments
        assert len(segments) >= 4, f"Expected at least 4 segments, got {len(segments)}"
        assert segments.shape[1] == SEGMENT_SAMPLES, \
            f"Each segment should be {SEGMENT_SAMPLES} samples"
    
    def test_segment_audio_short(self):
        """Test segmentation with audio shorter than one segment."""
        short_audio = np.random.randn(SEGMENT_SAMPLES // 2).astype(np.float32)
        
        segments = segment_audio(short_audio)
        
        # Should still produce at least one segment (padded)
        assert len(segments) >= 1
    
    def test_segment_overlap(self):
        """Test that segments have proper overlap."""
        audio = np.arange(SAMPLE_RATE * 2, dtype=np.float32)
        segments = segment_audio(audio)
        
        if len(segments) >= 2:
            # Second half of first segment should equal first half of second segment
            # (with 50% overlap)
            overlap_region1 = segments[0][SEGMENT_SAMPLES//2:]
            overlap_region2 = segments[1][:SEGMENT_SAMPLES//2]
            
            np.testing.assert_array_equal(overlap_region1, overlap_region2,
                err_msg="Segment overlap is incorrect")


class TestFeatureNormalizer:
    """Test feature normalizer."""
    
    def test_normalizer_fit(self):
        """Test normalizer fitting."""
        normalizer = FeatureNormalizer()
        features = np.random.randn(100, N_MELS, N_TIME_FRAMES) * 20 - 40
        
        normalizer.fit(features)
        
        assert normalizer.fitted == True
        assert normalizer.mean is not None
        assert normalizer.std is not None
        assert normalizer.std > 0
    
    def test_normalizer_transform(self):
        """Test normalizer transform."""
        normalizer = FeatureNormalizer()
        features = np.random.randn(100, N_MELS, N_TIME_FRAMES) * 20 - 40
        
        normalizer.fit(features)
        normalized = normalizer.transform(features)
        
        # Normalized data should have approximately zero mean and unit std
        assert abs(normalized.mean()) < 0.1, \
            f"Normalized mean should be ~0, got {normalized.mean()}"
        assert abs(normalized.std() - 1.0) < 0.1, \
            f"Normalized std should be ~1, got {normalized.std()}"
    
    def test_normalizer_fit_transform(self):
        """Test fit_transform convenience method."""
        normalizer = FeatureNormalizer()
        features = np.random.randn(100, N_MELS, N_TIME_FRAMES) * 20 - 40
        
        normalized = normalizer.fit_transform(features)
        
        assert normalizer.fitted == True
        assert abs(normalized.mean()) < 0.1
    
    def test_normalizer_inverse_transform(self):
        """Test that inverse_transform recovers original data."""
        normalizer = FeatureNormalizer()
        features = np.random.randn(100, N_MELS, N_TIME_FRAMES) * 20 - 40
        
        normalizer.fit(features)
        normalized = normalizer.transform(features)
        recovered = normalizer.inverse_transform(normalized)
        
        np.testing.assert_array_almost_equal(features, recovered, decimal=5,
            err_msg="Inverse transform did not recover original data")
    
    def test_normalizer_save_load(self, tmp_path):
        """Test saving and loading normalizer parameters."""
        normalizer1 = FeatureNormalizer()
        features = np.random.randn(100, N_MELS, N_TIME_FRAMES) * 20 - 40
        normalizer1.fit(features)
        
        # Save
        save_path = str(tmp_path / "norm_params.json")
        normalizer1.save(save_path)
        
        # Load into new normalizer
        normalizer2 = FeatureNormalizer()
        normalizer2.load(save_path)
        
        assert normalizer2.mean == normalizer1.mean
        assert normalizer2.std == normalizer1.std
        assert normalizer2.fitted == normalizer1.fitted
    
    def test_normalizer_transform_before_fit_raises(self):
        """Test that transform before fit raises error."""
        normalizer = FeatureNormalizer()
        features = np.random.randn(10, N_MELS, N_TIME_FRAMES)
        
        with pytest.raises(ValueError):
            normalizer.transform(features)


class TestPreprocessingConsistency:
    """Test preprocessing consistency between modules."""
    
    def test_inference_preprocessing_matches_training(self):
        """Test that inference.py preprocessing matches preprocessing.py."""
        # Import both implementations
        from python.preprocessing import compute_mel_spectrogram as train_mel
        from python.inference import extract_mel_spectrogram as infer_mel
        
        # Create test audio
        audio = np.random.randn(SEGMENT_SAMPLES).astype(np.float32) * 0.5
        
        # Extract mel spectrograms using both methods
        train_spec = train_mel(audio)
        infer_spec = infer_mel(audio)
        
        # Ensure consistent shape
        if infer_spec.shape[1] > N_TIME_FRAMES:
            infer_spec = infer_spec[:, :N_TIME_FRAMES]
        elif infer_spec.shape[1] < N_TIME_FRAMES:
            pad = N_TIME_FRAMES - infer_spec.shape[1]
            infer_spec = np.pad(infer_spec, ((0, 0), (0, pad)), mode='constant')
        
        # They should be identical (or very close)
        np.testing.assert_array_almost_equal(
            train_spec, infer_spec, decimal=4,
            err_msg="Training and inference mel extraction produce different results!"
        )


class TestFeatureExtraction:
    """Test full feature extraction pipeline."""
    
    def test_extract_features_from_audio(self):
        """Test full feature extraction pipeline."""
        # 5 seconds of audio
        audio = np.random.randn(SAMPLE_RATE * 5).astype(np.float32) * 0.5
        
        features = extract_features_from_audio(audio)
        
        # Should produce multiple feature samples
        assert len(features) >= 8, f"Expected at least 8 features, got {len(features)}"
        assert features.shape[1] == N_MELS
        assert features.shape[2] == N_TIME_FRAMES


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

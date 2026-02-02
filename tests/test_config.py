"""
Test configuration and utility functions.

Verifies:
- Config values are valid
- Utility functions work correctly
- Directory creation
"""

import pytest
import numpy as np
import os
import sys
import json
import tempfile

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python.config import (
    SAMPLE_RATE, SEGMENT_DURATION, SEGMENT_HOP, SEGMENT_SAMPLES, HOP_SAMPLES,
    N_FFT, HOP_LENGTH, N_MELS, F_MIN, F_MAX, N_TIME_FRAMES,
    INPUT_SHAPE, LATENT_DIM,
    BATCH_SIZE, EPOCHS, LEARNING_RATE,
    ANOMALY_THRESHOLD_K, ANOMALY_THRESHOLD_PERCENTILE,
    create_directories
)
from python.utils import save_json, load_json, set_seed, Timer, format_duration


class TestConfigValues:
    """Test that configuration values are valid."""
    
    def test_sample_rate_valid(self):
        """Test sample rate is a standard value."""
        assert SAMPLE_RATE in [8000, 16000, 22050, 44100, 48000], \
            f"Non-standard sample rate: {SAMPLE_RATE}"
    
    def test_segment_parameters_consistent(self):
        """Test segment duration and samples are consistent."""
        expected_samples = int(SAMPLE_RATE * SEGMENT_DURATION)
        assert SEGMENT_SAMPLES == expected_samples, \
            f"SEGMENT_SAMPLES ({SEGMENT_SAMPLES}) != SAMPLE_RATE * SEGMENT_DURATION ({expected_samples})"
        
        expected_hop_samples = int(SAMPLE_RATE * SEGMENT_HOP)
        assert HOP_SAMPLES == expected_hop_samples, \
            f"HOP_SAMPLES ({HOP_SAMPLES}) != SAMPLE_RATE * SEGMENT_HOP ({expected_hop_samples})"
    
    def test_fft_parameters_valid(self):
        """Test FFT parameters are valid."""
        # N_FFT should be power of 2
        assert N_FFT > 0 and (N_FFT & (N_FFT - 1)) == 0, \
            f"N_FFT ({N_FFT}) should be power of 2"
        
        # HOP_LENGTH should be less than N_FFT
        assert HOP_LENGTH < N_FFT, \
            f"HOP_LENGTH ({HOP_LENGTH}) should be < N_FFT ({N_FFT})"
    
    def test_mel_parameters_valid(self):
        """Test mel spectrogram parameters are valid."""
        assert N_MELS > 0 and N_MELS <= 128, \
            f"N_MELS ({N_MELS}) should be between 1 and 128"
        
        assert F_MIN >= 0, f"F_MIN ({F_MIN}) should be >= 0"
        assert F_MAX > F_MIN, f"F_MAX ({F_MAX}) should be > F_MIN ({F_MIN})"
        assert F_MAX <= SAMPLE_RATE / 2, \
            f"F_MAX ({F_MAX}) should be <= Nyquist ({SAMPLE_RATE / 2})"
    
    def test_input_shape_valid(self):
        """Test input shape matches expected format."""
        assert len(INPUT_SHAPE) == 3, \
            f"INPUT_SHAPE should have 3 dimensions (mel, time, channels)"
        assert INPUT_SHAPE[0] == N_MELS, \
            f"INPUT_SHAPE[0] ({INPUT_SHAPE[0]}) != N_MELS ({N_MELS})"
        assert INPUT_SHAPE[1] == N_TIME_FRAMES, \
            f"INPUT_SHAPE[1] ({INPUT_SHAPE[1]}) != N_TIME_FRAMES ({N_TIME_FRAMES})"
        assert INPUT_SHAPE[2] == 1, \
            f"INPUT_SHAPE[2] should be 1 (mono channel)"
    
    def test_latent_dim_reasonable(self):
        """Test latent dimension is reasonable."""
        assert LATENT_DIM >= 8, f"LATENT_DIM ({LATENT_DIM}) too small"
        assert LATENT_DIM <= 256, f"LATENT_DIM ({LATENT_DIM}) too large"
    
    def test_training_parameters_valid(self):
        """Test training parameters are valid."""
        assert BATCH_SIZE > 0 and BATCH_SIZE <= 256
        assert EPOCHS > 0
        assert LEARNING_RATE > 0 and LEARNING_RATE < 1
    
    def test_threshold_parameters_valid(self):
        """Test anomaly threshold parameters are valid."""
        assert ANOMALY_THRESHOLD_K > 0, "Threshold K should be positive"
        assert 0 < ANOMALY_THRESHOLD_PERCENTILE <= 100, \
            "Percentile should be between 0 and 100"


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_save_load_json(self):
        """Test JSON save and load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.json")
            
            data = {
                "string": "hello",
                "number": 42,
                "float": 3.14,
                "list": [1, 2, 3],
                "nested": {"a": 1, "b": 2}
            }
            
            save_json(data, filepath)
            loaded = load_json(filepath)
            
            assert loaded == data
    
    def test_set_seed_reproducibility(self):
        """Test that set_seed makes numpy operations reproducible."""
        set_seed(42)
        arr1 = np.random.randn(10)
        
        set_seed(42)
        arr2 = np.random.randn(10)
        
        np.testing.assert_array_equal(arr1, arr2)
    
    def test_timer(self):
        """Test Timer utility."""
        import time
        
        timer = Timer()
        timer.start()
        time.sleep(0.1)  # Sleep for 100ms
        elapsed = timer.stop()
        
        # Should be at least 100ms
        assert elapsed >= 0.1, f"Timer reported {elapsed}s for 100ms sleep"
        assert elapsed < 1.0, f"Timer reported {elapsed}s for 100ms sleep"
    
    def test_format_duration_seconds(self):
        """Test duration formatting for seconds."""
        result = format_duration(45.5)
        assert "45" in result or "s" in result
    
    def test_format_duration_minutes(self):
        """Test duration formatting for minutes."""
        result = format_duration(125)  # 2m 5s
        assert "2" in result and "m" in result
    
    def test_format_duration_hours(self):
        """Test duration formatting for hours."""
        result = format_duration(3700)  # 1h 1m
        assert "h" in result


class TestDirectoryCreation:
    """Test directory creation utility."""
    
    def test_create_directories(self):
        """Test that create_directories doesn't raise errors."""
        # This should not raise any errors
        create_directories()


class TestAudioRegistry:
    """Test audio registry functionality."""
    
    def test_audio_registry_exists(self):
        """Test that audio registry file exists."""
        registry_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "audio_registry.json"
        )
        
        assert os.path.exists(registry_path), "audio_registry.json not found"
    
    def test_audio_registry_valid_json(self):
        """Test that audio registry is valid JSON."""
        registry_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "audio_registry.json"
        )
        
        if os.path.exists(registry_path):
            with open(registry_path, 'r') as f:
                registry = json.load(f)
            
            assert 'training_audio_files' in registry, \
                "Registry missing 'training_audio_files' key"
            assert isinstance(registry['training_audio_files'], list), \
                "'training_audio_files' should be a list"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

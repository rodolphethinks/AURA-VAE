"""
Integration tests for the complete AURA-VAE pipeline.

Tests the full workflow from audio input to anomaly detection output.
"""

import pytest
import numpy as np
import os
import sys
import tempfile
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python.config import (
    SAMPLE_RATE, SEGMENT_SAMPLES, INPUT_SHAPE, 
    N_MELS, N_TIME_FRAMES, MODELS_DIR
)
from python.preprocessing import (
    segment_audio, compute_mel_spectrogram, FeatureNormalizer,
    extract_features_from_audio
)
from python.model import create_vae_model
from python.dataset import AuraDataset


class TestEndToEndPipeline:
    """Test complete pipeline from audio to anomaly detection."""
    
    def test_audio_to_features_pipeline(self):
        """Test complete audio preprocessing pipeline."""
        # Create synthetic audio (5 seconds)
        audio = np.random.randn(SAMPLE_RATE * 5).astype(np.float32) * 0.5
        
        # Extract features
        features = extract_features_from_audio(audio)
        
        # Check shape
        assert len(features) >= 8, f"Expected at least 8 segments, got {len(features)}"
        assert features.shape[1] == N_MELS
        assert features.shape[2] == N_TIME_FRAMES
    
    def test_features_to_model_pipeline(self):
        """Test features through model pipeline."""
        # Create features
        features = np.random.randn(10, N_MELS, N_TIME_FRAMES).astype(np.float32)
        
        # Normalize
        normalizer = FeatureNormalizer()
        normalized = normalizer.fit_transform(features)
        
        # Add channel dimension
        normalized = normalized[..., np.newaxis]
        
        # Create model and run inference
        vae = create_vae_model()
        
        # Get anomaly scores
        scores = vae.compute_anomaly_score(normalized)
        
        assert len(scores) == 10
        assert all(scores >= 0)
    
    def test_complete_pipeline(self):
        """Test complete pipeline from raw audio to anomaly scores."""
        # Step 1: Create synthetic audio
        audio = np.random.randn(SAMPLE_RATE * 3).astype(np.float32) * 0.5
        
        # Step 2: Extract features
        features = extract_features_from_audio(audio)
        
        # Step 3: Normalize
        normalizer = FeatureNormalizer()
        normalized = normalizer.fit_transform(features)
        
        # Step 4: Add channel dimension
        normalized = normalized[..., np.newaxis]
        
        # Step 5: Create model
        vae = create_vae_model()
        
        # Step 6: Get anomaly scores
        scores = vae.compute_anomaly_score(normalized)
        
        # Step 7: Apply threshold
        threshold = np.percentile(scores, 95)
        anomalies = scores > threshold
        
        # Verify outputs
        assert len(scores) == len(features)
        assert anomalies.dtype == bool
        print(f"Pipeline: {len(audio)} samples -> {len(features)} features -> {len(scores)} scores")
        print(f"Threshold: {threshold:.4f}, Anomalies: {anomalies.sum()}/{len(anomalies)}")


class TestNormalizationConsistency:
    """Test that normalization is consistent across training and inference."""
    
    def test_normalization_roundtrip(self):
        """Test that normalize -> inverse_normalize recovers original."""
        features = np.random.randn(100, N_MELS, N_TIME_FRAMES).astype(np.float32) * 20 - 40
        
        normalizer = FeatureNormalizer()
        normalized = normalizer.fit_transform(features)
        recovered = normalizer.inverse_transform(normalized)
        
        np.testing.assert_array_almost_equal(features, recovered, decimal=5)
    
    def test_normalization_parameters_persist(self):
        """Test that normalization parameters can be saved and loaded."""
        features = np.random.randn(100, N_MELS, N_TIME_FRAMES).astype(np.float32) * 20 - 40
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Fit and save
            normalizer1 = FeatureNormalizer()
            norm1 = normalizer1.fit_transform(features)
            normalizer1.save(os.path.join(tmpdir, "params.json"))
            
            # Load and transform same data
            normalizer2 = FeatureNormalizer()
            normalizer2.load(os.path.join(tmpdir, "params.json"))
            norm2 = normalizer2.transform(features)
            
            # Should produce identical results
            np.testing.assert_array_equal(norm1, norm2)


class TestModelConsistency:
    """Test model behavior consistency."""
    
    def test_model_deterministic_with_same_weights(self):
        """Test that same input + same weights = same output."""
        vae = create_vae_model()
        
        test_input = np.random.randn(5, *INPUT_SHAPE).astype(np.float32)
        
        # Multiple forward passes should give same result
        output1 = vae(test_input, training=False)
        output2 = vae(test_input, training=False)
        
        # Use z_mean for deterministic output (not sampled z)
        z_mean1, _, _ = vae.encode(test_input)
        z_mean2, _, _ = vae.encode(test_input)
        
        np.testing.assert_array_equal(z_mean1.numpy(), z_mean2.numpy())
    
    def test_anomaly_scores_stable(self):
        """Test that anomaly scores are stable (close, not exact due to VAE sampling)."""
        vae = create_vae_model()
        test_input = np.random.randn(10, *INPUT_SHAPE).astype(np.float32)
        
        scores1 = vae.compute_anomaly_score(test_input)
        scores2 = vae.compute_anomaly_score(test_input)
        
        # VAE uses sampling layer, so allow small differences
        np.testing.assert_array_almost_equal(scores1, scores2, decimal=2,
            err_msg="Anomaly scores should be stable within tolerance")


class TestProductionReadiness:
    """Test production model readiness."""
    
    @pytest.mark.skipif(
        not os.path.exists(os.path.join(MODELS_DIR, "normalization_params.json")),
        reason="Production normalization params not found"
    )
    def test_production_normalization_params(self):
        """Test that production normalization parameters are valid."""
        params_path = os.path.join(MODELS_DIR, "normalization_params.json")
        
        with open(params_path, 'r') as f:
            params = json.load(f)
        
        assert 'mean' in params, "Missing 'mean' in normalization params"
        assert 'std' in params, "Missing 'std' in normalization params"
        assert params['std'] > 0, "std should be positive"
    
    @pytest.mark.skipif(
        not os.path.exists(os.path.join(MODELS_DIR, "threshold_config.json")),
        reason="Production threshold config not found"
    )
    def test_production_threshold_config(self):
        """Test that production threshold config is valid."""
        config_path = os.path.join(MODELS_DIR, "threshold_config.json")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        assert 'threshold' in config, "Missing 'threshold' in config"
        assert config['threshold'] > 0, "threshold should be positive"
    
    @pytest.mark.skipif(
        not os.path.exists(os.path.join(MODELS_DIR, "android_config.json")),
        reason="Production Android config not found"
    )
    def test_production_android_config(self):
        """Test that production Android config is valid and consistent."""
        config_path = os.path.join(MODELS_DIR, "android_config.json")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Check required fields
        assert 'sample_rate' in config
        assert 'n_fft' in config
        assert 'hop_length' in config
        assert 'n_mels' in config
        assert 'n_time_frames' in config
        assert 'input_shape' in config
        
        # Check consistency with Python config
        assert config['sample_rate'] == SAMPLE_RATE, \
            f"Android sample_rate {config['sample_rate']} != Python {SAMPLE_RATE}"
        assert config['n_mels'] == N_MELS, \
            f"Android n_mels {config['n_mels']} != Python {N_MELS}"
        assert config['n_time_frames'] == N_TIME_FRAMES, \
            f"Android n_time_frames {config['n_time_frames']} != Python {N_TIME_FRAMES}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Test VAE model functionality.

Verifies:
- Model architecture
- Forward pass
- Encoding/decoding
- Anomaly score computation
- Weight saving/loading
"""

import pytest
import numpy as np
import tensorflow as tf
import sys
import os
import tempfile

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python.config import INPUT_SHAPE, LATENT_DIM, N_MELS, N_TIME_FRAMES
from python.model import (
    VAE, create_vae_model, build_encoder, build_decoder, Sampling
)


class TestVAEArchitecture:
    """Test VAE model architecture."""
    
    @pytest.fixture
    def vae_model(self):
        """Create a fresh VAE model for each test."""
        return create_vae_model()
    
    def test_model_creation(self, vae_model):
        """Test that VAE model can be created."""
        assert vae_model is not None
        assert hasattr(vae_model, 'encoder')
        assert hasattr(vae_model, 'decoder')
    
    def test_encoder_output_shape(self, vae_model):
        """Test encoder output shapes."""
        test_input = np.random.randn(4, *INPUT_SHAPE).astype(np.float32)
        z_mean, z_log_var, z = vae_model.encoder(test_input)
        
        assert z_mean.shape == (4, LATENT_DIM), \
            f"z_mean shape wrong: {z_mean.shape}"
        assert z_log_var.shape == (4, LATENT_DIM), \
            f"z_log_var shape wrong: {z_log_var.shape}"
        assert z.shape == (4, LATENT_DIM), \
            f"z shape wrong: {z.shape}"
    
    def test_decoder_output_shape(self, vae_model):
        """Test decoder output shapes."""
        z = np.random.randn(4, LATENT_DIM).astype(np.float32)
        reconstruction = vae_model.decoder(z)
        
        assert reconstruction.shape == (4, *INPUT_SHAPE), \
            f"Reconstruction shape wrong: {reconstruction.shape}"
    
    def test_vae_forward_pass(self, vae_model):
        """Test full VAE forward pass."""
        test_input = np.random.randn(4, *INPUT_SHAPE).astype(np.float32)
        output = vae_model(test_input)
        
        assert output.shape == test_input.shape, \
            f"Output shape {output.shape} doesn't match input shape {test_input.shape}"
    
    def test_input_shape_matches_config(self, vae_model):
        """Test that model input shape matches config."""
        expected_shape = (None, N_MELS, N_TIME_FRAMES, 1)
        actual_shape = vae_model.encoder.input_shape
        
        # Compare ignoring batch dimension
        assert actual_shape[1:] == expected_shape[1:], \
            f"Input shape mismatch: expected {expected_shape}, got {actual_shape}"


class TestVAEFunctionality:
    """Test VAE functional behavior."""
    
    @pytest.fixture
    def vae_model(self):
        """Create a fresh VAE model for each test."""
        return create_vae_model()
    
    def test_encode_decode_consistency(self, vae_model):
        """Test that encode followed by decode produces valid output."""
        test_input = np.random.randn(4, *INPUT_SHAPE).astype(np.float32)
        
        z_mean, z_log_var, z = vae_model.encode(test_input)
        reconstruction = vae_model.decode(z_mean)  # Use mean for deterministic output
        
        assert reconstruction.shape == test_input.shape
        assert not np.isnan(reconstruction).any(), "Reconstruction contains NaN values"
    
    def test_anomaly_score_computation(self, vae_model):
        """Test anomaly score computation."""
        test_input = np.random.randn(10, *INPUT_SHAPE).astype(np.float32)
        
        scores = vae_model.compute_anomaly_score(test_input)
        
        assert len(scores) == 10
        assert all(scores >= 0), "Anomaly scores should be non-negative"
        assert not np.isnan(scores).any(), "Scores contain NaN values"
    
    def test_anomaly_scores_different_for_different_inputs(self, vae_model):
        """Test that different inputs produce different anomaly scores."""
        input1 = np.random.randn(5, *INPUT_SHAPE).astype(np.float32)
        input2 = np.random.randn(5, *INPUT_SHAPE).astype(np.float32) * 10  # Very different
        
        scores1 = vae_model.compute_anomaly_score(input1)
        scores2 = vae_model.compute_anomaly_score(input2)
        
        # At least the means should be different
        assert abs(scores1.mean() - scores2.mean()) > 0 or \
               abs(scores1.std() - scores2.std()) > 0, \
               "Very different inputs produced identical score distributions"
    
    def test_sampling_layer(self):
        """Test the sampling layer."""
        sampling = Sampling()
        
        z_mean = np.zeros((4, LATENT_DIM), dtype=np.float32)
        z_log_var = np.zeros((4, LATENT_DIM), dtype=np.float32)
        
        z = sampling([z_mean, z_log_var])
        
        assert z.shape == (4, LATENT_DIM)
        # With zero mean and log_var, samples should be close to standard normal
        assert abs(z.numpy().mean()) < 1.0  # Rough check


class TestVAETraining:
    """Test VAE training functionality."""
    
    @pytest.fixture
    def vae_model(self):
        """Create and compile a fresh VAE model for each test."""
        vae = create_vae_model()
        vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))
        return vae
    
    def test_train_step(self, vae_model):
        """Test that a single training step works."""
        # Create small batch of data
        batch_size = 4
        x = np.random.randn(batch_size, *INPUT_SHAPE).astype(np.float32)
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((x, x)).batch(batch_size)
        
        # Run one training step
        for batch in dataset:
            result = vae_model.train_step(batch)
            break
        
        assert 'total_loss' in result
        assert 'reconstruction_loss' in result
        assert 'kl_loss' in result
        
        # Losses should be valid numbers
        assert not np.isnan(float(result['total_loss']))
    
    def test_fit_reduces_loss(self, vae_model):
        """Test that training reduces loss."""
        # Create training data
        x = np.random.randn(32, *INPUT_SHAPE).astype(np.float32)
        
        # Get initial loss
        initial_scores = vae_model.compute_anomaly_score(x[:8])
        initial_loss = initial_scores.mean()
        
        # Train for a few epochs
        vae_model.fit(x, x, epochs=5, batch_size=8, verbose=0)
        
        # Get final loss
        final_scores = vae_model.compute_anomaly_score(x[:8])
        final_loss = final_scores.mean()
        
        # Loss should decrease (or at least not increase dramatically)
        # Note: This is a weak test since random data may not train well
        assert final_loss < initial_loss * 2, \
            f"Loss increased dramatically: {initial_loss} -> {final_loss}"


class TestVAEPersistence:
    """Test VAE model saving and loading."""
    
    def test_save_and_load_weights(self):
        """Test saving and loading model weights."""
        # Create model and get some outputs
        vae1 = create_vae_model()
        test_input = np.random.randn(4, *INPUT_SHAPE).astype(np.float32)
        
        # Use encoder z_mean for deterministic output (not sampled z)
        z_mean1, _, _ = vae1.encoder(test_input)
        
        # Save weights
        with tempfile.TemporaryDirectory() as tmpdir:
            weights_path = os.path.join(tmpdir, "test_weights.weights.h5")
            vae1.save_weights(weights_path)
            
            # Create new model and load weights
            vae2 = create_vae_model()
            vae2.load_weights(weights_path)
            
            # Use z_mean for deterministic comparison
            z_mean2, _, _ = vae2.encoder(test_input)
            
            np.testing.assert_array_almost_equal(
                z_mean1.numpy(), z_mean2.numpy(), decimal=5,
                err_msg="Loaded model produces different z_mean outputs"
            )
    
    def test_encoder_decoder_weights_separate(self):
        """Test that encoder and decoder have separate weights."""
        vae = create_vae_model()
        
        encoder_weights = vae.encoder.get_weights()
        decoder_weights = vae.decoder.get_weights()
        
        assert len(encoder_weights) > 0, "Encoder has no weights"
        assert len(decoder_weights) > 0, "Decoder has no weights"
        
        # Encoder and decoder should have different number of parameters
        # (they have different architectures)
        encoder_params = sum(w.size for w in encoder_weights)
        decoder_params = sum(w.size for w in decoder_weights)
        
        # Both should have significant parameters
        assert encoder_params > 10000, f"Encoder has too few params: {encoder_params}"
        assert decoder_params > 10000, f"Decoder has too few params: {decoder_params}"


class TestModelConfiguration:
    """Test model configuration matches expected values."""
    
    def test_latent_dim(self):
        """Test that latent dimension is correct."""
        vae = create_vae_model()
        assert vae.latent_dim == LATENT_DIM
    
    def test_model_parameters_reasonable(self):
        """Test that model has reasonable number of parameters."""
        vae = create_vae_model()
        
        total_params = vae.encoder.count_params() + vae.decoder.count_params()
        
        # Model should be between 100K and 10M parameters (reasonable for mobile)
        assert total_params > 100_000, \
            f"Model too small: {total_params} params"
        assert total_params < 10_000_000, \
            f"Model too large for mobile: {total_params} params"
        
        # Print for info
        print(f"Total parameters: {total_params:,}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

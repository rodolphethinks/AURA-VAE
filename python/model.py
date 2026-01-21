"""
AURA-VAE Model Architecture

Variational Autoencoder (VAE) for acoustic anomaly detection.
Based on DCASE challenge methodologies.

The model:
1. Encodes mel spectrograms to a low-dimensional latent space
2. Reconstructs the input from the latent representation
3. Anomaly detection via reconstruction error

Architecture optimized for:
- Small model size (< 1 MB TFLite)
- Fast inference on mobile CPU
- Good reconstruction of normal sounds
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from typing import Tuple, Dict

from config import (
    INPUT_SHAPE, LATENT_DIM,
    ENCODER_FILTERS, ENCODER_KERNELS, ENCODER_STRIDES, ENCODER_DENSE,
    DECODER_DENSE, DECODER_FILTERS, DECODER_KERNELS, DECODER_STRIDES,
    RECONSTRUCTION_WEIGHT, KL_WEIGHT
)


class Sampling(layers.Layer):
    """
    Sampling layer for VAE.
    
    Uses the reparameterization trick: z = mean + std * epsilon
    where epsilon ~ N(0, 1)
    """
    
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def build_encoder(input_shape: Tuple[int, ...] = INPUT_SHAPE,
                  latent_dim: int = LATENT_DIM,
                  name: str = "encoder") -> Model:
    """
    Build the VAE encoder network.
    
    Architecture:
    - 3 convolutional blocks with batch norm and LeakyReLU
    - Flatten and dense layer
    - Output: z_mean, z_log_var, z (sampled)
    
    Args:
        input_shape: Shape of input spectrograms
        latent_dim: Dimension of latent space
        name: Model name
    
    Returns:
        Keras Model
    """
    inputs = keras.Input(shape=input_shape, name="encoder_input")
    x = inputs
    
    # Convolutional blocks
    for i, (filters, kernel, stride) in enumerate(zip(ENCODER_FILTERS, ENCODER_KERNELS, ENCODER_STRIDES)):
        x = layers.Conv2D(
            filters=filters,
            kernel_size=kernel,
            strides=stride,
            padding="same",
            name=f"encoder_conv_{i}"
        )(x)
        x = layers.BatchNormalization(name=f"encoder_bn_{i}")(x)
        x = layers.LeakyReLU(alpha=0.2, name=f"encoder_leaky_{i}")(x)
    
    # Flatten and dense
    x = layers.Flatten(name="encoder_flatten")(x)
    x = layers.Dense(ENCODER_DENSE, name="encoder_dense")(x)
    x = layers.LeakyReLU(alpha=0.2, name="encoder_dense_leaky")(x)
    
    # Latent space parameters
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    
    # Sampling
    z = Sampling(name="z_sampling")([z_mean, z_log_var])
    
    encoder = Model(inputs, [z_mean, z_log_var, z], name=name)
    return encoder


def build_decoder(latent_dim: int = LATENT_DIM,
                  output_shape: Tuple[int, ...] = INPUT_SHAPE,
                  name: str = "decoder") -> Model:
    """
    Build the VAE decoder network.
    
    Architecture mirrors encoder:
    - Dense layer to expand latent
    - Reshape to feature maps
    - 3 transposed convolution blocks
    - Output: reconstructed spectrogram
    
    Args:
        latent_dim: Dimension of latent space
        output_shape: Shape of output spectrograms
        name: Model name
    
    Returns:
        Keras Model
    """
    # Calculate shape before flatten in encoder
    # After 3 conv layers with stride 2: (64, 32, 1) -> (8, 4, 128)
    h = output_shape[0] // (2 ** len(ENCODER_FILTERS))  # 64 / 8 = 8
    w = output_shape[1] // (2 ** len(ENCODER_FILTERS))  # 32 / 8 = 4
    c = ENCODER_FILTERS[-1]  # 128
    
    inputs = keras.Input(shape=(latent_dim,), name="decoder_input")
    
    # Dense and reshape
    x = layers.Dense(DECODER_DENSE, name="decoder_dense")(inputs)
    x = layers.LeakyReLU(alpha=0.2, name="decoder_dense_leaky")(x)
    x = layers.Dense(h * w * c, name="decoder_dense_expand")(x)
    x = layers.LeakyReLU(alpha=0.2, name="decoder_expand_leaky")(x)
    x = layers.Reshape((h, w, c), name="decoder_reshape")(x)
    
    # Transposed convolution blocks
    for i, (filters, kernel, stride) in enumerate(zip(DECODER_FILTERS, DECODER_KERNELS, DECODER_STRIDES)):
        x = layers.Conv2DTranspose(
            filters=filters,
            kernel_size=kernel,
            strides=stride,
            padding="same",
            name=f"decoder_conv_t_{i}"
        )(x)
        x = layers.BatchNormalization(name=f"decoder_bn_{i}")(x)
        x = layers.LeakyReLU(alpha=0.2, name=f"decoder_leaky_{i}")(x)
    
    # Output layer
    outputs = layers.Conv2DTranspose(
        filters=output_shape[-1],  # 1 channel
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        activation="linear",  # Linear for reconstruction
        name="decoder_output"
    )(x)
    
    decoder = Model(inputs, outputs, name=name)
    return decoder


class VAE(Model):
    """
    Variational Autoencoder for acoustic anomaly detection.
    
    Loss = Reconstruction Loss + KL Divergence
    
    Anomaly score = MSE between input and reconstruction
    """
    
    def __init__(self, 
                 input_shape: Tuple[int, ...] = INPUT_SHAPE,
                 latent_dim: int = LATENT_DIM,
                 reconstruction_weight: float = RECONSTRUCTION_WEIGHT,
                 kl_weight: float = KL_WEIGHT,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.latent_dim = latent_dim
        self.reconstruction_weight = reconstruction_weight
        self.kl_weight = kl_weight
        
        # Build encoder and decoder
        self.encoder = build_encoder(input_shape, latent_dim)
        self.decoder = build_decoder(latent_dim, input_shape)
        
        # Metrics
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
    
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]
    
    def call(self, inputs, training=False):
        """Forward pass through VAE."""
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        return reconstruction
    
    def encode(self, inputs):
        """Encode inputs to latent space."""
        z_mean, z_log_var, z = self.encoder(inputs)
        return z_mean, z_log_var, z
    
    def decode(self, z):
        """Decode latent vectors to spectrograms."""
        return self.decoder(z)
    
    def train_step(self, data):
        """Custom training step."""
        x, _ = data  # For VAE, target is same as input
        
        with tf.GradientTape() as tape:
            # Forward pass
            z_mean, z_log_var, z = self.encoder(x, training=True)
            reconstruction = self.decoder(z, training=True)
            
            # Reconstruction loss (MSE)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.square(x - reconstruction),
                    axis=(1, 2, 3)
                )
            )
            
            # KL divergence loss
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(
                    1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
                    axis=1
                )
            )
            
            # Total loss
            total_loss = (self.reconstruction_weight * reconstruction_loss + 
                         self.kl_weight * kl_loss)
        
        # Compute gradients
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            "total_loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    
    def test_step(self, data):
        """Custom test step."""
        x, _ = data
        
        z_mean, z_log_var, z = self.encoder(x, training=False)
        reconstruction = self.decoder(z, training=False)
        
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                tf.square(x - reconstruction),
                axis=(1, 2, 3)
            )
        )
        
        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
                axis=1
            )
        )
        
        total_loss = (self.reconstruction_weight * reconstruction_loss + 
                     self.kl_weight * kl_loss)
        
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            "total_loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    
    def compute_anomaly_score(self, inputs: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores for inputs.
        
        Anomaly score = Mean Squared Error between input and reconstruction
        
        Args:
            inputs: Input spectrograms of shape (n_samples, mel_bins, time_frames, 1)
        
        Returns:
            Array of anomaly scores
        """
        reconstructions = self.predict(inputs, verbose=0)
        
        # MSE per sample
        mse = np.mean((inputs - reconstructions) ** 2, axis=(1, 2, 3))
        
        return mse
    
    def get_config(self):
        return {
            "latent_dim": self.latent_dim,
            "reconstruction_weight": self.reconstruction_weight,
            "kl_weight": self.kl_weight,
        }


def create_vae_model(input_shape: Tuple[int, ...] = INPUT_SHAPE,
                     latent_dim: int = LATENT_DIM) -> VAE:
    """
    Factory function to create VAE model.
    
    Args:
        input_shape: Shape of input spectrograms
        latent_dim: Dimension of latent space
    
    Returns:
        Compiled VAE model
    """
    vae = VAE(input_shape=input_shape, latent_dim=latent_dim)
    
    # Build model by calling it
    dummy_input = np.zeros((1,) + input_shape, dtype=np.float32)
    _ = vae(dummy_input)
    
    return vae


def get_model_summary(model: VAE) -> str:
    """Get detailed model summary."""
    import io
    
    stream = io.StringIO()
    model.encoder.summary(print_fn=lambda x: stream.write(x + '\n'))
    stream.write('\n' + '='*60 + '\n\n')
    model.decoder.summary(print_fn=lambda x: stream.write(x + '\n'))
    
    return stream.getvalue()


if __name__ == "__main__":
    # Test model creation
    print("Creating VAE model...")
    vae = create_vae_model()
    
    print("\nEncoder Summary:")
    vae.encoder.summary()
    
    print("\nDecoder Summary:")
    vae.decoder.summary()
    
    # Test forward pass
    test_input = np.random.randn(4, *INPUT_SHAPE).astype(np.float32)
    output = vae(test_input)
    print(f"\nInput shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test anomaly score
    scores = vae.compute_anomaly_score(test_input)
    print(f"Anomaly scores shape: {scores.shape}")
    print(f"Scores: {scores}")
    
    # Count parameters
    total_params = vae.encoder.count_params() + vae.decoder.count_params()
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Estimated size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")

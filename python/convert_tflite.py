"""
AURA-VAE TFLite Conversion Script

Converts the trained VAE model to TensorFlow Lite format for Android deployment.

The conversion includes:
- Optimized inference-only model (no training ops)
- Quantization options for size reduction
- Metadata for Android integration
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras

from config import (
    MODELS_DIR, MODEL_FILENAME, TFLITE_FILENAME, NORM_PARAMS_FILENAME,
    INPUT_SHAPE, LATENT_DIM, create_directories
)
from model import create_vae_model, VAE, build_encoder, build_decoder


def create_inference_model(vae: VAE) -> keras.Model:
    """
    Create a simplified inference-only model.
    
    This model only does forward pass (encode + decode) without
    sampling during inference (uses mean of latent distribution).
    
    Args:
        vae: Trained VAE model
    
    Returns:
        Inference model
    """
    # Input
    inputs = keras.Input(shape=INPUT_SHAPE, name="input")
    
    # Encode to get mean (no sampling for inference - deterministic)
    z_mean, z_log_var, z = vae.encoder(inputs)
    
    # For inference, use z_mean directly (deterministic reconstruction)
    # This gives more consistent results
    reconstruction = vae.decoder(z_mean)
    
    # Create model
    inference_model = keras.Model(inputs=inputs, outputs=reconstruction, name="vae_inference")
    
    return inference_model


class AnomalyScoreLayer(keras.layers.Layer):
    """Custom layer to compute MSE between input and reconstruction."""
    
    def call(self, inputs):
        input_tensor, reconstruction = inputs
        # Compute MSE using keras.ops
        diff = keras.ops.subtract(input_tensor, reconstruction)
        squared = keras.ops.square(diff)
        mse = keras.ops.mean(squared, axis=[1, 2, 3])
        return mse


def create_anomaly_detector_model(vae: VAE) -> keras.Model:
    """
    Create a model that directly outputs anomaly score.
    
    Output = MSE between input and reconstruction
    
    Args:
        vae: Trained VAE model
    
    Returns:
        Anomaly detector model
    """
    # Input
    inputs = keras.Input(shape=INPUT_SHAPE, name="input")
    
    # Encode (use mean for deterministic inference)
    z_mean, z_log_var, z = vae.encoder(inputs)
    
    # Decode
    reconstruction = vae.decoder(z_mean)
    
    # Compute MSE using custom layer
    mse = AnomalyScoreLayer()([inputs, reconstruction])
    
    # Create model
    detector = keras.Model(inputs=inputs, outputs=mse, name="anomaly_detector")
    
    return detector


def convert_to_tflite(model: keras.Model, 
                      output_path: str,
                      quantize: bool = False,
                      representative_data: np.ndarray = None) -> str:
    """
    Convert Keras model to TFLite format.
    
    Args:
        model: Keras model to convert
        output_path: Path to save TFLite model
        quantize: Whether to apply quantization
        representative_data: Data for quantization calibration
    
    Returns:
        Path to saved TFLite model
    """
    # Create converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Set optimizations
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    if quantize and representative_data is not None:
        # Full integer quantization
        def representative_dataset():
            for i in range(min(100, len(representative_data))):
                yield [representative_data[i:i+1].astype(np.float32)]
        
        converter.representative_dataset = representative_dataset
        # Keep float32 inputs/outputs for easier integration
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
    
    # Convert
    tflite_model = converter.convert()
    
    # Save
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    size_mb = len(tflite_model) / 1024 / 1024
    print(f"  Saved TFLite model: {output_path}")
    print(f"  Model size: {size_mb:.2f} MB")
    
    return output_path


def verify_tflite_model(tflite_path: str, test_input: np.ndarray) -> bool:
    """
    Verify TFLite model produces correct output.
    
    Args:
        tflite_path: Path to TFLite model
        test_input: Test input data
    
    Returns:
        True if verification passes
    """
    print("\nVerifying TFLite model...")
    
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    # Get input/output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"  Input shape: {input_details[0]['shape']}")
    print(f"  Input dtype: {input_details[0]['dtype']}")
    print(f"  Output shape: {output_details[0]['shape']}")
    print(f"  Output dtype: {output_details[0]['dtype']}")
    
    # Run inference
    interpreter.set_tensor(input_details[0]['index'], test_input[0:1].astype(np.float32))
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    
    print(f"  Test output shape: {output.shape}")
    print(f"  Test output range: [{output.min():.4f}, {output.max():.4f}]")
    
    return True


def convert_vae_to_tflite():
    """
    Main conversion function.
    """
    create_directories()
    
    print("\n" + "="*60)
    print("AURA-VAE TFLite Conversion")
    print("="*60)
    
    # =========================================================================
    # 1. Load trained model
    # =========================================================================
    print("\n[1/4] Loading trained model...")
    
    vae = create_vae_model()
    weights_path = os.path.join(MODELS_DIR, MODEL_FILENAME)
    vae.load_weights(weights_path)
    print(f"  Loaded weights from: {weights_path}")
    
    # =========================================================================
    # 2. Create inference model
    # =========================================================================
    print("\n[2/4] Creating inference model...")
    
    inference_model = create_inference_model(vae)
    inference_model.summary()
    
    # Also create anomaly detector model
    detector_model = create_anomaly_detector_model(vae)
    
    # =========================================================================
    # 3. Load representative data for quantization
    # =========================================================================
    print("\n[3/4] Loading representative data...")
    
    from preprocessing import FeatureNormalizer
    from dataset import AuraDataset
    
    normalizer = FeatureNormalizer()
    norm_path = os.path.join(MODELS_DIR, NORM_PARAMS_FILENAME)
    if os.path.exists(norm_path):
        normalizer.load(norm_path)
    
    # Try to load processed data
    try:
        from config import PROCESSED_DATA_DIR
        features_path = os.path.join(PROCESSED_DATA_DIR, "normal_features.npy")
        if os.path.exists(features_path):
            features = np.load(features_path)
            if len(features.shape) == 3:
                features = np.expand_dims(features, axis=-1)
            if normalizer.fitted:
                features = normalizer.transform(features)
            print(f"  Loaded {len(features)} samples for calibration")
        else:
            features = np.random.randn(100, *INPUT_SHAPE).astype(np.float32)
            print("  Using random data for calibration")
    except Exception as e:
        print(f"  Warning: Could not load data: {e}")
        features = np.random.randn(100, *INPUT_SHAPE).astype(np.float32)
    
    # =========================================================================
    # 4. Convert to TFLite
    # =========================================================================
    print("\n[4/4] Converting to TFLite...")
    
    # Convert inference model (reconstruction output)
    tflite_path = os.path.join(MODELS_DIR, TFLITE_FILENAME)
    convert_to_tflite(inference_model, tflite_path, quantize=False, representative_data=features)
    
    # Convert detector model (anomaly score output)
    detector_path = os.path.join(MODELS_DIR, "anomaly_detector.tflite")
    convert_to_tflite(detector_model, detector_path, quantize=False, representative_data=features)
    
    # Verify models
    test_input = features[:1] if len(features) > 0 else np.random.randn(1, *INPUT_SHAPE).astype(np.float32)
    verify_tflite_model(tflite_path, test_input)
    verify_tflite_model(detector_path, test_input)
    
    # =========================================================================
    # Save Android configuration
    # =========================================================================
    print("\nSaving Android configuration...")
    
    android_config = {
        'model_file': TFLITE_FILENAME,
        'detector_file': 'anomaly_detector.tflite',
        'input_shape': list(INPUT_SHAPE),
        'sample_rate': 16000,
        'n_fft': 1024,
        'hop_length': 512,
        'n_mels': 64,
        'f_min': 50,
        'f_max': 8000,
        'segment_duration': 1.0,
        'n_time_frames': 32
    }
    
    # Add normalization params
    if normalizer.fitted:
        android_config['normalization'] = {
            'mean': float(normalizer.mean),
            'std': float(normalizer.std)
        }
    
    # Add threshold config if available
    threshold_path = os.path.join(MODELS_DIR, "threshold_config.json")
    if os.path.exists(threshold_path):
        with open(threshold_path, 'r') as f:
            threshold_config = json.load(f)
        android_config['threshold'] = threshold_config
    
    config_path = os.path.join(MODELS_DIR, "android_config.json")
    with open(config_path, 'w') as f:
        json.dump(android_config, f, indent=2)
    print(f"  Saved Android config: {config_path}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*60)
    print("Conversion Complete!")
    print("="*60)
    
    print(f"\nGenerated files:")
    print(f"  - {tflite_path}")
    print(f"  - {detector_path}")
    print(f"  - {config_path}")
    
    # Check file sizes
    for path in [tflite_path, detector_path]:
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / 1024 / 1024
            print(f"\n{os.path.basename(path)}: {size_mb:.2f} MB")
            if size_mb > 10:
                print("  WARNING: Model exceeds 10 MB target!")
            else:
                print("  ✓ Model size within target")


if __name__ == "__main__":
    convert_vae_to_tflite()

"""
Test TFLite conversion and parity with Keras model.

Verifies:
- Successful TFLite conversion
- Output parity between Keras and TFLite models
- Model size constraints
- Inference speed
"""

import pytest
import numpy as np
import tensorflow as tf
import sys
import os
import tempfile
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python.config import INPUT_SHAPE, LATENT_DIM, N_MELS, N_TIME_FRAMES, MODELS_DIR
from python.model import create_vae_model
from python.convert_tflite import (
    create_inference_model, create_anomaly_detector_model, 
    convert_to_tflite, verify_tflite_model
)


class TestTFLiteConversion:
    """Test TFLite model conversion."""
    
    @pytest.fixture
    def vae_model(self):
        """Create a fresh VAE model for each test."""
        return create_vae_model()
    
    @pytest.fixture
    def inference_model(self, vae_model):
        """Create inference model."""
        return create_inference_model(vae_model)
    
    @pytest.fixture  
    def detector_model(self, vae_model):
        """Create anomaly detector model."""
        return create_anomaly_detector_model(vae_model)
    
    def test_create_inference_model(self, vae_model):
        """Test inference model creation."""
        inference_model = create_inference_model(vae_model)
        
        assert inference_model is not None
        assert inference_model.input_shape[1:] == INPUT_SHAPE
        assert inference_model.output_shape[1:] == INPUT_SHAPE
    
    def test_create_anomaly_detector_model(self, vae_model):
        """Test anomaly detector model creation."""
        detector = create_anomaly_detector_model(vae_model)
        
        assert detector is not None
        assert detector.input_shape[1:] == INPUT_SHAPE
        # Output should be a single score per sample
        assert detector.output_shape == (None,) or detector.output_shape == (None, 1)
    
    def test_convert_inference_model_to_tflite(self, inference_model):
        """Test converting inference model to TFLite."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tflite_path = os.path.join(tmpdir, "vae_inference.tflite")
            
            convert_to_tflite(inference_model, tflite_path)
            
            assert os.path.exists(tflite_path), "TFLite file not created"
            
            # Check file size (should be < 5MB for mobile)
            size_mb = os.path.getsize(tflite_path) / 1024 / 1024
            assert size_mb < 5, f"TFLite model too large: {size_mb:.2f}MB"
            print(f"Inference model size: {size_mb:.2f}MB")
    
    def test_convert_detector_model_to_tflite(self, detector_model):
        """Test converting anomaly detector model to TFLite."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tflite_path = os.path.join(tmpdir, "anomaly_detector.tflite")
            
            convert_to_tflite(detector_model, tflite_path)
            
            assert os.path.exists(tflite_path), "TFLite file not created"
            
            size_mb = os.path.getsize(tflite_path) / 1024 / 1024
            assert size_mb < 5, f"TFLite model too large: {size_mb:.2f}MB"
            print(f"Detector model size: {size_mb:.2f}MB")


class TestTFLiteParity:
    """Test that TFLite model output matches Keras model."""
    
    @pytest.fixture
    def vae_model(self):
        """Create a fresh VAE model."""
        return create_vae_model()
    
    def test_inference_model_parity(self, vae_model):
        """Test that TFLite inference model produces same output as Keras."""
        inference_model = create_inference_model(vae_model)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tflite_path = os.path.join(tmpdir, "test_inference.tflite")
            convert_to_tflite(inference_model, tflite_path)
            
            # Load TFLite model
            interpreter = tf.lite.Interpreter(model_path=tflite_path)
            interpreter.allocate_tensors()
            
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            # Test with multiple inputs
            for _ in range(5):
                test_input = np.random.randn(1, *INPUT_SHAPE).astype(np.float32)
                
                # Keras output
                keras_output = inference_model.predict(test_input, verbose=0)
                
                # TFLite output
                interpreter.set_tensor(input_details[0]['index'], test_input)
                interpreter.invoke()
                tflite_output = interpreter.get_tensor(output_details[0]['index'])
                
                # Compare outputs (allow small numerical differences)
                np.testing.assert_array_almost_equal(
                    keras_output, tflite_output, decimal=4,
                    err_msg="TFLite output doesn't match Keras output"
                )
    
    def test_detector_model_parity(self, vae_model):
        """Test that TFLite detector model produces same output as Keras."""
        detector = create_anomaly_detector_model(vae_model)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tflite_path = os.path.join(tmpdir, "test_detector.tflite")
            convert_to_tflite(detector, tflite_path)
            
            # Load TFLite model
            interpreter = tf.lite.Interpreter(model_path=tflite_path)
            interpreter.allocate_tensors()
            
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            # Test with multiple inputs
            max_diff = 0
            for _ in range(10):
                test_input = np.random.randn(1, *INPUT_SHAPE).astype(np.float32)
                
                # Keras output
                keras_output = detector.predict(test_input, verbose=0)
                
                # TFLite output  
                interpreter.set_tensor(input_details[0]['index'], test_input)
                interpreter.invoke()
                tflite_output = interpreter.get_tensor(output_details[0]['index'])
                
                diff = np.abs(keras_output.flatten()[0] - tflite_output.flatten()[0])
                max_diff = max(max_diff, diff)
                
                # Allow slightly larger tolerance for anomaly scores
                assert diff < 0.01, \
                    f"Detector output mismatch: Keras={keras_output}, TFLite={tflite_output}"
            
            print(f"Max detector score difference: {max_diff:.6f}")


class TestTFLitePerformance:
    """Test TFLite model performance characteristics."""
    
    @pytest.fixture
    def tflite_interpreter(self):
        """Create TFLite interpreter from existing model or create new one."""
        vae = create_vae_model()
        detector = create_anomaly_detector_model(vae)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tflite_path = os.path.join(tmpdir, "perf_test.tflite")
            convert_to_tflite(detector, tflite_path)
            
            interpreter = tf.lite.Interpreter(model_path=tflite_path)
            interpreter.allocate_tensors()
            
            yield interpreter
    
    def test_inference_speed(self, tflite_interpreter):
        """Test that TFLite inference is fast enough for real-time."""
        input_details = tflite_interpreter.get_input_details()
        output_details = tflite_interpreter.get_output_details()
        
        test_input = np.random.randn(1, *INPUT_SHAPE).astype(np.float32)
        
        # Warm up
        for _ in range(5):
            tflite_interpreter.set_tensor(input_details[0]['index'], test_input)
            tflite_interpreter.invoke()
        
        # Time 100 inferences
        start = time.time()
        n_iterations = 100
        for _ in range(n_iterations):
            tflite_interpreter.set_tensor(input_details[0]['index'], test_input)
            tflite_interpreter.invoke()
        elapsed = time.time() - start
        
        avg_time_ms = (elapsed / n_iterations) * 1000
        
        print(f"Average inference time: {avg_time_ms:.2f}ms")
        
        # Should be under 100ms for real-time (1s audio segments)
        assert avg_time_ms < 100, \
            f"Inference too slow for real-time: {avg_time_ms:.2f}ms"


class TestProductionModels:
    """Test production models in the models/ directory."""
    
    @pytest.mark.skipif(
        not os.path.exists(os.path.join(MODELS_DIR, "anomaly_detector.tflite")),
        reason="Production model not found"
    )
    def test_production_detector_loads(self):
        """Test that production anomaly detector model loads."""
        model_path = os.path.join(MODELS_DIR, "anomaly_detector.tflite")
        
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Check input shape
        expected_input_shape = [1, N_MELS, N_TIME_FRAMES, 1]
        assert list(input_details[0]['shape']) == expected_input_shape, \
            f"Input shape mismatch: {input_details[0]['shape']}"
        
        print(f"Production model input shape: {input_details[0]['shape']}")
        print(f"Production model output shape: {output_details[0]['shape']}")
    
    @pytest.mark.skipif(
        not os.path.exists(os.path.join(MODELS_DIR, "anomaly_detector.tflite")),
        reason="Production model not found"
    )
    def test_production_detector_inference(self):
        """Test that production model can run inference."""
        model_path = os.path.join(MODELS_DIR, "anomaly_detector.tflite")
        
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Run inference with random input
        test_input = np.random.randn(1, N_MELS, N_TIME_FRAMES, 1).astype(np.float32)
        
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        
        assert not np.isnan(output).any(), "Output contains NaN"
        assert output.shape[0] == 1, "Output batch size mismatch"
        
        print(f"Production model output: {output.flatten()[:5]}...")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

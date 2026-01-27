
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Add valid path
sys.path.append(os.getcwd())
from python.config import *
from python.model import create_vae_model

def check_weights():
    print(f"TensorFlow Version: {tf.__version__}")
    
    # 1. Create fresh model
    vae = create_vae_model()
    
    # 2. Touch a weight to ensure initialization
    # Get initial weights of a specific layer, e.g., the first dense layer in encoder
    # Note: Structure depends on model.py, usually encoder -> some layer
    encoder = vae.encoder
    # Find a dense layer
    target_layer = None
    for layer in encoder.layers:
        if isinstance(layer, keras.layers.Dense) or isinstance(layer, keras.layers.Conv2D):
            target_layer = layer
            break
            
    if not target_layer:
        print("Could not find a Dense/Conv layer to check.")
        return

    print(f"Checking layer: {target_layer.name}")
    initial_weights = target_layer.get_weights()[0]
    initial_sum = np.sum(initial_weights)
    print(f"Initial weight sum: {initial_sum:.4f}")

    # 3. Load weights
    weights_path = os.path.join("models", "vae_weights.weights.h5")
    if not os.path.exists(weights_path):
        print(f"Weights file not found at {weights_path}")
        return

    print(f"Loading weights from {weights_path} with skip_mismatch=True...")
    try:
        vae.load_weights(weights_path, by_name=True, skip_mismatch=True)
    except Exception as e:
        print(f"Error loading weights: {e}")
        
    # 4. Check if weights changed
    new_weights = target_layer.get_weights()[0]
    new_sum = np.sum(new_weights)
    print(f"New weight sum: {new_sum:.4f}")
    
    if initial_sum == new_sum:
        print("WARNING: Weights did NOT change. skip_mismatch might have ignored them.")
        print("Conclusion: Model needs rebuilding/retraining or correct weight mapping.")
    else:
        print("SUCCESS: Weights changed. Loading appears effective.")
        print("Conclusion: Model is likely valid.")

if __name__ == "__main__":
    check_weights()

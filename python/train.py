"""
AURA-VAE Training Script

Full training pipeline including:
- Data loading and preprocessing
- Model creation and compilation
- Training with callbacks
- Model saving
- Training history logging
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
)

from config import (
    MODELS_DIR, RESULTS_DIR, PLOTS_DIR,
    MODEL_FILENAME, NORM_PARAMS_FILENAME, TRAINING_HISTORY_FILENAME,
    EPOCHS, BATCH_SIZE, LEARNING_RATE,
    EARLY_STOPPING_PATIENCE, MIN_DELTA,
    LR_REDUCE_FACTOR, LR_REDUCE_PATIENCE, LR_MIN,
    create_directories
)
from preprocessing import FeatureNormalizer
from dataset import AuraDataset
from model import create_vae_model, VAE


def setup_callbacks(model_path: str, log_dir: str) -> list:
    """
    Setup training callbacks.
    
    Args:
        model_path: Path to save best model
        log_dir: TensorBoard log directory
    
    Returns:
        List of callbacks
    """
    callbacks = [
        # Save best model
        ModelCheckpoint(
            model_path,
            monitor='val_total_loss',
            mode='min',
            save_best_only=True,
            save_weights_only=True,  # Save weights only for custom model
            verbose=1
        ),
        
        # Early stopping
        EarlyStopping(
            monitor='val_total_loss',
            mode='min',
            patience=EARLY_STOPPING_PATIENCE,
            min_delta=MIN_DELTA,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Learning rate reduction
        ReduceLROnPlateau(
            monitor='val_total_loss',
            mode='min',
            factor=LR_REDUCE_FACTOR,
            patience=LR_REDUCE_PATIENCE,
            min_lr=LR_MIN,
            verbose=1
        ),
        
        # TensorBoard logging
        TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True
        )
    ]
    
    return callbacks


def plot_training_history(history: dict, save_path: str):
    """
    Plot and save training history.
    
    Args:
        history: Training history dictionary
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Total loss
    axes[0].plot(history['total_loss'], label='Train', linewidth=2)
    axes[0].plot(history['val_total_loss'], label='Validation', linewidth=2)
    axes[0].set_title('Total Loss', fontsize=12)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Reconstruction loss
    axes[1].plot(history['reconstruction_loss'], label='Train', linewidth=2)
    axes[1].plot(history['val_reconstruction_loss'], label='Validation', linewidth=2)
    axes[1].set_title('Reconstruction Loss', fontsize=12)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # KL loss
    axes[2].plot(history['kl_loss'], label='Train', linewidth=2)
    axes[2].plot(history['val_kl_loss'], label='Validation', linewidth=2)
    axes[2].set_title('KL Divergence Loss', fontsize=12)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training history plot saved to: {save_path}")


def train_vae(output_dir=None):
    """
    Main training function.
    
    Args:
        output_dir: Directory to save model/metrics (optional)

    Pipeline:
    1. Create directories
    2. Load and prepare data
    3. Create model
    4. Train
    5. Save model and artifacts
    """
    # Determine output directories
    if output_dir:
        models_dir = os.path.join(output_dir, "models")
        plots_dir = os.path.join(output_dir, "plots")
        logs_root_dir = os.path.join(output_dir, "logs")
    else:
        models_dir = MODELS_DIR
        plots_dir = PLOTS_DIR
        logs_root_dir = os.path.join(RESULTS_DIR, "logs")
        
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(logs_root_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("AURA-VAE Training Pipeline")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # =========================================================================
    # 1. Load and prepare data
    # =========================================================================
    print("\n[1/5] Loading and preparing data...")
    
    normalizer = FeatureNormalizer()
    dataset = AuraDataset(normalizer=normalizer)
    
    # Load features
    # Note: dataset.load_features uses default PROCESSED_DATA_DIR.
    # If run_pipeline copied features there, this works fine.
    normal_features, anomaly_features = dataset.load_features()
    
    # Prepare data
    train_data, test_data = dataset.prepare_data(normal_features, anomaly_features)
    
    # Save normalization parameters
    norm_path = os.path.join(models_dir, NORM_PARAMS_FILENAME)
    normalizer.save(norm_path)
    
    # Create TensorFlow datasets
    train_dataset = dataset.create_tf_dataset(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_dataset = dataset.create_tf_dataset(test_data, batch_size=BATCH_SIZE, shuffle=False)
    
    # =========================================================================
    # 2. Create model
    # =========================================================================
    print("\n[2/5] Creating VAE model...")
    
    vae = create_vae_model()
    
    # Compile model
    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE))
    
    # Print model info
    print(f"  Input shape: {vae.encoder.input_shape}")
    print(f"  Latent dimension: {vae.latent_dim}")
    total_params = vae.encoder.count_params() + vae.decoder.count_params()
    print(f"  Total parameters: {total_params:,}")
    
    # =========================================================================
    # 3. Setup training
    # =========================================================================
    print("\n[3/5] Setting up training...")
    
    # Model save path
    weights_path = os.path.join(models_dir, "vae_weights.weights.h5")
    
    # TensorBoard log directory
    log_dir = os.path.join(logs_root_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
    
    # Callbacks
    callbacks = setup_callbacks(weights_path, log_dir)
    
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Early stopping patience: {EARLY_STOPPING_PATIENCE}")
    
    # =========================================================================
    # 4. Train model
    # =========================================================================
    print("\n[4/5] Training model...")
    print("-"*60)
    
    history = vae.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    print("-"*60)
    
    # =========================================================================
    # 5. Save results
    # =========================================================================
    print("\n[5/5] Saving results...")
    
    # Load best weights
    vae.load_weights(weights_path)
    
    # Save full model (for TFLite conversion)
    model_path = os.path.join(models_dir, MODEL_FILENAME)
    vae.save_weights(model_path)
    print(f"  Model weights saved to: {model_path}")
    
    # Save training history
    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    history_path = os.path.join(models_dir, TRAINING_HISTORY_FILENAME)
    with open(history_path, 'w') as f:
        json.dump(history_dict, f, indent=2)
    print(f"  Training history saved to: {history_path}")
    
    # Plot training history
    plot_path = os.path.join(plots_dir, "training_history.png")
    plot_training_history(history_dict, plot_path)
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    
    final_train_loss = history_dict['total_loss'][-1]
    final_val_loss = history_dict['val_total_loss'][-1]
    best_val_loss = min(history_dict['val_total_loss'])
    best_epoch = history_dict['val_total_loss'].index(best_val_loss) + 1
    
    print(f"\nResults:")
    print(f"  Final training loss: {final_train_loss:.6f}")
    print(f"  Final validation loss: {final_val_loss:.6f}")
    print(f"  Best validation loss: {best_val_loss:.6f} (epoch {best_epoch})")
    print(f"  Total epochs trained: {len(history_dict['total_loss'])}")
    
    print(f"\nArtifacts saved:")
    print(f"  - Model weights: {model_path}")
    print(f"  - Normalization params: {norm_path}")
    print(f"  - Training history: {history_path}")
    print(f"  - Training plot: {plot_path}")
    print(f"  - TensorBoard logs: {log_dir}")
    
    return vae, history_dict, normalizer


if __name__ == "__main__":
    # Set memory growth for GPU (if available)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU(s) available: {len(gpus)}")
        except RuntimeError as e:
            print(e)
    else:
        print("No GPU detected, using CPU")
    
    # Train model
    vae, history, normalizer = train_vae()

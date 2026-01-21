"""
AURA-VAE Dataset Module

Handles dataset loading, splitting, and batching for training and evaluation.
"""

import os
import numpy as np
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split

from config import (
    PROCESSED_DATA_DIR, TRAIN_SPLIT, TEST_SPLIT,
    INPUT_SHAPE, BATCH_SIZE
)


class AuraDataset:
    """
    Dataset class for AURA-VAE training and evaluation.
    
    Handles:
    - Loading preprocessed features
    - Train/test splitting
    - Normalization
    - Batching
    """
    
    def __init__(self, normalizer=None):
        """
        Initialize dataset.
        
        Args:
            normalizer: Optional FeatureNormalizer instance
        """
        self.normalizer = normalizer
        self.train_data = None
        self.test_data = None
        self.anomaly_data = None
        self.is_loaded = False
    
    def load_features(self, 
                      features_path: Optional[str] = None,
                      anomalies_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load preprocessed features from disk.
        
        Args:
            features_path: Path to normal features file
            anomalies_path: Path to synthetic anomalies file
        
        Returns:
            Tuple of (normal_features, anomaly_features)
        """
        if features_path is None:
            features_path = os.path.join(PROCESSED_DATA_DIR, "normal_features.npy")
        
        if anomalies_path is None:
            anomalies_path = os.path.join(PROCESSED_DATA_DIR, "synthetic_anomalies.npy")
        
        print("Loading features...")
        
        # Load normal features
        if os.path.exists(features_path):
            normal_features = np.load(features_path)
            print(f"  Normal features: {normal_features.shape}")
        else:
            raise FileNotFoundError(f"Normal features not found: {features_path}")
        
        # Load anomaly features (optional)
        if os.path.exists(anomalies_path):
            anomaly_features = np.load(anomalies_path)
            print(f"  Anomaly features: {anomaly_features.shape}")
        else:
            print("  No anomaly features found (will generate during evaluation)")
            anomaly_features = None
        
        self.is_loaded = True
        return normal_features, anomaly_features
    
    def prepare_data(self, 
                     normal_features: np.ndarray,
                     anomaly_features: Optional[np.ndarray] = None,
                     train_split: float = TRAIN_SPLIT,
                     random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for training and evaluation.
        
        Steps:
        1. Add channel dimension
        2. Split into train/test
        3. Normalize using training data statistics
        
        Args:
            normal_features: Normal audio features
            anomaly_features: Anomaly features for evaluation
            train_split: Fraction for training
            random_state: Random seed for reproducibility
        
        Returns:
            Tuple of (train_data, test_data)
        """
        print("\nPreparing data...")
        
        # Add channel dimension if needed (n_samples, mel_bins, time_frames) -> (n_samples, mel_bins, time_frames, 1)
        if len(normal_features.shape) == 3:
            normal_features = np.expand_dims(normal_features, axis=-1)
        
        print(f"  Feature shape: {normal_features.shape}")
        
        # Split into train/test
        self.train_data, self.test_data = train_test_split(
            normal_features,
            train_size=train_split,
            random_state=random_state,
            shuffle=True
        )
        
        print(f"  Training samples: {len(self.train_data)}")
        print(f"  Test samples: {len(self.test_data)}")
        
        # Handle anomaly data
        if anomaly_features is not None:
            if len(anomaly_features.shape) == 3:
                anomaly_features = np.expand_dims(anomaly_features, axis=-1)
            self.anomaly_data = anomaly_features
            print(f"  Anomaly samples: {len(self.anomaly_data)}")
        
        # Normalize using training data
        if self.normalizer is not None:
            print("\n  Fitting normalizer on training data...")
            self.train_data = self.normalizer.fit_transform(self.train_data)
            self.test_data = self.normalizer.transform(self.test_data)
            
            if self.anomaly_data is not None:
                self.anomaly_data = self.normalizer.transform(self.anomaly_data)
        
        return self.train_data, self.test_data
    
    def get_train_data(self) -> np.ndarray:
        """Get training data."""
        if self.train_data is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")
        return self.train_data
    
    def get_test_data(self) -> np.ndarray:
        """Get test data."""
        if self.test_data is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")
        return self.test_data
    
    def get_anomaly_data(self) -> Optional[np.ndarray]:
        """Get anomaly data."""
        return self.anomaly_data
    
    def create_tf_dataset(self, 
                          data: np.ndarray, 
                          batch_size: int = BATCH_SIZE,
                          shuffle: bool = True,
                          buffer_size: int = 1000):
        """
        Create TensorFlow dataset for efficient training.
        
        Args:
            data: Input data array
            batch_size: Batch size
            shuffle: Whether to shuffle
            buffer_size: Shuffle buffer size
        
        Returns:
            tf.data.Dataset
        """
        import tensorflow as tf
        
        dataset = tf.data.Dataset.from_tensor_slices((data, data))  # VAE: input = target
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size)
        
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        return dataset


def load_and_prepare_dataset(normalizer=None) -> Tuple[AuraDataset, np.ndarray, np.ndarray]:
    """
    Convenience function to load and prepare the full dataset.
    
    Args:
        normalizer: Optional FeatureNormalizer
    
    Returns:
        Tuple of (dataset, train_data, test_data)
    """
    dataset = AuraDataset(normalizer=normalizer)
    normal_features, anomaly_features = dataset.load_features()
    train_data, test_data = dataset.prepare_data(normal_features, anomaly_features)
    
    return dataset, train_data, test_data


if __name__ == "__main__":
    # Test dataset loading
    from preprocessing import FeatureNormalizer
    
    normalizer = FeatureNormalizer()
    dataset, train_data, test_data = load_and_prepare_dataset(normalizer)
    
    print("\nDataset Statistics:")
    print(f"  Train - min: {train_data.min():.4f}, max: {train_data.max():.4f}")
    print(f"  Train - mean: {train_data.mean():.4f}, std: {train_data.std():.4f}")
    print(f"  Test - min: {test_data.min():.4f}, max: {test_data.max():.4f}")
    
    anomaly_data = dataset.get_anomaly_data()
    if anomaly_data is not None:
        print(f"  Anomaly - min: {anomaly_data.min():.4f}, max: {anomaly_data.max():.4f}")

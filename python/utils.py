"""
AURA-VAE Utility Functions

Common utility functions used across the project.
"""

import os
import json
import numpy as np
from typing import Dict, Any, Optional


def save_json(data: Dict[str, Any], file_path: str):
    """Save dictionary to JSON file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(file_path: str) -> Dict[str, Any]:
    """Load dictionary from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    import random
    import tensorflow as tf
    
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def get_available_devices():
    """Get available compute devices."""
    import tensorflow as tf
    
    devices = {
        'gpus': tf.config.list_physical_devices('GPU'),
        'cpus': tf.config.list_physical_devices('CPU')
    }
    return devices


def print_gpu_info():
    """Print GPU information if available."""
    import tensorflow as tf
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPUs available: {len(gpus)}")
        for gpu in gpus:
            print(f"  {gpu}")
    else:
        print("No GPU available, using CPU")


def calculate_model_size(model) -> float:
    """Calculate model size in MB."""
    total_params = model.count_params()
    size_mb = total_params * 4 / 1024 / 1024  # float32 = 4 bytes
    return size_mb


def format_duration(seconds: float) -> str:
    """Format seconds as human-readable duration."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{int(minutes)}m {int(secs)}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{int(hours)}h {int(minutes)}m"


class Timer:
    """Simple timer for measuring execution time."""
    
    def __init__(self):
        import time
        self.time = time
        self.start_time = None
        self.end_time = None
    
    def start(self):
        self.start_time = self.time.time()
        return self
    
    def stop(self):
        self.end_time = self.time.time()
        return self.elapsed
    
    @property
    def elapsed(self) -> float:
        if self.start_time is None:
            return 0
        end = self.end_time if self.end_time else self.time.time()
        return end - self.start_time
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, *args):
        self.stop()


class ProgressLogger:
    """Simple progress logger."""
    
    def __init__(self, total: int, description: str = "Progress"):
        self.total = total
        self.description = description
        self.current = 0
    
    def update(self, n: int = 1):
        self.current += n
        percent = self.current / self.total * 100
        print(f"\r{self.description}: {self.current}/{self.total} ({percent:.1f}%)", end="")
        if self.current >= self.total:
            print()
    
    def reset(self):
        self.current = 0

"""
AURA-VAE Evaluation Script

Comprehensive evaluation including:
- Reconstruction error analysis
- Anomaly score distribution
- Threshold calculation
- Normal vs anomaly comparison
- ROC curve analysis
- Visualization of results
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from typing import Tuple, Dict, Optional

from config import (
    MODELS_DIR, RESULTS_DIR, PLOTS_DIR, METRICS_DIR, PROCESSED_DATA_DIR,
    MODEL_FILENAME, NORM_PARAMS_FILENAME,
    ANOMALY_THRESHOLD_K, ANOMALY_THRESHOLD_PERCENTILE, ANOMALY_THRESHOLD_OVERRIDE,
    INPUT_SHAPE, create_directories
)
from preprocessing import FeatureNormalizer, generate_synthetic_anomalies
from dataset import AuraDataset
from model import create_vae_model, VAE


def load_trained_model() -> Tuple[VAE, FeatureNormalizer]:
    """
    Load trained VAE model and normalizer.
    
    Returns:
        Tuple of (model, normalizer)
    """
    print("Loading trained model...")
    
    # Load normalizer
    normalizer = FeatureNormalizer()
    norm_path = os.path.join(MODELS_DIR, NORM_PARAMS_FILENAME)
    normalizer.load(norm_path)
    
    # Create and load model
    vae = create_vae_model()
    weights_path = os.path.join(MODELS_DIR, MODEL_FILENAME)
    vae.load_weights(weights_path)
    print(f"  Loaded weights from: {weights_path}")
    
    return vae, normalizer


def compute_threshold(scores: np.ndarray, 
                      method: str = 'std',
                      k: float = ANOMALY_THRESHOLD_K,
                      percentile: float = ANOMALY_THRESHOLD_PERCENTILE) -> float:
    """
    Compute anomaly threshold from normal data scores.
    
    Args:
        scores: Anomaly scores from normal data
        method: 'std' for mean + k*std, 'percentile' for percentile-based
        k: Number of standard deviations (for std method)
        percentile: Percentile value (for percentile method)
    
    Returns:
        Threshold value
    """
    # Check for manual override
    if ANOMALY_THRESHOLD_OVERRIDE is not None:
        print(f"Using manual threshold override: {ANOMALY_THRESHOLD_OVERRIDE}")
        return ANOMALY_THRESHOLD_OVERRIDE

    if method == 'std':
        threshold = np.mean(scores) + k * np.std(scores)
    elif method == 'percentile':
        threshold = np.percentile(scores, percentile)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return threshold


def plot_score_distribution(normal_scores: np.ndarray,
                           anomaly_scores: np.ndarray,
                           threshold: float,
                           save_path: str):
    """
    Plot anomaly score distributions.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax = axes[0]
    bins = np.linspace(
        min(normal_scores.min(), anomaly_scores.min()),
        max(normal_scores.max(), anomaly_scores.max()),
        50
    )
    
    ax.hist(normal_scores, bins=bins, alpha=0.7, label='Normal', color='green', density=True)
    ax.hist(anomaly_scores, bins=bins, alpha=0.7, label='Anomaly', color='red', density=True)
    ax.axvline(threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.4f})')
    
    ax.set_xlabel('Anomaly Score (MSE)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Anomaly Score Distribution', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Box plot
    ax = axes[1]
    data = [normal_scores, anomaly_scores]
    bp = ax.boxplot(data, labels=['Normal', 'Anomaly'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightgreen')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax.axhline(threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold')
    
    ax.set_ylabel('Anomaly Score (MSE)', fontsize=12)
    ax.set_title('Score Comparison', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Score distribution plot saved to: {save_path}")


def plot_roc_curve(y_true: np.ndarray, 
                   y_scores: np.ndarray,
                   save_path: str) -> Dict[str, float]:
    """
    Plot ROC curve and compute AUC.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Find optimal threshold (Youden's J statistic)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.plot(fpr, tpr, color='blue', linewidth=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1)
    ax.scatter([fpr[optimal_idx]], [tpr[optimal_idx]], color='red', s=100, 
               zorder=5, label=f'Optimal (threshold={optimal_threshold:.4f})')
    
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"ROC curve saved to: {save_path}")
    
    return {
        'auc': roc_auc,
        'optimal_threshold': optimal_threshold,
        'optimal_tpr': tpr[optimal_idx],
        'optimal_fpr': fpr[optimal_idx]
    }


def plot_precision_recall(y_true: np.ndarray,
                         y_scores: np.ndarray,
                         save_path: str) -> Dict[str, float]:
    """
    Plot Precision-Recall curve.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.plot(recall, precision, color='blue', linewidth=2, label=f'PR curve (AUC = {pr_auc:.3f})')
    ax.axhline(y=y_true.sum() / len(y_true), color='gray', linestyle='--', 
               linewidth=1, label='Random classifier')
    
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Precision-Recall curve saved to: {save_path}")
    
    return {'pr_auc': pr_auc}


def plot_reconstruction_examples(model: VAE,
                                normal_data: np.ndarray,
                                anomaly_data: np.ndarray,
                                save_path: str,
                                n_examples: int = 4):
    """
    Plot reconstruction examples for normal and anomaly samples.
    """
    fig, axes = plt.subplots(4, n_examples, figsize=(n_examples * 3, 12))
    
    # Select random samples
    normal_idx = np.random.choice(len(normal_data), n_examples, replace=False)
    anomaly_idx = np.random.choice(len(anomaly_data), n_examples, replace=False)
    
    normal_samples = normal_data[normal_idx]
    anomaly_samples = anomaly_data[anomaly_idx]
    
    # Get reconstructions
    normal_recon = model.predict(normal_samples, verbose=0)
    anomaly_recon = model.predict(anomaly_samples, verbose=0)
    
    # Plot normal samples
    for i in range(n_examples):
        # Original
        axes[0, i].imshow(normal_samples[i, :, :, 0], aspect='auto', origin='lower', cmap='viridis')
        axes[0, i].set_title(f'Normal {i+1}' if i == 0 else f'{i+1}')
        axes[0, i].axis('off')
        
        # Reconstruction
        axes[1, i].imshow(normal_recon[i, :, :, 0], aspect='auto', origin='lower', cmap='viridis')
        axes[1, i].axis('off')
    
    # Plot anomaly samples
    for i in range(n_examples):
        # Original
        axes[2, i].imshow(anomaly_samples[i, :, :, 0], aspect='auto', origin='lower', cmap='viridis')
        axes[2, i].set_title(f'Anomaly {i+1}' if i == 0 else f'{i+1}')
        axes[2, i].axis('off')
        
        # Reconstruction
        axes[3, i].imshow(anomaly_recon[i, :, :, 0], aspect='auto', origin='lower', cmap='viridis')
        axes[3, i].axis('off')
    
    # Row labels
    axes[0, 0].set_ylabel('Normal\nOriginal', fontsize=10)
    axes[1, 0].set_ylabel('Normal\nRecon', fontsize=10)
    axes[2, 0].set_ylabel('Anomaly\nOriginal', fontsize=10)
    axes[3, 0].set_ylabel('Anomaly\nRecon', fontsize=10)
    
    plt.suptitle('Reconstruction Examples', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Reconstruction examples saved to: {save_path}")


def evaluate_model(vae=None, norm_params=None, features_path=None, segments_path=None, output_dir=None):
    """
    Main evaluation function.
    
    Args:
        vae: Trained VAE model (optional, loads from disk if None)
        norm_params: Normalization parameters dict (optional)
        features_path: Path to features file (not used by AuraDataset currently but good for future)
        segments_path: Path to segments file (not used by AuraDataset currently)
        output_dir: Directory to save plots/metrics (optional, uses default if None)
    """
    if output_dir:
        plots_dir = os.path.join(output_dir, "plots")
        metrics_dir = os.path.join(output_dir, "metrics")
        models_dir = os.path.join(output_dir, "models")
    else:
        plots_dir = PLOTS_DIR
        metrics_dir = METRICS_DIR
        models_dir = MODELS_DIR
        
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("AURA-VAE Model Evaluation")
    print("="*60)
    
    # =========================================================================
    # 1. Load model and data
    # =========================================================================
    print("\n[1/5] Loading model and data...")
    
    if vae is None:
        vae, normalizer = load_trained_model()
    else:
        # Reconstruct normalizer
        normalizer = FeatureNormalizer()
        if norm_params:
            normalizer.mean = norm_params['mean']
            normalizer.std = norm_params['std']
            normalizer.fitted = True
        else:
             # Try loading default params
             _, normalizer_loaded = load_trained_model() # Inefficient but safe fallback
             normalizer = normalizer_loaded
    
    # Load data
    dataset = AuraDataset(normalizer=normalizer)
    normal_features, anomaly_features = dataset.load_features()
    train_data, test_data = dataset.prepare_data(normal_features, anomaly_features)
    
    anomaly_data = dataset.get_anomaly_data()
    
    # Generate more synthetic anomalies if needed
    if anomaly_data is None or len(anomaly_data) < 100:
        print("  Generating additional synthetic anomalies...")
        extra_anomalies = generate_synthetic_anomalies(n_samples=200)
        if len(extra_anomalies.shape) == 3:
            extra_anomalies = np.expand_dims(extra_anomalies, axis=-1)
        extra_anomalies = normalizer.transform(extra_anomalies)
        
        if anomaly_data is not None:
            anomaly_data = np.concatenate([anomaly_data, extra_anomalies], axis=0)
        else:
            anomaly_data = extra_anomalies
    
    print(f"  Test data (normal): {test_data.shape}")
    print(f"  Anomaly data: {anomaly_data.shape}")
    
    # =========================================================================
    # 2. Compute anomaly scores
    # =========================================================================
    print("\n[2/5] Computing anomaly scores...")
    
    normal_scores = vae.compute_anomaly_score(test_data)
    anomaly_scores = vae.compute_anomaly_score(anomaly_data)
    
    print(f"  Normal scores - mean: {normal_scores.mean():.6f}, std: {normal_scores.std():.6f}")
    print(f"  Anomaly scores - mean: {anomaly_scores.mean():.6f}, std: {anomaly_scores.std():.6f}")
    
    # =========================================================================
    # 3. Compute threshold
    # =========================================================================
    print("\n[3/5] Computing anomaly threshold...")
    
    # Use training data for threshold computation
    train_scores = vae.compute_anomaly_score(train_data)
    
    threshold_std = compute_threshold(train_scores, method='std', k=ANOMALY_THRESHOLD_K)
    threshold_pct = compute_threshold(train_scores, method='percentile', percentile=ANOMALY_THRESHOLD_PERCENTILE)
    
    print(f"  Threshold (mean + {ANOMALY_THRESHOLD_K}σ): {threshold_std:.6f}")
    print(f"  Threshold ({ANOMALY_THRESHOLD_PERCENTILE}th percentile): {threshold_pct:.6f}")
    
    # Use percentile-based threshold for better robustness on mixed datasets
    # (Prevents silence from dragging the threshold down too far)
    threshold = threshold_pct
    print(f"  > Selected Threshold (Percentile): {threshold:.6f}")
    
    # =========================================================================
    # 4. Evaluate detection performance
    # =========================================================================
    print("\n[4/5] Evaluating detection performance...")
    
    # Classification results
    normal_detected = (normal_scores <= threshold).sum()
    anomaly_detected = (anomaly_scores > threshold).sum()
    
    normal_accuracy = normal_detected / len(normal_scores) * 100
    anomaly_accuracy = anomaly_detected / len(anomaly_scores) * 100
    
    print(f"  Normal correctly classified: {normal_detected}/{len(normal_scores)} ({normal_accuracy:.1f}%)")
    print(f"  Anomaly correctly detected: {anomaly_detected}/{len(anomaly_scores)} ({anomaly_accuracy:.1f}%)")
    
    # Prepare data for ROC/PR curves
    y_true = np.concatenate([np.zeros(len(normal_scores)), np.ones(len(anomaly_scores))])
    y_scores = np.concatenate([normal_scores, anomaly_scores])
    
    # =========================================================================
    # 5. Generate visualizations and save metrics
    # =========================================================================
    print("\n[5/5] Generating visualizations...")
    
    # Score distribution plot
    plot_score_distribution(
        normal_scores, anomaly_scores, threshold,
        os.path.join(plots_dir, "score_distribution.png")
    )
    
    # ROC curve
    roc_metrics = plot_roc_curve(
        y_true, y_scores,
        os.path.join(plots_dir, "roc_curve.png")
    )
    
    # Precision-Recall curve
    pr_metrics = plot_precision_recall(
        y_true, y_scores,
        os.path.join(plots_dir, "precision_recall.png")
    )
    
    # Reconstruction examples
    plot_reconstruction_examples(
        vae, test_data, anomaly_data,
        os.path.join(plots_dir, "reconstruction_examples.png")
    )
    
    # Save metrics
    metrics = {
        'threshold': float(threshold),
        'threshold_method': f'mean + {ANOMALY_THRESHOLD_K}*std',
        'normal_scores': {
            'mean': float(normal_scores.mean()),
            'std': float(normal_scores.std()),
            'min': float(normal_scores.min()),
            'max': float(normal_scores.max())
        },
        'anomaly_scores': {
            'mean': float(anomaly_scores.mean()),
            'std': float(anomaly_scores.std()),
            'min': float(anomaly_scores.min()),
            'max': float(anomaly_scores.max())
        },
        'detection': {
            'normal_accuracy': float(normal_accuracy),
            'anomaly_detection_rate': float(anomaly_accuracy),
            'false_positive_rate': float(100 - normal_accuracy),
            'false_negative_rate': float(100 - anomaly_accuracy)
        },
        'roc': roc_metrics,
        'precision_recall': pr_metrics
    }
    
    metrics_path = os.path.join(metrics_dir, "evaluation_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to: {metrics_path}")
    
    # Save threshold for Android
    threshold_config = {
        'threshold': float(threshold),
        'mean': float(train_scores.mean()),
        'std': float(train_scores.std()),
        'k': ANOMALY_THRESHOLD_K
    }
    threshold_path = os.path.join(models_dir, "threshold_config.json")
    with open(threshold_path, 'w') as f:
        json.dump(threshold_config, f, indent=2)
    print(f"Threshold config saved to: {threshold_path}")

    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*60)
    print("Evaluation Complete!")
    print("="*60)
    
    print(f"\nKey Results:")
    print(f"  ROC AUC: {roc_metrics['auc']:.3f}")
    print(f"  PR AUC: {pr_metrics['pr_auc']:.3f}")
    print(f"  Detection threshold: {threshold:.6f}")
    print(f"  Normal accuracy: {normal_accuracy:.1f}%")
    print(f"  Anomaly detection rate: {anomaly_accuracy:.1f}%")
    
    print(f"\nScore separation:")
    print(f"  Normal mean: {normal_scores.mean():.6f}")
    print(f"  Anomaly mean: {anomaly_scores.mean():.6f}")
    print(f"  Separation ratio: {anomaly_scores.mean() / normal_scores.mean():.2f}x")
    
    return metrics


if __name__ == "__main__":
    metrics = evaluate_model()

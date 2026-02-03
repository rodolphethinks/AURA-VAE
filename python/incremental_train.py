"""
Incremental training script to see model evolution.
Trains models with progressively more data:
- v1: Sound Acquisition 1 only
- v2: Sound Acquisitions 1 + 2
- v3: Sound Acquisitions 1 + 2 + 3
- v4: Sound Acquisitions 1 + 2 + 3 + 4

Records complete metrics for evolution graphs including:
- Full training curves (loss per epoch)
- Evaluation metrics (ROC AUC, accuracy)
- Model weights for each version
"""

import os
import sys
import json
import shutil
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Add python directory for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    RAW_DATA_DIR, MODELS_DIR, PROCESSED_DATA_DIR, BASE_DIR,
    SAMPLE_RATE, SEGMENT_SAMPLES, HOP_SAMPLES
)
from model_tracker import ModelTracker
from preprocessing import load_audio, normalize_audio, segment_audio, compute_mel_spectrogram

# Audio registry path (at project root)
AUDIO_REGISTRY_PATH = Path(BASE_DIR) / "audio_registry.json"

# Directory to save evolution data
EVOLUTION_DIR = Path(BASE_DIR) / "experiments" / "model_evolution"

# Define the training configurations
TRAINING_CONFIGS = [
    {
        "version": "v1_acquisition1_only",
        "description": "Training with Sound Acquisition 1 only",
        "files": ["Filante Sound Acquisition.m4a"]
    },
    {
        "version": "v2_acquisitions_1_2",
        "description": "Training with Sound Acquisitions 1 + 2",
        "files": ["Filante Sound Acquisition.m4a", "Filante Sound Acquisition 2_cleaned.wav"]
    },
    {
        "version": "v3_acquisitions_1_2_3",
        "description": "Training with Sound Acquisitions 1 + 2 + 3",
        "files": ["Filante Sound Acquisition.m4a", "Filante Sound Acquisition 2_cleaned.wav", "Filante Sound Acquisition 3_cleaned.wav"]
    },
    {
        "version": "v4_all_acquisitions",
        "description": "Training with all Sound Acquisitions 1 + 2 + 3 + 4",
        "files": ["Filante Sound Acquisition.m4a", "Filante Sound Acquisition 2_cleaned.wav", "Filante Sound Acquisition 3_cleaned.wav", "Filante Sound Acquisition 4_cleaned.wav"]
    }
]


def setup_evolution_dir():
    """Create evolution directory with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = EVOLUTION_DIR / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (run_dir / "models").mkdir(exist_ok=True)
    (run_dir / "histories").mkdir(exist_ok=True)
    (run_dir / "metrics").mkdir(exist_ok=True)
    (run_dir / "plots").mkdir(exist_ok=True)
    
    return run_dir


def backup_audio_registry():
    """Backup the current audio registry."""
    backup_path = AUDIO_REGISTRY_PATH.with_suffix('.json.backup')
    if AUDIO_REGISTRY_PATH.exists():
        shutil.copy(AUDIO_REGISTRY_PATH, backup_path)
        print(f"Backed up audio registry to {backup_path}")
    return backup_path


def restore_audio_registry(backup_path: Path):
    """Restore the audio registry from backup."""
    if backup_path.exists():
        shutil.copy(backup_path, AUDIO_REGISTRY_PATH)
        print(f"Restored audio registry from {backup_path}")


def preprocess_audio_files(files: list):
    """
    Preprocess audio files and save features to PROCESSED_DATA_DIR.
    
    Args:
        files: List of audio filenames to process
        
    Returns:
        Tuple of (features, segments, total_duration)
    """
    print("\n" + "="*60)
    print("PREPROCESSING")
    print("="*60)
    
    all_features = []
    all_segments = []
    total_duration = 0
    
    for filename in files:
        filepath = Path(RAW_DATA_DIR) / filename
        if not filepath.exists():
            print(f"  WARNING: File not found: {filepath}")
            continue
            
        print(f"\n  Processing: {filename}")
        
        # Load audio (load_audio returns just the array, sr is SAMPLE_RATE)
        audio = load_audio(str(filepath), target_sr=SAMPLE_RATE)
        sr = SAMPLE_RATE
        duration = len(audio) / sr
        total_duration += duration
        print(f"    Duration: {duration:.1f}s, Sample rate: {sr}")
        
        # Normalize
        audio = normalize_audio(audio)
        
        # Segment
        segments = segment_audio(audio, SEGMENT_SAMPLES, HOP_SAMPLES)
        print(f"    Segments: {len(segments)}")
        
        # Compute mel spectrograms
        for seg in segments:
            mel = compute_mel_spectrogram(seg, sr)
            all_features.append(mel)
            all_segments.append(seg)
    
    # Convert to arrays
    features = np.array(all_features)
    all_segments = np.array(all_segments)
    
    print(f"\n  Total: {len(features)} segments, {total_duration:.1f}s")
    print(f"  Features shape: {features.shape}")
    
    # Save to processed data directory
    features_path = Path(PROCESSED_DATA_DIR) / "normal_features.npy"
    segments_path = Path(PROCESSED_DATA_DIR) / "normal_segments.npy"
    
    np.save(features_path, features)
    np.save(segments_path, all_segments)
    
    print(f"\n  Saved features to: {features_path}")
    
    return features, all_segments, total_duration


def train_model():
    """
    Train the VAE model using data in PROCESSED_DATA_DIR.
    
    Returns:
        Tuple of (vae, history_dict, normalizer)
    """
    from train import train_vae
    
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    
    # Train the model
    vae, history, normalizer = train_vae()
    
    return vae, history, normalizer


def run_evaluation(vae, normalizer):
    """
    Run evaluation on the trained model.
    
    Returns:
        Dict of evaluation metrics
    """
    from evaluate import evaluate_model
    
    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)
    
    # Run evaluation (it saves to default directories)
    evaluate_model(vae=vae)
    
    # Load the metrics that were saved
    metrics_path = Path(MODELS_DIR).parent / "results" / "metrics" / "evaluation_metrics.json"
    if not metrics_path.exists():
        metrics_path = Path(BASE_DIR) / "results" / "metrics" / "evaluation_metrics.json"
    
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        return metrics
    
    return {}


def get_training_metrics():
    """Get metrics from the latest training."""
    history_path = Path(MODELS_DIR) / "training_history.json"
    if history_path.exists():
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        # Keys can be 'loss'/'val_loss' or 'total_loss'/'val_total_loss'
        train_losses = history.get("total_loss", history.get("loss", []))
        val_losses = history.get("val_total_loss", history.get("val_loss", []))
        
        return {
            "final_train_loss": train_losses[-1] if train_losses else None,
            "final_val_loss": val_losses[-1] if val_losses else None,
            "best_val_loss": min(val_losses) if val_losses else None,
            "epochs_trained": len(train_losses),
            "full_history": history  # Include full history for graphs
        }
    return {}


def save_version_artifacts(version_name: str, run_dir: Path, training_metrics: dict, eval_metrics: dict):
    """
    Save all artifacts for a version.
    
    Args:
        version_name: Name of the version
        run_dir: Directory for this evolution run
        training_metrics: Training metrics including full history
        eval_metrics: Evaluation metrics
    """
    # Save model weights
    weights_src = Path(MODELS_DIR) / "vae_model.weights.h5"
    if weights_src.exists():
        weights_dst = run_dir / "models" / f"{version_name}_weights.h5"
        shutil.copy(weights_src, weights_dst)
        print(f"  Saved weights: {weights_dst.name}")
    
    # Save full training history
    if "full_history" in training_metrics:
        history_path = run_dir / "histories" / f"{version_name}_history.json"
        with open(history_path, 'w') as f:
            json.dump(training_metrics["full_history"], f, indent=2)
        print(f"  Saved history: {history_path.name}")
    
    # Save evaluation metrics
    if eval_metrics:
        metrics_path = run_dir / "metrics" / f"{version_name}_eval.json"
        with open(metrics_path, 'w') as f:
            json.dump(eval_metrics, f, indent=2)
        print(f"  Saved eval metrics: {metrics_path.name}")


def plot_evolution_graphs(run_dir: Path, evolution_data: list):
    """
    Generate graphs showing model evolution across versions.
    
    Args:
        run_dir: Directory for this evolution run
        evolution_data: List of dicts with version data
    """
    plots_dir = run_dir / "plots"
    
    # Extract data for plotting
    versions = [d['version'] for d in evolution_data]
    short_versions = [f"v{i+1}" for i in range(len(versions))]
    
    # 1. Training data growth
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    segments = [d['segments'] for d in evolution_data]
    durations = [d['duration_min'] for d in evolution_data]
    num_files = [d['num_files'] for d in evolution_data]
    
    axes[0].bar(short_versions, segments, color='steelblue', edgecolor='black')
    axes[0].set_ylabel('Segments')
    axes[0].set_title('Training Data Segments')
    for i, v in enumerate(segments):
        axes[0].text(i, v + 100, str(v), ha='center', fontsize=9)
    
    axes[1].bar(short_versions, durations, color='forestgreen', edgecolor='black')
    axes[1].set_ylabel('Minutes')
    axes[1].set_title('Training Data Duration')
    for i, v in enumerate(durations):
        axes[1].text(i, v + 1, f"{v:.1f}", ha='center', fontsize=9)
    
    axes[2].bar(short_versions, num_files, color='coral', edgecolor='black')
    axes[2].set_ylabel('Files')
    axes[2].set_title('Number of Audio Files')
    for i, v in enumerate(num_files):
        axes[2].text(i, v + 0.1, str(v), ha='center', fontsize=9)
    
    plt.suptitle('Training Data Growth Across Versions', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(plots_dir / "data_growth.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: data_growth.png")
    
    # 2. Training metrics evolution
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    best_val_loss = [d.get('best_val_loss') for d in evolution_data]
    final_val_loss = [d.get('final_val_loss') for d in evolution_data]
    epochs = [d.get('epochs_trained', 0) for d in evolution_data]
    
    # Filter None values
    valid_idx = [i for i, v in enumerate(best_val_loss) if v is not None]
    
    if valid_idx:
        axes[0].plot([short_versions[i] for i in valid_idx], 
                     [best_val_loss[i] for i in valid_idx], 
                     'o-', color='crimson', linewidth=2, markersize=8)
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Best Validation Loss')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot([short_versions[i] for i in valid_idx], 
                     [final_val_loss[i] for i in valid_idx], 
                     's-', color='darkorange', linewidth=2, markersize=8)
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Final Validation Loss')
        axes[1].grid(True, alpha=0.3)
    
    axes[2].bar(short_versions, epochs, color='purple', edgecolor='black', alpha=0.7)
    axes[2].set_ylabel('Epochs')
    axes[2].set_title('Training Epochs')
    for i, v in enumerate(epochs):
        if v:
            axes[2].text(i, v + 1, str(v), ha='center', fontsize=9)
    
    plt.suptitle('Training Metrics Evolution', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(plots_dir / "training_metrics.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: training_metrics.png")
    
    # 3. Evaluation metrics evolution
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    roc_auc = [d.get('roc_auc') for d in evolution_data]
    normal_acc = [d.get('normal_accuracy') for d in evolution_data]
    anomaly_rate = [d.get('anomaly_detection_rate') for d in evolution_data]
    
    valid_idx_eval = [i for i, v in enumerate(roc_auc) if v is not None]
    
    if valid_idx_eval:
        axes[0].plot([short_versions[i] for i in valid_idx_eval], 
                     [roc_auc[i] for i in valid_idx_eval], 
                     'o-', color='green', linewidth=2, markersize=8)
        axes[0].set_ylabel('ROC AUC')
        axes[0].set_title('ROC AUC Score')
        axes[0].set_ylim(0, 1.05)
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot([short_versions[i] for i in valid_idx_eval], 
                     [normal_acc[i] for i in valid_idx_eval], 
                     's-', color='blue', linewidth=2, markersize=8)
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Normal Classification Accuracy')
        axes[1].set_ylim(0, 105)
        axes[1].grid(True, alpha=0.3)
        
        axes[2].plot([short_versions[i] for i in valid_idx_eval], 
                     [anomaly_rate[i] for i in valid_idx_eval], 
                     '^-', color='red', linewidth=2, markersize=8)
        axes[2].set_ylabel('Detection Rate (%)')
        axes[2].set_title('Anomaly Detection Rate')
        axes[2].set_ylim(0, 105)
        axes[2].grid(True, alpha=0.3)
    else:
        for ax in axes:
            ax.text(0.5, 0.5, 'No evaluation data', ha='center', va='center', transform=ax.transAxes)
    
    plt.suptitle('Evaluation Metrics Evolution', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(plots_dir / "evaluation_metrics.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: evaluation_metrics.png")
    
    # 4. Training curves comparison (all versions on same plot)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, data in enumerate(evolution_data):
        if 'full_history' in data and data['full_history']:
            history = data['full_history']
            train_loss = history.get("total_loss", history.get("loss", []))
            val_loss = history.get("val_total_loss", history.get("val_loss", []))
            
            if train_loss:
                axes[0].plot(train_loss, label=f"v{i+1}", color=colors[i], alpha=0.8)
            if val_loss:
                axes[1].plot(val_loss, label=f"v{i+1}", color=colors[i], alpha=0.8)
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Validation Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('Training Curves Comparison', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(plots_dir / "training_curves_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: training_curves_comparison.png")
    
    # 5. Summary table as image
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')
    
    table_data = []
    headers = ['Version', 'Files', 'Segments', 'Duration (min)', 'Best Val Loss', 'ROC AUC', 'Normal Acc', 'Anomaly Det']
    
    for i, d in enumerate(evolution_data):
        row = [
            f"v{i+1}",
            d.get('num_files', 'N/A'),
            d.get('segments', 'N/A'),
            f"{d.get('duration_min', 0):.1f}",
            f"{d.get('best_val_loss', 0):.2f}" if d.get('best_val_loss') else 'N/A',
            f"{d.get('roc_auc', 0):.3f}" if d.get('roc_auc') else 'N/A',
            f"{d.get('normal_accuracy', 0):.1f}%" if d.get('normal_accuracy') else 'N/A',
            f"{d.get('anomaly_detection_rate', 0):.1f}%" if d.get('anomaly_detection_rate') else 'N/A',
        ]
        table_data.append(row)
    
    table = ax.table(cellText=table_data, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Color header
    for j in range(len(headers)):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    
    # Alternate row colors
    for i in range(len(table_data)):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i+1, j)].set_facecolor('#D9E2F3')
    
    plt.title('Model Evolution Summary', fontsize=14, pad=20)
    plt.savefig(plots_dir / "evolution_summary.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: evolution_summary.png")


def run_incremental_training(start_from: int = 1, skip_eval: bool = False):
    """
    Run incremental training for all configurations.
    
    Args:
        start_from: Start from this version (1-4)
        skip_eval: If True, skip evaluation step
    """
    # Setup directories
    run_dir = setup_evolution_dir()
    print(f"\nEvolution run directory: {run_dir}")
    
    tracker = ModelTracker(str(BASE_DIR))
    backup_path = backup_audio_registry()
    
    # Clear existing versions in tracker for fresh run
    tracker.history["versions"] = []
    tracker._save_history()
    
    evolution_data = []
    
    try:
        for i, config in enumerate(TRAINING_CONFIGS, 1):
            if i < start_from:
                print(f"\nSkipping {config['version']} (starting from v{start_from})")
                continue
                
            print("\n" + "="*70)
            print(f"[{i}/4] TRAINING {config['version'].upper()}")
            print(f"      {config['description']}")
            print(f"      Files: {config['files']}")
            print("="*70)
            
            # Preprocess
            features, segments, total_duration = preprocess_audio_files(config['files'])
            
            # Train
            try:
                vae, history, normalizer = train_model()
                success = True
            except Exception as e:
                print(f"Training failed: {e}")
                import traceback
                traceback.print_exc()
                success = False
                vae, normalizer = None, None
            
            eval_metrics = {}
            if success:
                # Get training metrics
                training_metrics = get_training_metrics()
                
                # Run evaluation
                if not skip_eval:
                    try:
                        eval_metrics = run_evaluation(vae, normalizer)
                    except Exception as e:
                        print(f"Evaluation failed: {e}")
                        import traceback
                        traceback.print_exc()
                
                # Save version artifacts
                print("\n--- Saving Artifacts ---")
                save_version_artifacts(config['version'], run_dir, training_metrics, eval_metrics)
                
                # Prepare tracker metrics
                evaluation_metrics = None
                threshold_config = None
                
                if eval_metrics:
                    evaluation_metrics = {
                        "roc_auc": eval_metrics.get("roc", {}).get("auc"),
                        "pr_auc": eval_metrics.get("precision_recall", {}).get("auc"),
                        "normal_accuracy": eval_metrics.get("detection", {}).get("normal_accuracy"),
                        "anomaly_detection_rate": eval_metrics.get("detection", {}).get("anomaly_detection_rate"),
                        "normal_mean_score": eval_metrics.get("normal_scores", {}).get("mean"),
                        "anomaly_mean_score": eval_metrics.get("anomaly_scores", {}).get("mean"),
                    }
                    threshold_config = {
                        "threshold": eval_metrics.get("threshold"),
                        "mean": eval_metrics.get("normal_scores", {}).get("mean"),
                        "std": eval_metrics.get("normal_scores", {}).get("std"),
                    }
                
                # Record in tracker
                tracker.record_version(
                    version_name=config['version'],
                    audio_files=config['files'],
                    total_segments=len(features),
                    total_duration_minutes=total_duration / 60,
                    training_metrics={
                        "final_train_loss": training_metrics.get("final_train_loss"),
                        "final_val_loss": training_metrics.get("final_val_loss"),
                        "best_val_loss": training_metrics.get("best_val_loss"),
                        "epochs_trained": training_metrics.get("epochs_trained"),
                    },
                    evaluation_metrics=evaluation_metrics,
                    threshold_config=threshold_config,
                    experiment_dir=str(run_dir),
                    notes=config['description']
                )
                
                # Collect data for evolution graphs
                version_data = {
                    'version': config['version'],
                    'num_files': len(config['files']),
                    'segments': len(features),
                    'duration_min': total_duration / 60,
                    'best_val_loss': training_metrics.get("best_val_loss"),
                    'final_val_loss': training_metrics.get("final_val_loss"),
                    'epochs_trained': training_metrics.get("epochs_trained"),
                    'full_history': training_metrics.get("full_history"),
                }
                
                if eval_metrics:
                    version_data.update({
                        'roc_auc': eval_metrics.get("roc", {}).get("auc"),
                        'normal_accuracy': eval_metrics.get("detection", {}).get("normal_accuracy"),
                        'anomaly_detection_rate': eval_metrics.get("detection", {}).get("anomaly_detection_rate"),
                    })
                
                evolution_data.append(version_data)
                
                print(f"\n[OK] Completed {config['version']}")
            else:
                print(f"\n[FAIL] Failed {config['version']}")
            
            # Small delay between trainings
            time.sleep(2)
    
    finally:
        # Restore original audio registry
        restore_audio_registry(backup_path)
        # Clean up backup
        if backup_path.exists():
            backup_path.unlink()
    
    # Generate evolution graphs
    if evolution_data:
        print("\n" + "="*60)
        print("GENERATING EVOLUTION GRAPHS")
        print("="*60)
        plot_evolution_graphs(run_dir, evolution_data)
        
        # Save evolution data
        evolution_path = run_dir / "evolution_data.json"
        # Remove full_history for JSON (too large)
        evolution_json = []
        for d in evolution_data:
            d_copy = {k: v for k, v in d.items() if k != 'full_history'}
            evolution_json.append(d_copy)
        with open(evolution_path, 'w') as f:
            json.dump(evolution_json, f, indent=2)
        print(f"  Saved: evolution_data.json")
    
    # Print final evolution summary
    print("\n" + "="*60)
    print("MODEL EVOLUTION SUMMARY")
    print("="*60)
    tracker.print_history()
    
    print(f"\n[DONE] All artifacts saved to: {run_dir}")
    
    return run_dir


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Incremental training for model evolution")
    parser.add_argument("--start-from", type=int, default=1, 
                        help="Start from version (1-4)")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip evaluation step (faster but no ROC AUC metrics)")
    
    args = parser.parse_args()
    
    run_incremental_training(start_from=args.start_from, skip_eval=args.skip_eval)

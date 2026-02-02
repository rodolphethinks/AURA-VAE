"""
AURA-VAE Model Evolution Tracker

Tracks model performance metrics over time as training data grows.
Enables comparison of model versions and analysis of improvement trends.
"""

import os
import json
import hashlib
from datetime import datetime
from typing import Optional, Dict, List, Any
import numpy as np


class ModelTracker:
    """
    Tracks model evolution across training iterations.
    
    Records:
    - Training data info (files, sizes, duration)
    - Model metrics (loss, accuracy, ROC AUC, etc.)
    - Threshold configuration
    - Git commit (if available)
    """
    
    TRACKER_FILE = "model_evolution.json"
    
    def __init__(self, base_dir: str):
        """
        Initialize tracker.
        
        Args:
            base_dir: Base directory for the project (where models/ is located)
        """
        self.base_dir = base_dir
        self.tracker_path = os.path.join(base_dir, "models", self.TRACKER_FILE)
        self.history = self._load_history()
    
    def _load_history(self) -> Dict:
        """Load existing history or create new."""
        if os.path.exists(self.tracker_path):
            with open(self.tracker_path, 'r') as f:
                return json.load(f)
        return {
            "description": "AURA-VAE Model Evolution Tracker - Records model performance over training iterations",
            "created": datetime.now().isoformat(),
            "versions": []
        }
    
    def _save_history(self):
        """Save history to disk."""
        os.makedirs(os.path.dirname(self.tracker_path), exist_ok=True)
        with open(self.tracker_path, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def _get_git_info(self) -> Dict[str, str]:
        """Get current git commit info."""
        try:
            import subprocess
            commit = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'],
                cwd=self.base_dir,
                stderr=subprocess.DEVNULL
            ).decode().strip()[:8]
            
            branch = subprocess.check_output(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                cwd=self.base_dir,
                stderr=subprocess.DEVNULL
            ).decode().strip()
            
            return {"commit": commit, "branch": branch}
        except:
            return {"commit": "unknown", "branch": "unknown"}
    
    def _compute_data_hash(self, audio_files: List[str]) -> str:
        """Compute hash of training data for tracking changes."""
        content = "|".join(sorted(audio_files))
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def record_version(
        self,
        version_name: Optional[str] = None,
        audio_files: List[str] = None,
        total_segments: int = 0,
        total_duration_minutes: float = 0,
        training_metrics: Dict[str, Any] = None,
        evaluation_metrics: Dict[str, Any] = None,
        threshold_config: Dict[str, float] = None,
        experiment_dir: str = None,
        notes: str = ""
    ) -> str:
        """
        Record a new model version.
        
        Args:
            version_name: Optional custom version name
            audio_files: List of audio files used for training
            total_segments: Total number of training segments
            total_duration_minutes: Total audio duration in minutes
            training_metrics: Training history (losses, etc.)
            evaluation_metrics: Evaluation results (ROC AUC, accuracy, etc.)
            threshold_config: Anomaly threshold configuration
            experiment_dir: Path to experiment directory
            notes: Additional notes about this version
        
        Returns:
            Version ID
        """
        # Generate version ID
        version_num = len(self.history["versions"]) + 1
        timestamp = datetime.now()
        
        if version_name is None:
            version_name = f"v{version_num}"
        
        version_id = f"{version_name}_{timestamp.strftime('%Y%m%d')}"
        
        # Get git info
        git_info = self._get_git_info()
        
        # Compute data hash
        data_hash = self._compute_data_hash(audio_files or [])
        
        # Build version record
        version = {
            "id": version_id,
            "version_number": version_num,
            "timestamp": timestamp.isoformat(),
            "git": git_info,
            
            # Training data info
            "training_data": {
                "audio_files": audio_files or [],
                "num_files": len(audio_files) if audio_files else 0,
                "total_segments": total_segments,
                "total_duration_minutes": round(total_duration_minutes, 1),
                "data_hash": data_hash
            },
            
            # Model metrics
            "training_metrics": {
                "final_train_loss": training_metrics.get("final_train_loss") if training_metrics else None,
                "final_val_loss": training_metrics.get("final_val_loss") if training_metrics else None,
                "best_val_loss": training_metrics.get("best_val_loss") if training_metrics else None,
                "epochs_trained": training_metrics.get("epochs_trained") if training_metrics else None,
            },
            
            "evaluation_metrics": {
                "roc_auc": evaluation_metrics.get("roc_auc") if evaluation_metrics else None,
                "pr_auc": evaluation_metrics.get("pr_auc") if evaluation_metrics else None,
                "normal_accuracy": evaluation_metrics.get("normal_accuracy") if evaluation_metrics else None,
                "anomaly_detection_rate": evaluation_metrics.get("anomaly_detection_rate") if evaluation_metrics else None,
                "normal_mean_score": evaluation_metrics.get("normal_mean_score") if evaluation_metrics else None,
                "anomaly_mean_score": evaluation_metrics.get("anomaly_mean_score") if evaluation_metrics else None,
            },
            
            "threshold": threshold_config or {},
            "experiment_dir": experiment_dir,
            "notes": notes
        }
        
        # Add to history
        self.history["versions"].append(version)
        self.history["last_updated"] = timestamp.isoformat()
        
        # Save
        self._save_history()
        
        print(f"\n[MODEL TRACKER] Recorded version: {version_id}")
        print(f"  Training data: {len(audio_files) if audio_files else 0} files, {total_duration_minutes:.1f} min")
        print(f"  Segments: {total_segments}")
        if evaluation_metrics:
            print(f"  ROC AUC: {evaluation_metrics.get('roc_auc', 'N/A')}")
        
        return version_id
    
    def get_latest_version(self) -> Optional[Dict]:
        """Get the most recent version."""
        if self.history["versions"]:
            return self.history["versions"][-1]
        return None
    
    def get_version(self, version_id: str) -> Optional[Dict]:
        """Get a specific version by ID."""
        for v in self.history["versions"]:
            if v["id"] == version_id:
                return v
        return None
    
    def compare_versions(self, v1_id: str = None, v2_id: str = None) -> Dict:
        """
        Compare two versions.
        
        Args:
            v1_id: First version ID (default: second to last)
            v2_id: Second version ID (default: latest)
        
        Returns:
            Comparison dictionary with deltas
        """
        versions = self.history["versions"]
        
        if len(versions) < 2:
            return {"error": "Need at least 2 versions to compare"}
        
        # Default to last two versions
        v1 = self.get_version(v1_id) if v1_id else versions[-2]
        v2 = self.get_version(v2_id) if v2_id else versions[-1]
        
        if not v1 or not v2:
            return {"error": "Version not found"}
        
        def safe_delta(a, b):
            if a is None or b is None:
                return None
            return round(b - a, 6)
        
        def safe_pct_change(a, b):
            if a is None or b is None or a == 0:
                return None
            return round((b - a) / abs(a) * 100, 2)
        
        return {
            "v1": v1["id"],
            "v2": v2["id"],
            "data_growth": {
                "files_added": v2["training_data"]["num_files"] - v1["training_data"]["num_files"],
                "segments_added": v2["training_data"]["total_segments"] - v1["training_data"]["total_segments"],
                "minutes_added": round(v2["training_data"]["total_duration_minutes"] - v1["training_data"]["total_duration_minutes"], 1)
            },
            "metric_changes": {
                "val_loss_delta": safe_delta(
                    v1["training_metrics"].get("best_val_loss"),
                    v2["training_metrics"].get("best_val_loss")
                ),
                "roc_auc_delta": safe_delta(
                    v1["evaluation_metrics"].get("roc_auc"),
                    v2["evaluation_metrics"].get("roc_auc")
                ),
                "normal_accuracy_delta": safe_delta(
                    v1["evaluation_metrics"].get("normal_accuracy"),
                    v2["evaluation_metrics"].get("normal_accuracy")
                ),
            },
            "threshold_change": safe_delta(
                v1["threshold"].get("threshold"),
                v2["threshold"].get("threshold")
            )
        }
    
    def get_evolution_summary(self) -> Dict:
        """Get summary of model evolution."""
        versions = self.history["versions"]
        
        if not versions:
            return {"error": "No versions recorded"}
        
        # Extract trends
        data_growth = []
        val_losses = []
        roc_aucs = []
        
        for v in versions:
            data_growth.append(v["training_data"]["total_duration_minutes"])
            if v["training_metrics"].get("best_val_loss"):
                val_losses.append(v["training_metrics"]["best_val_loss"])
            if v["evaluation_metrics"].get("roc_auc"):
                roc_aucs.append(v["evaluation_metrics"]["roc_auc"])
        
        return {
            "total_versions": len(versions),
            "first_version": versions[0]["id"],
            "latest_version": versions[-1]["id"],
            "data_evolution": {
                "initial_duration_minutes": data_growth[0] if data_growth else 0,
                "current_duration_minutes": data_growth[-1] if data_growth else 0,
                "total_growth_minutes": round(data_growth[-1] - data_growth[0], 1) if len(data_growth) >= 2 else 0
            },
            "performance_evolution": {
                "initial_val_loss": val_losses[0] if val_losses else None,
                "current_val_loss": val_losses[-1] if val_losses else None,
                "loss_improvement": round(val_losses[0] - val_losses[-1], 4) if len(val_losses) >= 2 else None,
                "current_roc_auc": roc_aucs[-1] if roc_aucs else None
            }
        }
    
    def print_history(self):
        """Print formatted history."""
        print("\n" + "=" * 70)
        print("AURA-VAE MODEL EVOLUTION HISTORY")
        print("=" * 70)
        
        summary = self.get_evolution_summary()
        if "error" in summary:
            print(f"  {summary['error']}")
            return
        
        print(f"\nTotal Versions: {summary['total_versions']}")
        print(f"Data Growth: {summary['data_evolution']['initial_duration_minutes']:.1f} → {summary['data_evolution']['current_duration_minutes']:.1f} minutes (+{summary['data_evolution']['total_growth_minutes']:.1f} min)")
        
        print("\n" + "-" * 70)
        print(f"{'Version':<20} {'Date':<12} {'Files':<6} {'Duration':<10} {'Val Loss':<12} {'ROC AUC':<8}")
        print("-" * 70)
        
        for v in self.history["versions"]:
            date = v["timestamp"][:10]
            files = v["training_data"]["num_files"]
            duration = f"{v['training_data']['total_duration_minutes']:.1f}m"
            val_loss = v["training_metrics"].get("best_val_loss")
            val_loss_str = f"{val_loss:.4f}" if val_loss else "N/A"
            roc_auc = v["evaluation_metrics"].get("roc_auc")
            roc_auc_str = f"{roc_auc:.4f}" if roc_auc else "N/A"
            
            print(f"{v['id']:<20} {date:<12} {files:<6} {duration:<10} {val_loss_str:<12} {roc_auc_str:<8}")
        
        print("-" * 70)


def initialize_baseline(base_dir: str = None):
    """
    Initialize the tracker with current model as baseline.
    
    Call this once to establish the starting point for tracking.
    """
    import sys
    if base_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    sys.path.insert(0, os.path.join(base_dir, 'python'))
    
    tracker = ModelTracker(base_dir)
    
    # Load current training info
    audio_registry_path = os.path.join(base_dir, "audio_registry.json")
    if os.path.exists(audio_registry_path):
        with open(audio_registry_path, 'r') as f:
            registry = json.load(f)
        audio_files = registry.get("training_audio_files", [])
    else:
        audio_files = []
    
    # Load evaluation metrics from latest experiment or models dir
    eval_metrics_path = os.path.join(base_dir, "models", "threshold_config.json")
    eval_metrics = {}
    threshold_config = {}
    if os.path.exists(eval_metrics_path):
        with open(eval_metrics_path, 'r') as f:
            threshold_config = json.load(f)
    
    # Try to load from latest experiment
    experiments_dir = os.path.join(base_dir, "experiments")
    if os.path.exists(experiments_dir):
        exp_dirs = sorted([d for d in os.listdir(experiments_dir) if d.endswith("_train")])
        if exp_dirs:
            latest_exp = exp_dirs[-1]
            exp_path = os.path.join(experiments_dir, latest_exp)
            
            # Load evaluation metrics
            metrics_file = os.path.join(exp_path, "metrics", "evaluation_metrics.json")
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    eval_data = json.load(f)
                    eval_metrics = {
                        "roc_auc": eval_data.get("roc", {}).get("auc"),
                        "pr_auc": eval_data.get("precision_recall", {}).get("pr_auc"),
                        "normal_accuracy": eval_data.get("detection", {}).get("normal_accuracy"),
                        "anomaly_detection_rate": eval_data.get("detection", {}).get("anomaly_detection_rate"),
                        "normal_mean_score": eval_data.get("normal_scores", {}).get("mean"),
                        "anomaly_mean_score": eval_data.get("anomaly_scores", {}).get("mean"),
                    }
            
            # Load training history
            history_file = os.path.join(exp_path, "models", "training_history.json")
            training_metrics = {}
            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    history = json.load(f)
                    val_losses = history.get("val_total_loss", [])
                    train_losses = history.get("total_loss", [])
                    training_metrics = {
                        "final_train_loss": train_losses[-1] if train_losses else None,
                        "final_val_loss": val_losses[-1] if val_losses else None,
                        "best_val_loss": min(val_losses) if val_losses else None,
                        "epochs_trained": len(val_losses)
                    }
            
            # Load preprocessing metadata
            meta_file = os.path.join(exp_path, "preprocessing_metadata.json")
            total_segments = 0
            total_duration = 0
            if os.path.exists(meta_file):
                with open(meta_file, 'r') as f:
                    meta = json.load(f)
                    total_segments = meta.get("num_segments", 0)
                    total_duration = meta.get("total_duration_seconds", 0) / 60
    
    # Record baseline
    version_id = tracker.record_version(
        version_name="v1_baseline",
        audio_files=audio_files,
        total_segments=total_segments,
        total_duration_minutes=total_duration,
        training_metrics=training_metrics,
        evaluation_metrics=eval_metrics,
        threshold_config=threshold_config,
        experiment_dir=os.path.join(experiments_dir, latest_exp) if exp_dirs else None,
        notes="Baseline model - standardized preprocessing with ref=np.max"
    )
    
    return tracker


if __name__ == "__main__":
    import sys
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if len(sys.argv) > 1 and sys.argv[1] == "--init":
        print("Initializing model tracker with current state as baseline...")
        tracker = initialize_baseline(base_dir)
        tracker.print_history()
    else:
        tracker = ModelTracker(base_dir)
        tracker.print_history()
        
        if len(tracker.history["versions"]) >= 2:
            print("\n" + "=" * 70)
            print("LATEST VERSION COMPARISON")
            print("=" * 70)
            comparison = tracker.compare_versions()
            print(json.dumps(comparison, indent=2))

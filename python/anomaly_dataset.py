"""
AURA-VAE Labeled Anomaly Dataset Manager

Manages a dataset of labeled anomalous audio segments for:
1. Real-world evaluation (instead of just synthetic anomalies)
2. Potential future supervised/semi-supervised training
3. Building a ground-truth anomaly library

Sources:
- Segments marked as anomalous during inference review (y/n/r workflow)
- Extracted differences between raw and cleaned audio files
"""

import os
import sys
import json
import hashlib
import numpy as np
import soundfile as sf
from datetime import datetime
from typing import Optional, List, Dict, Tuple
import librosa

# Add parent dir for imports when run as module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    BASE_DIR, DATA_DIR, SAMPLE_RATE, SEGMENT_SAMPLES, HOP_SAMPLES,
    N_FFT, HOP_LENGTH, N_MELS, F_MIN, F_MAX, N_TIME_FRAMES,
    SPEC_AMIN, SPEC_TOP_DB
)


class LabeledAnomalyDataset:
    """
    Manages labeled anomalous audio segments.
    
    Structure:
    data/anomalies/
    ├── anomaly_registry.json    # Metadata for all anomalies
    ├── audio/                   # Raw audio segments (.wav)
    │   ├── anomaly_001.wav
    │   └── ...
    └── features/                # Pre-extracted mel features (.npy)
        ├── anomaly_001.npy
        └── ...
    """
    
    REGISTRY_FILE = "anomaly_registry.json"
    
    def __init__(self, base_dir: str = None):
        """Initialize the anomaly dataset manager."""
        if base_dir is None:
            base_dir = BASE_DIR
        
        self.base_dir = base_dir
        self.anomaly_dir = os.path.join(base_dir, "data", "anomalies")
        self.audio_dir = os.path.join(self.anomaly_dir, "audio")
        self.features_dir = os.path.join(self.anomaly_dir, "features")
        self.registry_path = os.path.join(self.anomaly_dir, self.REGISTRY_FILE)
        
        # Ensure directories exist
        os.makedirs(self.audio_dir, exist_ok=True)
        os.makedirs(self.features_dir, exist_ok=True)
        
        # Load or create registry
        self.registry = self._load_registry()
    
    def _load_registry(self) -> Dict:
        """Load existing registry or create new one."""
        if os.path.exists(self.registry_path):
            with open(self.registry_path, 'r') as f:
                return json.load(f)
        return {
            "description": "Labeled anomaly audio segments for evaluation and training",
            "created": datetime.now().isoformat(),
            "anomalies": [],
            "categories": {
                "vehicle_mechanical": "Real vehicle mechanical anomalies (knocks, squeaks, rattles)",
                "vehicle_other": "Other vehicle sounds that may indicate issues",
                "non_vehicle": "Non-vehicle sounds removed during cleaning (talking, paper, etc.)",
                "unknown": "Anomalies of unknown origin"
            },
            "stats": {
                "total_count": 0,
                "total_duration_seconds": 0,
                "by_category": {}
            }
        }
    
    def _save_registry(self):
        """Save registry to disk."""
        # Update stats
        self.registry["stats"]["total_count"] = len(self.registry["anomalies"])
        self.registry["stats"]["total_duration_seconds"] = sum(
            a.get("duration_seconds", 0) for a in self.registry["anomalies"]
        )
        
        # Count by category
        by_cat = {}
        for a in self.registry["anomalies"]:
            cat = a.get("category", "unknown")
            by_cat[cat] = by_cat.get(cat, 0) + 1
        self.registry["stats"]["by_category"] = by_cat
        
        self.registry["last_updated"] = datetime.now().isoformat()
        
        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def _compute_audio_hash(self, audio: np.ndarray) -> str:
        """Compute hash of audio for deduplication."""
        return hashlib.md5(audio.tobytes()).hexdigest()[:16]
    
    def _extract_mel_features(self, audio: np.ndarray) -> np.ndarray:
        """Extract mel spectrogram features from audio segment."""
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            fmin=F_MIN,
            fmax=F_MAX,
            power=2.0
        )
        mel_spec_db = librosa.power_to_db(
            mel_spec,
            ref=np.max,
            amin=SPEC_AMIN,
            top_db=SPEC_TOP_DB
        )
        
        # Ensure correct shape
        if mel_spec_db.shape[1] > N_TIME_FRAMES:
            mel_spec_db = mel_spec_db[:, :N_TIME_FRAMES]
        elif mel_spec_db.shape[1] < N_TIME_FRAMES:
            pad = N_TIME_FRAMES - mel_spec_db.shape[1]
            mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad)), mode='constant')
        
        return mel_spec_db.astype(np.float32)
    
    def add_anomaly(
        self,
        audio: np.ndarray,
        source_file: str,
        start_time: float,
        end_time: float,
        anomaly_score: float = None,
        category: str = "unknown",
        notes: str = "",
        auto_extract_features: bool = True
    ) -> Optional[str]:
        """
        Add an anomalous audio segment to the dataset.
        
        Args:
            audio: Audio samples (1D array at SAMPLE_RATE)
            source_file: Original audio file name
            start_time: Start time in source file (seconds)
            end_time: End time in source file (seconds)
            anomaly_score: Model's anomaly score (if available)
            category: Anomaly category
            notes: Additional notes
            auto_extract_features: Whether to pre-extract mel features
        
        Returns:
            Anomaly ID if added, None if duplicate
        """
        # Check for duplicate
        audio_hash = self._compute_audio_hash(audio)
        for a in self.registry["anomalies"]:
            if a.get("audio_hash") == audio_hash:
                print(f"  [Skip] Duplicate anomaly (hash: {audio_hash[:8]})")
                return None
        
        # Generate ID
        anomaly_id = f"anomaly_{len(self.registry['anomalies']) + 1:04d}"
        
        # Save audio
        audio_path = os.path.join(self.audio_dir, f"{anomaly_id}.wav")
        sf.write(audio_path, audio, SAMPLE_RATE)
        
        # Extract and save features
        features_path = None
        if auto_extract_features:
            # For segments longer than 1 second, extract multiple feature windows
            if len(audio) > SEGMENT_SAMPLES:
                features_list = []
                for start in range(0, len(audio) - SEGMENT_SAMPLES + 1, HOP_SAMPLES):
                    segment = audio[start:start + SEGMENT_SAMPLES]
                    features_list.append(self._extract_mel_features(segment))
                features = np.array(features_list)
            else:
                # Pad if needed
                if len(audio) < SEGMENT_SAMPLES:
                    audio_padded = np.zeros(SEGMENT_SAMPLES, dtype=audio.dtype)
                    audio_padded[:len(audio)] = audio
                    audio = audio_padded
                features = self._extract_mel_features(audio)[np.newaxis, ...]
            
            features_path = os.path.join(self.features_dir, f"{anomaly_id}.npy")
            np.save(features_path, features)
        
        # Create registry entry
        entry = {
            "id": anomaly_id,
            "audio_hash": audio_hash,
            "source_file": os.path.basename(source_file),
            "start_time": round(start_time, 2),
            "end_time": round(end_time, 2),
            "duration_seconds": round(end_time - start_time, 2),
            "anomaly_score": round(anomaly_score, 6) if anomaly_score else None,
            "category": category,
            "notes": notes,
            "audio_file": os.path.basename(audio_path),
            "features_file": os.path.basename(features_path) if features_path else None,
            "added_at": datetime.now().isoformat()
        }
        
        self.registry["anomalies"].append(entry)
        self._save_registry()
        
        print(f"  [Added] {anomaly_id}: {source_file} @ {start_time:.1f}-{end_time:.1f}s ({category})")
        return anomaly_id
    
    def add_anomaly_from_segment(
        self,
        source_audio_path: str,
        start_time: float,
        end_time: float,
        anomaly_score: float = None,
        category: str = "unknown",
        notes: str = ""
    ) -> Optional[str]:
        """
        Add anomaly by extracting from source file.
        
        Args:
            source_audio_path: Path to source audio file
            start_time: Start time in seconds
            end_time: End time in seconds
            anomaly_score: Model's anomaly score
            category: Anomaly category
            notes: Additional notes
        
        Returns:
            Anomaly ID if added
        """
        # Load audio segment
        audio, _ = librosa.load(
            source_audio_path,
            sr=SAMPLE_RATE,
            mono=True,
            offset=start_time,
            duration=end_time - start_time
        )
        
        return self.add_anomaly(
            audio=audio,
            source_file=source_audio_path,
            start_time=start_time,
            end_time=end_time,
            anomaly_score=anomaly_score,
            category=category,
            notes=notes
        )
    
    def extract_uncleaned_portions(
        self,
        raw_audio_path: str,
        cleaned_audio_path: str,
        threshold_db: float = -40,
        min_duration: float = 0.5,
        category: str = "non_vehicle"
    ) -> List[str]:
        """
        Extract portions present in raw but removed in cleaned version.
        
        Uses energy difference to find removed segments.
        
        Args:
            raw_audio_path: Path to original (uncleaned) audio
            cleaned_audio_path: Path to cleaned audio
            threshold_db: Energy difference threshold (dB)
            min_duration: Minimum duration for extracted segments (seconds)
            category: Category for extracted anomalies
        
        Returns:
            List of added anomaly IDs
        """
        print(f"\nExtracting uncleaned portions...")
        print(f"  Raw: {os.path.basename(raw_audio_path)}")
        print(f"  Cleaned: {os.path.basename(cleaned_audio_path)}")
        
        # Load both files
        raw_audio, _ = librosa.load(raw_audio_path, sr=SAMPLE_RATE, mono=True)
        cleaned_audio, _ = librosa.load(cleaned_audio_path, sr=SAMPLE_RATE, mono=True)
        
        # Ensure same length (use shorter)
        min_len = min(len(raw_audio), len(cleaned_audio))
        raw_audio = raw_audio[:min_len]
        cleaned_audio = cleaned_audio[:min_len]
        
        # Compute energy difference in windows
        window_samples = int(0.1 * SAMPLE_RATE)  # 100ms windows
        hop_samples = window_samples // 2
        
        differences = []
        for start in range(0, min_len - window_samples, hop_samples):
            end = start + window_samples
            
            raw_energy = np.mean(raw_audio[start:end] ** 2) + 1e-10
            clean_energy = np.mean(cleaned_audio[start:end] ** 2) + 1e-10
            
            # If raw has significantly more energy, it was removed
            diff_db = 10 * np.log10(raw_energy / clean_energy)
            
            if diff_db > abs(threshold_db):
                time_sec = start / SAMPLE_RATE
                differences.append((time_sec, diff_db))
        
        if not differences:
            print("  No significant differences found")
            return []
        
        # Group consecutive differences into segments
        segments = []
        current_start = differences[0][0]
        current_end = current_start + 0.1
        
        for time, diff in differences[1:]:
            if time <= current_end + 0.2:  # Allow 200ms gap
                current_end = time + 0.1
            else:
                if current_end - current_start >= min_duration:
                    segments.append((current_start, current_end))
                current_start = time
                current_end = time + 0.1
        
        # Don't forget last segment
        if current_end - current_start >= min_duration:
            segments.append((current_start, current_end))
        
        print(f"  Found {len(segments)} removed segments")
        
        # Add each segment to dataset
        added_ids = []
        for start, end in segments:
            # Add margin
            start = max(0, start - 0.1)
            end = min(len(raw_audio) / SAMPLE_RATE, end + 0.1)
            
            start_sample = int(start * SAMPLE_RATE)
            end_sample = int(end * SAMPLE_RATE)
            audio_segment = raw_audio[start_sample:end_sample]
            
            anomaly_id = self.add_anomaly(
                audio=audio_segment,
                source_file=raw_audio_path,
                start_time=start,
                end_time=end,
                category=category,
                notes=f"Auto-extracted from cleaned version diff"
            )
            if anomaly_id:
                added_ids.append(anomaly_id)
        
        return added_ids
    
    def get_all_features(self, normalized: bool = False, normalizer=None) -> np.ndarray:
        """
        Load all anomaly features for evaluation.
        
        Args:
            normalized: Whether to apply normalization
            normalizer: FeatureNormalizer instance (required if normalized=True)
        
        Returns:
            Array of shape (n_samples, n_mels, n_time, 1)
        """
        all_features = []
        
        for anomaly in self.registry["anomalies"]:
            if anomaly.get("features_file"):
                features_path = os.path.join(self.features_dir, anomaly["features_file"])
                if os.path.exists(features_path):
                    features = np.load(features_path)
                    all_features.append(features)
        
        if not all_features:
            return np.array([])
        
        # Concatenate all features
        features = np.concatenate(all_features, axis=0)
        
        # Add channel dimension if needed
        if len(features.shape) == 3:
            features = np.expand_dims(features, axis=-1)
        
        # Normalize if requested
        if normalized and normalizer:
            features = normalizer.transform(features)
        
        return features
    
    def get_stats(self) -> Dict:
        """Get dataset statistics."""
        return self.registry["stats"]
    
    def print_summary(self):
        """Print dataset summary."""
        stats = self.get_stats()
        
        print("\n" + "=" * 60)
        print("LABELED ANOMALY DATASET")
        print("=" * 60)
        print(f"Total anomalies: {stats['total_count']}")
        print(f"Total duration: {stats['total_duration_seconds']:.1f}s ({stats['total_duration_seconds']/60:.1f} min)")
        
        if stats.get("by_category"):
            print("\nBy category:")
            for cat, count in stats["by_category"].items():
                print(f"  {cat}: {count}")
        
        print("-" * 60)


def extract_all_uncleaned_portions(base_dir: str = None):
    """
    Extract uncleaned portions from all raw/cleaned pairs.
    """
    if base_dir is None:
        base_dir = BASE_DIR
    
    raw_dir = os.path.join(base_dir, "data", "raw")
    dataset = LabeledAnomalyDataset(base_dir)
    
    # Find raw/cleaned pairs
    pairs = []
    for f in os.listdir(raw_dir):
        if f.endswith("_cleaned.wav"):
            raw_name = f.replace("_cleaned.wav", ".m4a")
            raw_path = os.path.join(raw_dir, raw_name)
            cleaned_path = os.path.join(raw_dir, f)
            
            if os.path.exists(raw_path):
                pairs.append((raw_path, cleaned_path))
    
    print(f"Found {len(pairs)} raw/cleaned pairs")
    
    total_added = 0
    for raw_path, cleaned_path in pairs:
        added = dataset.extract_uncleaned_portions(raw_path, cleaned_path)
        total_added += len(added)
    
    print(f"\nTotal anomalies extracted: {total_added}")
    dataset.print_summary()
    
    return dataset


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--extract":
        extract_all_uncleaned_portions()
    else:
        dataset = LabeledAnomalyDataset()
        dataset.print_summary()

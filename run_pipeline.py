"""
AURA-VAE Iterative Training Pipeline

Supports the iterative workflow:
1. Evaluate new recordings with latest model
2. Clean audio (remove non-vehicle anomalies)
3. Add cleaned audio to training set
4. Retrain/test/eval with combined dataset
5. Convert to TFLite and deploy

Usage:
    # Full pipeline with all audio files
    python run_pipeline.py
    
    # Evaluate a new recording
    python run_pipeline.py --evaluate "path/to/new_recording.m4a"
    
    # Add a cleaned audio file to training set
    python run_pipeline.py --add-audio "path/to/cleaned_audio.wav"
    
    # Retrain only (skip preprocessing if features exist)
    python run_pipeline.py --retrain
    
    # Skip specific steps
    python run_pipeline.py --skip-train --skip-eval
"""

import argparse
import sys
import os
import json
import numpy as np
from datetime import datetime

# Add python directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))


# =============================================================================
# AUDIO REGISTRY - Loaded from external JSON file for safety
# =============================================================================
AUDIO_REGISTRY_FILE = os.path.join(os.path.dirname(__file__), "audio_registry.json")

def load_audio_registry():
    """Load audio file list from external JSON registry."""
    if os.path.exists(AUDIO_REGISTRY_FILE):
        with open(AUDIO_REGISTRY_FILE, 'r') as f:
            registry = json.load(f)
        return registry.get('training_audio_files', [])
    else:
        # Fallback to hardcoded list if registry doesn't exist
        print(f"[WARNING] Audio registry not found at {AUDIO_REGISTRY_FILE}, using defaults")
        return [
            "Filante Sound Acquisition.m4a",
            "Filante Sound Acquisition 2_cleaned.wav",
            "Filante Sound Acquisition 3_cleaned.wav",
            "Filante Sound Acquisition 4_cleaned.wav",
        ]

def save_audio_registry(audio_files):
    """Save audio file list to external JSON registry."""
    registry = {
        "description": "Registry of audio files used for training. Edit this file to add/remove training audio.",
        "training_audio_files": audio_files
    }
    with open(AUDIO_REGISTRY_FILE, 'w') as f:
        json.dump(registry, f, indent=2)
    print(f"  Updated audio registry: {AUDIO_REGISTRY_FILE}")


def get_audio_files(base_dir):
    """Get list of existing audio files from registry."""
    training_audio_files = load_audio_registry()
    existing = []
    missing = []
    for audio_file in training_audio_files:
        path = os.path.join(base_dir, audio_file)
        if os.path.exists(path):
            existing.append(path)
        else:
            missing.append(audio_file)
    return existing, missing

# =============================================================================
# HELPER FUNCTIONS - EXPERIMENT MANAGEMENT
# =============================================================================
def get_experiment_dir(tag="run"):
    """Create a versioned directory for this execution."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_name = f"{timestamp}_{tag}"
    exp_dir = os.path.join(os.path.dirname(__file__), "experiments", exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Create subdirectories
    os.makedirs(os.path.join(exp_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "inference"), exist_ok=True)
    
    print(f"\n[EXPERIMENT] Created experiment directory: {exp_name}")
    return exp_dir

def update_production_models(exp_dir):
    """Copy successful models from experiment dir to root models/ for Android deployment."""
    prod_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(prod_dir, exist_ok=True)
    
    import shutil
    
    # Files to copy
    files = [
        ("models/vae_model.tflite", "vae_model.tflite"),
        ("models/anomaly_detector.tflite", "anomaly_detector.tflite"),
        ("models/android_config.json", "android_config.json"),
        ("models/normalization_params.json", "normalization_params.json"),
        ("models/threshold_config.json", "threshold_config.json"),
        ("models/vae_model.weights.h5", "vae_model.weights.h5") 
    ]
    
    print("\n[DEPLOY] Updating production models in /models/...")
    for src_rel, dst_rel in files:
        src = os.path.join(exp_dir, src_rel)
        dst = os.path.join(prod_dir, dst_rel)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  Updated: {dst_rel}")
        else:
            pass # Silent skip for missing files


def preprocess_combined_audio(audio_files, output_dir):
    """Preprocess and combine multiple audio files."""
    import librosa
    from config import (SAMPLE_RATE, SEGMENT_SAMPLES, HOP_SAMPLES, 
                        N_FFT, HOP_LENGTH, N_MELS, F_MIN, F_MAX, N_TIME_FRAMES)
    
    print(f"\nProcessing {len(audio_files)} audio files...")
    
    all_audio = []
    total_duration = 0
    
    for i, audio_path in enumerate(audio_files):
        print(f"\n  [{i+1}/{len(audio_files)}] Loading: {os.path.basename(audio_path)}")
        audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        duration = len(audio) / SAMPLE_RATE
        print(f"       Duration: {duration:.1f}s ({duration/60:.1f} min)")
        all_audio.append(audio)
        total_duration += duration
    
    # Combine all audio
    combined = np.concatenate(all_audio)
    print(f"\n  Combined duration: {total_duration:.1f}s ({total_duration/60:.1f} min)")
    
    # Segment into windows
    print("\n  Segmenting audio...")
    segments = []
    for start in range(0, len(combined) - SEGMENT_SAMPLES + 1, HOP_SAMPLES):
        segment = combined[start:start + SEGMENT_SAMPLES]
        segments.append(segment)
    
    segments = np.array(segments)
    print(f"  Total segments: {len(segments)}")
    
    # Extract mel spectrograms
    print("\n  Extracting mel spectrograms...")
    features = []
    for i, seg in enumerate(segments):
        mel_spec = librosa.feature.melspectrogram(
            y=seg, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH,
            n_mels=N_MELS, fmin=F_MIN, fmax=F_MAX
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Ensure correct shape
        if mel_spec_db.shape[1] > N_TIME_FRAMES:
            mel_spec_db = mel_spec_db[:, :N_TIME_FRAMES]
        elif mel_spec_db.shape[1] < N_TIME_FRAMES:
            pad = N_TIME_FRAMES - mel_spec_db.shape[1]
            mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad)), mode='constant')
        
        features.append(mel_spec_db)
        
        if (i + 1) % 1000 == 0:
            print(f"    Processed {i+1}/{len(segments)}...")
    
    features = np.array(features)
    print(f"  Features shape: {features.shape}")
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "combined_segments.npy"), segments)
    np.save(os.path.join(output_dir, "combined_features.npy"), features)
    
    # Save metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "audio_files": [os.path.basename(f) for f in audio_files],
        "total_duration_seconds": total_duration,
        "num_segments": len(segments),
        "feature_shape": list(features.shape)
    }
    with open(os.path.join(output_dir, "preprocessing_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    return features, segments


# [Function removed - using python.train.train_vae instead]



def evaluate_new_recording(audio_path, output_dir=None):
    """Evaluate a new recording and show top anomalies."""
    from inference import run_inference
    import shutil
    
    # Use provided output directory or default results/metrics
    if output_dir:
        inference_out_dir = os.path.join(output_dir, "inference")
    else:
        inference_out_dir = "results/inference"
        
    print(f"\nEvaluating: {audio_path}")
    results = run_inference(audio_path, inference_out_dir)
    
    # Show top anomalies for review
    print("\n" + "="*60)
    print("TOP 10 ANOMALY REGIONS FOR REVIEW")
    print("="*60)
    
    segments = results['segment_results']
    threshold = results['threshold']
    
    # Group consecutive anomalies
    anomaly_segments = [(i, s) for i, s in enumerate(segments) if s['is_anomaly']]
    
    if not anomaly_segments:
        print("\n✓ No anomalies detected!")
        return results
    
    regions = []
    current_start = anomaly_segments[0][1]['start_time']
    current_end = anomaly_segments[0][1]['end_time']
    max_score = anomaly_segments[0][1]['anomaly_score']
    
    for i, seg in anomaly_segments[1:]:
        if seg['start_time'] <= current_end + 0.5:
            current_end = seg['end_time']
            max_score = max(max_score, seg['anomaly_score'])
        else:
            regions.append((current_start, current_end, max_score))
            current_start = seg['start_time']
            current_end = seg['end_time']
            max_score = seg['anomaly_score']
    
    regions.append((current_start, current_end, max_score))
    
    # Sort by score
    regions_sorted = sorted(regions, key=lambda x: x[2], reverse=True)
    
    print("\nRank   Time (MM:SS)           Duration    Max Score")
    print("-"*60)
    
    for i, (start, end, score) in enumerate(regions_sorted[:10]):
        start_mm = int(start // 60)
        start_ss = start % 60
        end_mm = int(end // 60)
        end_ss = end % 60
        duration = end - start
        print(f"{i+1:<6} {start_mm:02d}:{start_ss:04.1f} - {end_mm:02d}:{end_ss:04.1f}     {duration:.1f}s        {score:.4f}")
    
    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print("1. Listen to each region in the audio file")
    print("2. Identify which are real anomalies (paper, talking, etc.)")
    print("3. Create cleaned audio by removing those regions")
    print("4. Add cleaned audio to TRAINING_AUDIO_FILES in run_pipeline.py")
    print("5. Run: python run_pipeline.py --retrain")
    
    # Interactive Review
    interactive_anomaly_review(audio_path, regions_sorted)
    
    return results


def interactive_anomaly_review(audio_path, regions_sorted):
    """
    Interactive review of anomalies with playback and removal.
    """
    import soundfile as sf
    import librosa
    import numpy as np
    try:
        import winsound
    except ImportError:
        print("\nWinsound not available (Windows only). Skipping interactive playback.")
        return

    print("\n" + "="*60)
    print("INTERACTIVE ANOMALY REVIEW")
    print("="*60)
    
    confirm = input("Start interactive review of top 10 anomalies? [y/N]: ").strip().lower()
    if confirm != 'y':
        return

    # Load audio once
    print(f"\nLoading audio for playback: {os.path.basename(audio_path)}...")
    try:
        # Load with native SR for playback
        y, sr = librosa.load(audio_path, sr=None, mono=True)
    except Exception as e:
        print(f"Error loading audio: {e}")
        return
    
    # Store regions to remove (start, end)
    to_remove = []
    
    # Only review top 10
    total_anomalies = min(10, len(regions_sorted))
    
    for i, (start, end, score) in enumerate(regions_sorted[:total_anomalies]):
        # Convert to samples
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        
        # Add 0.5s padding for context
        pad = int(0.5 * sr)
        play_start = max(0, start_sample - pad)
        play_end = min(len(y), end_sample + pad)
        
        snippet = y[play_start:play_end]
        
        print(f"\nAnomaly {i+1}/{total_anomalies}: {int(start//60):02d}:{start%60:04.1f} - {int(end//60):02d}:{end%60:04.1f} (Score: {score:.4f})")
        
        while True:
            print("  Playing clip... ", end="", flush=True)
            # Save temp wav for winsound
            temp_wav = "temp_review.wav"
            try:
                sf.write(temp_wav, snippet, sr)
                winsound.PlaySound(temp_wav, winsound.SND_FILENAME)
                print("Done.")
            except Exception as e:
                print(f"Playback error: {e}")
                break
            
            choice = input("  Is this an anomaly to REMOVE? [y=Yes / n=No / r=Replay]: ").strip().lower()
            if choice == 'r':
                continue
            elif choice == 'y':
                to_remove.append((start, end))
                print("  MARKED FOR REMOVAL.")
                break
            elif choice == 'n':
                print("  Kept as normal.")
                break
            else:
                print("  Invalid choice. n assumed.")
                break
                
    # Cleanup temp
    if os.path.exists("temp_review.wav"):
        try:
            os.remove("temp_review.wav")
        except:
            pass
        
    if not to_remove:
        print("\nNo regions marked for removal.")
        return
        
    print(f"\nMarked {len(to_remove)} regions for removal. Generating cleaned audio...")
    
    # Sort removal regions descending by start time to avoid index shift issues?
    # No, we will make a boolean mask.
    mask = np.ones(len(y), dtype=bool)
    for start, end in to_remove:
        s = int(start * sr)
        e = int(end * sr)
        mask[s:e] = False
    
    y_cleaned = y[mask]
    
    # Filename
    filename = os.path.basename(audio_path)
    name, ext = os.path.splitext(filename)
    if "_cleaned" in name:
        # Avoid _cleaned_cleaned
        clean_name = f"{name}.wav"
    else:
        clean_name = f"{name}_cleaned.wav"
        
    output_path = os.path.join(os.path.dirname(audio_path), clean_name)
    
    sf.write(output_path, y_cleaned, sr)
    print(f"✓ Saved cleaned audio: {output_path}")
    print(f"  Removed {len(y) - len(y_cleaned)} samples ({(len(y) - len(y_cleaned))/sr:.1f}s)")
    
    # Auto-add to registry (using external JSON file)
    add_confirm = input(f"\nAdd '{clean_name}' to training registry? [y/N]: ").strip().lower()
    if add_confirm == 'y':
        try:
            # Load current registry
            current_files = load_audio_registry()
            
            if clean_name not in current_files:
                current_files.append(clean_name)
                save_audio_registry(current_files)
                print(f"✓ Added '{clean_name}' to audio_registry.json")
                
                # Auto-retrain
                retrain = input("\nStart retraining pipeline now? (Will preprocess all files including new one) [y/N]: ").strip().lower()
                if retrain == 'y':
                    import subprocess
                    print("\n" + "="*60)
                    print("TRIGGERING RETRAINING PIPELINE")
                    print("="*60)
                    subprocess.check_call([sys.executable, __file__])
                    sys.exit(0)
            else:
                print(f"! '{clean_name}' already in registry.")
        except Exception as e:
            print(f"! Error updating registry: {e}")


def main():
    parser = argparse.ArgumentParser(description='AURA-VAE Iterative Training Pipeline')
    parser.add_argument('--evaluate', type=str, metavar='AUDIO_PATH',
                        help='Evaluate a new recording with the current model')
    parser.add_argument('--add-audio', type=str, metavar='AUDIO_PATH',
                        help='Add a cleaned audio file path to display (manual edit needed)')
    parser.add_argument('--retrain', action='store_true',
                        help='Retrain using existing combined features')
    parser.add_argument('--skip-preprocess', action='store_true',
                        help='Skip preprocessing (use existing features)')
    parser.add_argument('--skip-train', action='store_true',
                        help='Skip training (use existing model)')
    parser.add_argument('--skip-eval', action='store_true',
                        help='Skip evaluation')
    parser.add_argument('--skip-convert', action='store_true',
                        help='Skip TFLite conversion')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of training epochs')
    args = parser.parse_args()
    
    from config import BASE_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
    
    print("="*60)
    print("AURA-VAE Iterative Training Pipeline")
    print("="*60)
    
    # Mode: Evaluate new recording
    if args.evaluate:
        # Create experiment directory for inference
        filename = os.path.basename(args.evaluate).split('.')[0]
        # Sanitize filename
        filename = "".join([c for c in filename if c.isalpha() or c.isdigit() or c==' ' or c=='_']).rstrip()
        exp_dir = get_experiment_dir(tag=f"eval_{filename}")
        
        evaluate_new_recording(args.evaluate, output_dir=exp_dir)
        print(f"\nResults saved to: {exp_dir}")
        return
    
    # Mode: Show how to add audio
    if args.add_audio:
        print(f"\nTo add '{args.add_audio}' to training:")
        print(f"\n1. Edit run_pipeline.py")
        print(f"2. Add to TRAINING_AUDIO_FILES list:")
        print(f'   "{os.path.basename(args.add_audio)}",')
        print(f"\n3. Run: python run_pipeline.py")
        return
    
    # ==========================
    # TRAINING PIPELINE START
    # ==========================
    
    # Create experiment directory for this training run
    if args.retrain or not args.skip_train:
         exp_dir = get_experiment_dir(tag="train")
    else:
         # If just preprocessing not creating full experiment unless needed
         exp_dir = get_experiment_dir(tag="preprocess")

    # Get audio files (Look in RAW_DATA_DIR)
    audio_files, missing = get_audio_files(RAW_DATA_DIR)
    
    print(f"\nTraining audio files ({len(audio_files)} found):")
    for f in audio_files:
        print(f"  ✓ {os.path.basename(f)}")
    if missing:
        print(f"\nMissing files ({len(missing)}):")
        for f in missing:
            print(f"  ✗ {f}")
    
    if not audio_files:
        print("\nERROR: No audio files found!")
        return
    
    # Step 1: Preprocessing
    # Note: We save combined features to PROCESSED_DATA_DIR (cache) AND exp_dir
    features_path_cache = os.path.join(PROCESSED_DATA_DIR, "combined_features.npy")
    features_path_exp = os.path.join(exp_dir, "combined_features.npy")
    segments_path_exp = os.path.join(exp_dir, "combined_segments.npy")

    if not args.skip_preprocess and not args.retrain:
        print("\n" + "="*60)
        print("STEP 1: PREPROCESSING")
        print("="*60)
        # Preprocess saves to the output directory provided
        features, segments = preprocess_combined_audio(audio_files, exp_dir)
        # Verify it was saved there
        if not os.path.exists(features_path_exp):
             np.save(features_path_exp, features)
             np.save(segments_path_exp, segments)
             
    elif os.path.exists(features_path_cache):
        print("\n[Using existing preprocessed features from cache]")
        # Copy to experiment dir for reproducibility
        import shutil
        shutil.copy2(features_path_cache, features_path_exp)
        segments_path_cache = os.path.join(PROCESSED_DATA_DIR, "combined_segments.npy")
        if os.path.exists(segments_path_cache):
             shutil.copy2(segments_path_cache, segments_path_exp)
    else:
        print("\nERROR: No features found. Run without --skip-preprocess")
        return
    
    # Step 2: Training
    if not args.skip_train:
        print("\n" + "="*60)
        print("STEP 2: TRAINING")
        print("="*60)
        
        if args.epochs:
            import config
            config.EPOCHS = args.epochs
            print(f"Overriding epochs to: {args.epochs}")

        from train import train_vae
        # Pass exp_dir to save models there
        vae, history, normalizer = train_vae(output_dir=exp_dir)
        
        norm_params = {'mean': normalizer.mean, 'std': normalizer.std}
        
        # Step 3: Evaluation
        if not args.skip_eval:
            print("\n" + "="*60)
            print("STEP 3: EVALUATION")
            print("="*60)
            from evaluate import evaluate_model
            # Save metrics/plots to exp_dir
            evaluate_model(vae, norm_params=norm_params, features_path=features_path_exp, segments_path=segments_path_exp, output_dir=exp_dir)
            
        # Step 4: Convert
        if not args.skip_convert:
            print("\n" + "="*60)
            print("STEP 4: TFLITE CONVERSION")
            print("="*60)
            from convert_tflite import convert_vae_to_tflite
            # Save tflite to exp_dir/models
            convert_vae_to_tflite(vae, norm_params, output_dir=exp_dir)
            
            # Step 5: Update Production
            update_production_models(exp_dir)
            
    else:
        print("\n[Skipping training]")
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"Results saved to: {exp_dir}")
    print("Production models updated in: models/")
    
    print("\n" + "-"*60)
    print("ITERATIVE WORKFLOW:")
    print("-"*60)
    print("1. Record new audio in vehicle")
    print("2. Evaluate: python run_pipeline.py --evaluate 'new_recording.m4a'")
    print("3. Review top anomalies, identify non-vehicle sounds")
    print("4. Create cleaned audio (remove paper, talking, etc.)")
    print("5. Add cleaned file to TRAINING_AUDIO_FILES in this script")
    print("6. Retrain: python run_pipeline.py")
    print("7. Deploy new model to Android")
    
    # Copy instructions
    android_assets = os.path.join(BASE_DIR, 'android', 'AuraVAE', 'app', 'src', 'main', 'assets')
    print(f"\nTo deploy to Android:")
    print(f"  copy models\\anomaly_detector.tflite {android_assets}")
    print(f"  copy models\\*.json {android_assets}")


if __name__ == "__main__":
    main()

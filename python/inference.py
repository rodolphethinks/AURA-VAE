"""
Inference script for AURA-VAE Anomaly Detection
Evaluates a new audio file and detects anomalies
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf
import librosa
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python.config import *


def load_audio(audio_path: str) -> np.ndarray:
    """Load and resample audio file."""
    print(f"Loading audio: {audio_path}")
    audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    duration = len(audio) / sr
    print(f"  Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
    print(f"  Sample rate: {sr} Hz")
    return audio


def extract_mel_spectrogram(audio_segment: np.ndarray) -> np.ndarray:
    """Extract mel spectrogram from audio segment."""
    mel_spec = librosa.feature.melspectrogram(
        y=audio_segment,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=F_MIN,
        fmax=F_MAX
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db


def segment_audio(audio: np.ndarray) -> list:
    """Segment audio into overlapping windows."""
    segment_samples = SEGMENT_SAMPLES
    hop_samples = HOP_SAMPLES
    
    segments = []
    for start in range(0, len(audio) - segment_samples + 1, hop_samples):
        segment = audio[start:start + segment_samples]
        segments.append({
            'audio': segment,
            'start_time': start / SAMPLE_RATE,
            'end_time': (start + segment_samples) / SAMPLE_RATE
        })
    
    print(f"  Created {len(segments)} segments")
    return segments


def process_segments(segments: list, norm_params: dict) -> np.ndarray:
    """Extract and normalize features from all segments."""
    features = []
    
    for seg in segments:
        mel_spec = extract_mel_spectrogram(seg['audio'])
        
        # Ensure correct shape
        if mel_spec.shape[1] > N_TIME_FRAMES:
            mel_spec = mel_spec[:, :N_TIME_FRAMES]
        elif mel_spec.shape[1] < N_TIME_FRAMES:
            pad_width = N_TIME_FRAMES - mel_spec.shape[1]
            mel_spec = np.pad(mel_spec, ((0, 0), (0, pad_width)), mode='constant')
        
        features.append(mel_spec)
    
    features = np.array(features)
    
    # Normalize
    features = (features - norm_params['mean']) / norm_params['std']
    
    # Add channel dimension
    features = features[..., np.newaxis]
    
    return features


def run_inference(audio_path: str, output_dir: str = None):
    """Run anomaly detection inference on an audio file."""
    
    # Setup paths
    project_root = Path(__file__).parent.parent
    models_dir = project_root / "models"
    
    if output_dir is None:
        output_dir = project_root / "results" / "inference"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load normalization parameters
    norm_path = models_dir / "normalization_params.json"
    with open(norm_path, 'r') as f:
        norm_params = json.load(f)
    print(f"Loaded normalization params: mean={norm_params['mean']:.4f}, std={norm_params['std']:.4f}")
    
    # Load threshold config
    threshold_path = models_dir / "threshold_config.json"
    with open(threshold_path, 'r') as f:
        threshold_config = json.load(f)
    threshold = threshold_config['threshold']
    print(f"Anomaly threshold: {threshold:.6f}")
    
    # Load TFLite model
    model_path = models_dir / "anomaly_detector.tflite"
    print(f"Loading TFLite model: {model_path}")
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"  Input shape: {input_details[0]['shape']}")
    print(f"  Output shape: {output_details[0]['shape']}")
    
    # Load and process audio
    print("\n" + "="*60)
    print("PROCESSING AUDIO")
    print("="*60)
    
    audio = load_audio(audio_path)
    segments = segment_audio(audio)
    features = process_segments(segments, norm_params)
    
    print(f"  Features shape: {features.shape}")
    
    # Run inference
    print("\n" + "="*60)
    print("RUNNING INFERENCE")
    print("="*60)
    
    anomaly_scores = []
    
    for i, feature in enumerate(features):
        # Prepare input
        input_data = feature[np.newaxis, ...].astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        
        # Run inference
        interpreter.invoke()
        
        # Get output (anomaly score)
        output = interpreter.get_tensor(output_details[0]['index'])
        score = output.flatten()[0]  # Handle different output shapes
        anomaly_scores.append(score)
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(features)} segments...")
    
    anomaly_scores = np.array(anomaly_scores)
    
    # Analyze results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    is_anomaly = anomaly_scores > threshold
    num_anomalies = np.sum(is_anomaly)
    anomaly_percentage = 100 * num_anomalies / len(anomaly_scores)
    
    print(f"\nTotal segments: {len(segments)}")
    print(f"Anomalous segments: {num_anomalies} ({anomaly_percentage:.1f}%)")
    print(f"Normal segments: {len(segments) - num_anomalies} ({100-anomaly_percentage:.1f}%)")
    
    print(f"\nScore Statistics:")
    print(f"  Min:    {anomaly_scores.min():.6f}")
    print(f"  Max:    {anomaly_scores.max():.6f}")
    print(f"  Mean:   {anomaly_scores.mean():.6f}")
    print(f"  Median: {np.median(anomaly_scores):.6f}")
    print(f"  Std:    {anomaly_scores.std():.6f}")
    print(f"  Threshold: {threshold:.6f}")
    
    # Find anomaly time ranges
    print("\n" + "="*60)
    print("ANOMALY TIMELINE")
    print("="*60)
    
    if num_anomalies > 0:
        anomaly_times = []
        for i, (seg, is_anom) in enumerate(zip(segments, is_anomaly)):
            if is_anom:
                anomaly_times.append({
                    'segment': i,
                    'start': seg['start_time'],
                    'end': seg['end_time'],
                    'score': anomaly_scores[i]
                })
        
        # Group consecutive anomalies
        print("\nDetected anomaly regions:")
        if len(anomaly_times) > 0:
            current_start = anomaly_times[0]['start']
            current_end = anomaly_times[0]['end']
            max_score = anomaly_times[0]['score']
            
            regions = []
            for at in anomaly_times[1:]:
                if at['start'] <= current_end + 0.5:  # Allow 0.5s gap
                    current_end = at['end']
                    max_score = max(max_score, at['score'])
                else:
                    regions.append((current_start, current_end, max_score))
                    current_start = at['start']
                    current_end = at['end']
                    max_score = at['score']
            
            regions.append((current_start, current_end, max_score))
            
            for i, (start, end, score) in enumerate(regions[:20]):  # Show first 20
                print(f"  Region {i+1}: {start:.1f}s - {end:.1f}s (duration: {end-start:.1f}s, max_score: {score:.4f})")
            
            if len(regions) > 20:
                print(f"  ... and {len(regions) - 20} more regions")
    else:
        print("\n✓ No anomalies detected! Audio appears normal.")
    
    # Create visualizations
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    audio_name = Path(audio_path).stem
    
    # Plot 1: Anomaly scores over time
    fig, axes = plt.subplots(3, 1, figsize=(16, 10))
    
    times = [seg['start_time'] for seg in segments]
    
    # Scores over time
    ax1 = axes[0]
    ax1.plot(times, anomaly_scores, 'b-', linewidth=0.5, alpha=0.7)
    ax1.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({threshold:.4f})')
    ax1.fill_between(times, 0, anomaly_scores, where=is_anomaly, color='red', alpha=0.3, label='Anomaly')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Anomaly Score')
    ax1.set_title(f'Anomaly Scores Over Time - {audio_name}')
    ax1.legend()
    ax1.set_xlim(0, times[-1])
    ax1.grid(True, alpha=0.3)
    
    # Score distribution
    ax2 = axes[1]
    ax2.hist(anomaly_scores, bins=100, color='blue', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax2.axvline(x=threshold, color='r', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.4f})')
    ax2.set_xlabel('Anomaly Score')
    ax2.set_ylabel('Count')
    ax2.set_title('Score Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Binary anomaly timeline
    ax3 = axes[2]
    ax3.fill_between(times, 0, 1, where=is_anomaly, color='red', alpha=0.7, label='Anomaly')
    ax3.fill_between(times, 0, 1, where=~is_anomaly, color='green', alpha=0.7, label='Normal')
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('Status')
    ax3.set_title('Anomaly Detection Timeline')
    ax3.set_yticks([0.5])
    ax3.set_yticklabels([''])
    ax3.set_xlim(0, times[-1])
    ax3.legend(loc='upper right')
    
    plt.tight_layout()
    plot_path = output_dir / f'{audio_name}_analysis.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved: {plot_path}")
    
    # Plot 2: Detailed view of top anomalies
    if num_anomalies > 0:
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        # Get top 8 anomalous segments
        top_indices = np.argsort(anomaly_scores)[-8:][::-1]
        
        for idx, ax in enumerate(axes):
            if idx < len(top_indices):
                seg_idx = top_indices[idx]
                mel_spec = extract_mel_spectrogram(segments[seg_idx]['audio'])
                
                ax.imshow(mel_spec, aspect='auto', origin='lower', cmap='viridis')
                ax.set_title(f"Score: {anomaly_scores[seg_idx]:.2f}\n{segments[seg_idx]['start_time']:.1f}s - {segments[seg_idx]['end_time']:.1f}s")
                ax.set_xlabel('Time')
                ax.set_ylabel('Mel Bin')
            else:
                ax.axis('off')
        
        plt.suptitle(f'Top Anomalous Segments - {audio_name}', fontsize=14)
        plt.tight_layout()
        plot_path = output_dir / f'{audio_name}_top_anomalies.png'
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Saved: {plot_path}")
    
    # Save detailed results to JSON
    results = {
        'audio_file': str(audio_path),
        'total_segments': len(segments),
        'anomalous_segments': int(num_anomalies),
        'anomaly_percentage': float(anomaly_percentage),
        'threshold': float(threshold),
        'score_stats': {
            'min': float(anomaly_scores.min()),
            'max': float(anomaly_scores.max()),
            'mean': float(anomaly_scores.mean()),
            'median': float(np.median(anomaly_scores)),
            'std': float(anomaly_scores.std())
        },
        'segment_results': [
            {
                'segment_id': i,
                'start_time': float(seg['start_time']),
                'end_time': float(seg['end_time']),
                'anomaly_score': float(anomaly_scores[i]),
                'is_anomaly': bool(is_anomaly[i])
            }
            for i, seg in enumerate(segments)
        ]
    }
    
    results_path = output_dir / f'{audio_name}_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {results_path}")
    
    print("\n" + "="*60)
    print("INFERENCE COMPLETE")
    print("="*60)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='AURA-VAE Anomaly Detection Inference')
    parser.add_argument('audio_path', type=str, help='Path to audio file to analyze')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for results')
    
    args = parser.parse_args()
    
    run_inference(args.audio_path, args.output_dir)


import os
import sys
import json
import numpy as np
import tensorflow as tf
import librosa
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Hardcoded config
SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 64
F_MIN = 50
F_MAX = 8000
N_TIME_FRAMES = 32
SPEC_AMIN = 1e-10
SPEC_TOP_DB = 80.0

def run_inference_with_setup(audio_path, model_path, norm_params, htk_mode, norm_mode):
    # Load audio
    audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    
    # Segment (just process first few segments for speed)
    SEGMENT_SAMPLES = int(SAMPLE_RATE * 1.0)
    HOP_SAMPLES = int(SAMPLE_RATE * 0.5)
    
    segments = []
    for start in range(0, len(audio) - SEGMENT_SAMPLES + 1, HOP_SAMPLES):
        segments.append(audio[start:start + SEGMENT_SAMPLES])
    
    # Load model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    scores = []
    for seg in segments:
        # Extract Mel
        mel_spec = librosa.feature.melspectrogram(
            y=seg,
            sr=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            fmin=F_MIN,
            fmax=F_MAX,
            htk=htk_mode,
            norm=norm_mode,
            power=2.0
        )
        
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max, amin=SPEC_AMIN, top_db=SPEC_TOP_DB)
        
        # Resize
        if mel_spec_db.shape[1] > N_TIME_FRAMES:
            mel_spec_db = mel_spec_db[:, :N_TIME_FRAMES]
        elif mel_spec_db.shape[1] < N_TIME_FRAMES:
            pad_width = N_TIME_FRAMES - mel_spec_db.shape[1]
            mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant', constant_values=-SPEC_TOP_DB)
            
        # Normalize with params
        feat = (mel_spec_db - norm_params['mean']) / norm_params['std']
        
        # Inference
        input_data = feat[np.newaxis, ..., np.newaxis].astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        scores.append(output.flatten()[0])
        
    scores = np.array(scores)
    print(f"HTK={htk_mode}, Norm={norm_mode} -> Mean Score: {scores.mean():.4f}, Max: {scores.max():.4f}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('audio_path')
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent
    model_path = str(project_root / "models/anomaly_detector.tflite")
    norm_path = project_root / "models/normalization_params.json"
    
    with open(norm_path, 'r') as f:
        norm_params = json.load(f)

    # 1. Standard (Librosa default) - This is what inference.py does
    run_inference_with_setup(args.audio_path, model_path, norm_params, htk_mode=False, norm_mode='slaney')

    # 2. HTK points, Slaney Norm (This is what Java current implementation looks like)
    run_inference_with_setup(args.audio_path, model_path, norm_params, htk_mode=True, norm_mode='slaney')
    
    # 3. Reference check (What Java does for mel calculation)
    # Print out frequencies to see difference
    mel_slaney = librosa.mel_frequencies(n_mels=66, fmin=50, fmax=8000, htk=False)
    mel_htk = librosa.mel_frequencies(n_mels=66, fmin=50, fmax=8000, htk=True)
    
    print("\nFreq Check (first 5):")
    print("Slaney:", mel_slaney[:5])
    print("HTK:   ", mel_htk[:5])
    print("Freq Check (last 5):")
    print("Slaney:", mel_slaney[-5:])
    print("HTK:   ", mel_htk[-5:])

if __name__ == "__main__":
    main()

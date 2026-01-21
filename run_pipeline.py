"""
AURA-VAE Full Pipeline Runner

Runs the complete pipeline:
1. Preprocessing
2. Training
3. Evaluation
4. TFLite conversion

Usage:
    python run_pipeline.py [--skip-train] [--epochs N]
"""

import argparse
import sys
import os

# Add python directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))


def main():
    parser = argparse.ArgumentParser(description='AURA-VAE Full Pipeline')
    parser.add_argument('--skip-preprocess', action='store_true', 
                        help='Skip preprocessing (use existing features)')
    parser.add_argument('--skip-train', action='store_true',
                        help='Skip training (use existing model)')
    parser.add_argument('--skip-eval', action='store_true',
                        help='Skip evaluation')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of training epochs')
    args = parser.parse_args()
    
    print("="*60)
    print("AURA-VAE Full Pipeline")
    print("="*60)
    
    # Step 1: Preprocessing
    if not args.skip_preprocess:
        print("\n" + "="*60)
        print("STEP 1: PREPROCESSING")
        print("="*60)
        from preprocessing import preprocess_source_audio, generate_synthetic_anomalies
        from config import PROCESSED_DATA_DIR
        import numpy as np
        
        features = preprocess_source_audio()
        anomalies = generate_synthetic_anomalies(n_samples=200)
        
        anomalies_path = os.path.join(PROCESSED_DATA_DIR, "synthetic_anomalies.npy")
        np.save(anomalies_path, anomalies)
        print(f"Saved synthetic anomalies to: {anomalies_path}")
    else:
        print("\n[Skipping preprocessing]")
    
    # Step 2: Training
    if not args.skip_train:
        print("\n" + "="*60)
        print("STEP 2: TRAINING")
        print("="*60)
        
        # Override epochs if specified
        if args.epochs:
            import config
            config.EPOCHS = args.epochs
            print(f"Overriding epochs to: {args.epochs}")
        
        from train import train_vae
        vae, history, normalizer = train_vae()
    else:
        print("\n[Skipping training]")
    
    # Step 3: Evaluation
    if not args.skip_eval:
        print("\n" + "="*60)
        print("STEP 3: EVALUATION")
        print("="*60)
        from evaluate import evaluate_model
        metrics = evaluate_model()
    else:
        print("\n[Skipping evaluation]")
    
    # Step 4: TFLite Conversion
    print("\n" + "="*60)
    print("STEP 4: TFLITE CONVERSION")
    print("="*60)
    from convert_tflite import convert_vae_to_tflite
    convert_vae_to_tflite()
    
    # Final Summary
    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)
    
    from config import MODELS_DIR, BASE_DIR
    print(f"\nGenerated artifacts in: {MODELS_DIR}")
    print("\nNext steps:")
    print("1. Copy TFLite models to Android assets:")
    print(f"   - {os.path.join(MODELS_DIR, 'vae_model.tflite')}")
    print(f"   - {os.path.join(MODELS_DIR, 'anomaly_detector.tflite')}")
    print(f"   - {os.path.join(MODELS_DIR, 'android_config.json')}")
    print(f"\n   To: {os.path.join(BASE_DIR, 'android', 'AuraVAE', 'app', 'src', 'main', 'assets')}")
    print("\n2. Open Android project in Android Studio")
    print("3. Build and deploy to device")


if __name__ == "__main__":
    main()

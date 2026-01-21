package com.neos.auravae.audio;

import android.util.Log;

/**
 * Mel Spectrogram Feature Extractor
 * 
 * Converts raw PCM audio to log-mel spectrograms for VAE input.
 * Parameters MUST match Python preprocessing exactly.
 * 
 * Based on librosa's mel spectrogram implementation.
 */
public class MelSpectrogramExtractor {
    
    private static final String TAG = "MelSpectrogram";
    
    // Audio parameters - MUST match config.py
    public static final int SAMPLE_RATE = 16000;
    public static final int N_FFT = 1024;
    public static final int HOP_LENGTH = 512;
    public static final int N_MELS = 64;
    public static final float F_MIN = 50f;
    public static final float F_MAX = 8000f;
    public static final int N_TIME_FRAMES = 32;
    
    // Spectrogram parameters
    private static final float SPEC_AMIN = 1e-10f;
    private static final float SPEC_TOP_DB = 80f;
    
    // Mel filterbank
    private float[][] melFilterbank;
    
    // Hanning window for STFT
    private float[] window;
    
    public MelSpectrogramExtractor() {
        initMelFilterbank();
        initWindow();
    }
    
    /**
     * Initialize Hanning window.
     */
    private void initWindow() {
        window = new float[N_FFT];
        for (int i = 0; i < N_FFT; i++) {
            window[i] = 0.5f * (1 - (float) Math.cos(2 * Math.PI * i / (N_FFT - 1)));
        }
    }
    
    /**
     * Initialize mel filterbank.
     * Creates triangular filters spaced in mel scale.
     */
    private void initMelFilterbank() {
        melFilterbank = new float[N_MELS][N_FFT / 2 + 1];
        
        // Convert frequency bounds to mel scale
        float melMin = hzToMel(F_MIN);
        float melMax = hzToMel(F_MAX);
        
        // Create equally spaced mel points
        float[] melPoints = new float[N_MELS + 2];
        for (int i = 0; i < melPoints.length; i++) {
            melPoints[i] = melMin + (melMax - melMin) * i / (N_MELS + 1);
        }
        
        // Convert back to Hz
        float[] hzPoints = new float[melPoints.length];
        for (int i = 0; i < melPoints.length; i++) {
            hzPoints[i] = melToHz(melPoints[i]);
        }
        
        // Convert to FFT bin indices
        int[] binPoints = new int[melPoints.length];
        for (int i = 0; i < melPoints.length; i++) {
            binPoints[i] = (int) Math.floor((N_FFT + 1) * hzPoints[i] / SAMPLE_RATE);
        }
        
        // Create triangular filters
        for (int m = 0; m < N_MELS; m++) {
            for (int k = binPoints[m]; k < binPoints[m + 1]; k++) {
                if (k >= 0 && k < melFilterbank[m].length) {
                    melFilterbank[m][k] = (float) (k - binPoints[m]) / (binPoints[m + 1] - binPoints[m]);
                }
            }
            for (int k = binPoints[m + 1]; k < binPoints[m + 2]; k++) {
                if (k >= 0 && k < melFilterbank[m].length) {
                    melFilterbank[m][k] = (float) (binPoints[m + 2] - k) / (binPoints[m + 2] - binPoints[m + 1]);
                }
            }
        }
    }
    
    /**
     * Convert Hz to mel scale.
     */
    private float hzToMel(float hz) {
        return 2595f * (float) Math.log10(1 + hz / 700f);
    }
    
    /**
     * Convert mel to Hz.
     */
    private float melToHz(float mel) {
        return 700f * ((float) Math.pow(10, mel / 2595f) - 1);
    }
    
    /**
     * Extract mel spectrogram from audio segment.
     * 
     * @param audio Audio samples (16-bit PCM as floats, normalized to [-1, 1])
     * @return Mel spectrogram of shape [N_MELS][N_TIME_FRAMES]
     */
    public float[][] extractMelSpectrogram(float[] audio) {
        // Calculate number of frames
        int numFrames = Math.max(1, (audio.length - N_FFT) / HOP_LENGTH + 1);
        
        // Compute STFT magnitude
        float[][] stftMag = new float[numFrames][N_FFT / 2 + 1];
        
        for (int frame = 0; frame < numFrames; frame++) {
            int start = frame * HOP_LENGTH;
            
            // Apply window and compute FFT
            float[] windowed = new float[N_FFT];
            for (int i = 0; i < N_FFT; i++) {
                int idx = start + i;
                if (idx < audio.length) {
                    windowed[i] = audio[idx] * window[i];
                }
            }
            
            // Compute FFT magnitude (simplified real FFT)
            float[] fftMag = computeFFTMagnitude(windowed);
            stftMag[frame] = fftMag;
        }
        
        // Apply mel filterbank
        float[][] melSpec = new float[N_MELS][numFrames];
        
        for (int frame = 0; frame < numFrames; frame++) {
            for (int m = 0; m < N_MELS; m++) {
                float sum = 0;
                for (int k = 0; k < melFilterbank[m].length; k++) {
                    sum += melFilterbank[m][k] * stftMag[frame][k];
                }
                melSpec[m][frame] = sum;
            }
        }
        
        // Convert to log scale (dB)
        float[][] logMelSpec = powerToDb(melSpec);
        
        // Ensure consistent time dimension
        float[][] output = new float[N_MELS][N_TIME_FRAMES];
        for (int m = 0; m < N_MELS; m++) {
            for (int t = 0; t < N_TIME_FRAMES; t++) {
                if (t < logMelSpec[m].length) {
                    output[m][t] = logMelSpec[m][t];
                } else {
                    output[m][t] = -SPEC_TOP_DB;  // Pad with min value
                }
            }
        }
        
        return output;
    }
    
    /**
     * Compute FFT magnitude (simplified real FFT).
     */
    private float[] computeFFTMagnitude(float[] signal) {
        int n = signal.length;
        float[] real = new float[n];
        float[] imag = new float[n];
        System.arraycopy(signal, 0, real, 0, n);
        
        // Simple DFT (for small N_FFT this is acceptable)
        // For production, use a proper FFT library
        fft(real, imag);
        
        // Compute magnitude for positive frequencies
        float[] mag = new float[n / 2 + 1];
        for (int k = 0; k <= n / 2; k++) {
            mag[k] = real[k] * real[k] + imag[k] * imag[k];  // Power spectrum
        }
        
        return mag;
    }
    
    /**
     * In-place Cooley-Tukey FFT.
     */
    private void fft(float[] real, float[] imag) {
        int n = real.length;
        
        // Bit reversal
        int j = 0;
        for (int i = 0; i < n - 1; i++) {
            if (i < j) {
                float tempR = real[i];
                float tempI = imag[i];
                real[i] = real[j];
                imag[i] = imag[j];
                real[j] = tempR;
                imag[j] = tempI;
            }
            int k = n / 2;
            while (k <= j) {
                j -= k;
                k /= 2;
            }
            j += k;
        }
        
        // FFT
        for (int len = 2; len <= n; len *= 2) {
            float angle = (float) (-2 * Math.PI / len);
            float wR = (float) Math.cos(angle);
            float wI = (float) Math.sin(angle);
            
            for (int i = 0; i < n; i += len) {
                float curWR = 1;
                float curWI = 0;
                
                for (int k = 0; k < len / 2; k++) {
                    int idx1 = i + k;
                    int idx2 = i + k + len / 2;
                    
                    float tR = curWR * real[idx2] - curWI * imag[idx2];
                    float tI = curWR * imag[idx2] + curWI * real[idx2];
                    
                    real[idx2] = real[idx1] - tR;
                    imag[idx2] = imag[idx1] - tI;
                    real[idx1] = real[idx1] + tR;
                    imag[idx1] = imag[idx1] + tI;
                    
                    float nextWR = curWR * wR - curWI * wI;
                    float nextWI = curWR * wI + curWI * wR;
                    curWR = nextWR;
                    curWI = nextWI;
                }
            }
        }
    }
    
    /**
     * Convert power spectrogram to dB scale.
     */
    private float[][] powerToDb(float[][] spec) {
        float[][] db = new float[spec.length][spec[0].length];
        float refValue = 1.0f;
        
        // Find max for reference
        float maxVal = SPEC_AMIN;
        for (float[] row : spec) {
            for (float val : row) {
                if (val > maxVal) {
                    maxVal = val;
                }
            }
        }
        refValue = maxVal;
        
        // Convert to dB
        for (int i = 0; i < spec.length; i++) {
            for (int j = 0; j < spec[i].length; j++) {
                float val = Math.max(SPEC_AMIN, spec[i][j]);
                db[i][j] = 10 * (float) Math.log10(val / refValue);
                db[i][j] = Math.max(db[i][j], -SPEC_TOP_DB);
            }
        }
        
        return db;
    }
}

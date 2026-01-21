package com.neos.auravae.ml;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.util.Log;

import com.google.gson.Gson;
import com.neos.auravae.audio.AudioRecorder;
import com.neos.auravae.audio.MelSpectrogramExtractor;

import org.tensorflow.lite.Interpreter;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;

/**
 * VAE-based Anomaly Detector
 * 
 * Performs on-device acoustic anomaly detection using a VAE model.
 * 
 * Pipeline:
 * 1. Segment audio into fixed-length windows
 * 2. Extract log-mel spectrograms
 * 3. Normalize features
 * 4. Run VAE inference
 * 5. Compute reconstruction error (anomaly score)
 * 6. Compare against threshold
 */
public class AnomalyDetector {
    
    private static final String TAG = "AnomalyDetector";
    
    // Model files in assets folder
    private static final String MODEL_FILE = "vae_model.tflite";
    private static final String DETECTOR_FILE = "anomaly_detector.tflite";
    private static final String CONFIG_FILE = "android_config.json";
    
    // Audio segmentation parameters
    private static final int SAMPLE_RATE = 16000;
    private static final int SEGMENT_SAMPLES = 16000;  // 1 second
    private static final int HOP_SAMPLES = 8000;  // 0.5 second hop (50% overlap)
    
    // Feature dimensions
    private static final int N_MELS = 64;
    private static final int N_TIME_FRAMES = 32;
    
    // TFLite interpreter
    private Interpreter interpreter;
    private Interpreter detectorInterpreter;
    
    // Feature extractor
    private MelSpectrogramExtractor melExtractor;
    
    // Normalization parameters
    private float normMean = 0f;
    private float normStd = 1f;
    
    // Anomaly threshold
    private float threshold = 0.1f;
    
    // Config
    private ModelConfig config;
    
    public AnomalyDetector(Context context) throws IOException {
        // Load configuration
        loadConfig(context);
        
        // Load TFLite models
        loadModels(context);
        
        // Initialize feature extractor
        melExtractor = new MelSpectrogramExtractor();
        
        Log.d(TAG, "AnomalyDetector initialized");
        Log.d(TAG, "  Normalization: mean=" + normMean + ", std=" + normStd);
        Log.d(TAG, "  Threshold: " + threshold);
    }
    
    /**
     * Load configuration from assets.
     */
    private void loadConfig(Context context) throws IOException {
        try {
            InputStream is = context.getAssets().open(CONFIG_FILE);
            BufferedReader reader = new BufferedReader(new InputStreamReader(is));
            StringBuilder sb = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) {
                sb.append(line);
            }
            reader.close();
            
            Gson gson = new Gson();
            config = gson.fromJson(sb.toString(), ModelConfig.class);
            
            if (config.normalization != null) {
                normMean = config.normalization.mean;
                normStd = config.normalization.std;
            }
            
            if (config.threshold != null) {
                threshold = config.threshold.threshold;
            }
            
            Log.d(TAG, "Configuration loaded from " + CONFIG_FILE);
            
        } catch (IOException e) {
            Log.w(TAG, "Could not load config file, using defaults: " + e.getMessage());
            // Use defaults
            normMean = -40f;  // Typical log-mel mean
            normStd = 20f;    // Typical log-mel std
            threshold = 0.1f;
        }
    }
    
    /**
     * Load TFLite models from assets.
     */
    private void loadModels(Context context) throws IOException {
        // Try to load the direct anomaly detector model first
        try {
            MappedByteBuffer detectorBuffer = loadModelFile(context, DETECTOR_FILE);
            detectorInterpreter = new Interpreter(detectorBuffer);
            Log.d(TAG, "Loaded detector model: " + DETECTOR_FILE);
        } catch (IOException e) {
            Log.w(TAG, "Could not load detector model: " + e.getMessage());
        }
        
        // Load reconstruction model as fallback
        try {
            MappedByteBuffer modelBuffer = loadModelFile(context, MODEL_FILE);
            interpreter = new Interpreter(modelBuffer);
            Log.d(TAG, "Loaded VAE model: " + MODEL_FILE);
        } catch (IOException e) {
            if (detectorInterpreter == null) {
                throw new IOException("Could not load any model: " + e.getMessage());
            }
            Log.w(TAG, "Could not load VAE model, using detector only");
        }
    }
    
    /**
     * Load model file as MappedByteBuffer.
     */
    private MappedByteBuffer loadModelFile(Context context, String filename) throws IOException {
        AssetFileDescriptor fileDescriptor = context.getAssets().openFd(filename);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }
    
    /**
     * Detect anomaly in audio data.
     * 
     * @param audioData Raw 16-bit PCM audio samples
     * @return Detection result
     */
    public DetectionResult detectAnomaly(short[] audioData) {
        Log.d(TAG, "Analyzing audio: " + audioData.length + " samples");
        
        // Convert to float and normalize to [-1, 1]
        float[] audio = convertToFloat(audioData);
        
        // Segment audio
        List<float[]> segments = segmentAudio(audio);
        Log.d(TAG, "Created " + segments.size() + " segments");
        
        if (segments.isEmpty()) {
            return new DetectionResult(false, 0f, threshold, 0, new float[0]);
        }
        
        // Process each segment and compute anomaly scores
        float[] segmentScores = new float[segments.size()];
        
        for (int i = 0; i < segments.size(); i++) {
            float[] segment = segments.get(i);
            
            // Extract mel spectrogram
            float[][] melSpec = melExtractor.extractMelSpectrogram(segment);
            
            // Normalize
            float[][] normalizedSpec = normalize(melSpec);
            
            // Compute anomaly score
            segmentScores[i] = computeAnomalyScore(normalizedSpec);
        }
        
        // Compute overall score (mean of segment scores)
        float meanScore = 0f;
        for (float score : segmentScores) {
            meanScore += score;
        }
        meanScore /= segmentScores.length;
        
        // Determine if anomaly
        boolean isAnomaly = meanScore > threshold;
        
        Log.d(TAG, "Detection complete: score=" + meanScore + ", threshold=" + threshold + 
                ", isAnomaly=" + isAnomaly);
        
        return new DetectionResult(isAnomaly, meanScore, threshold, segments.size(), segmentScores);
    }
    
    /**
     * Convert 16-bit PCM to float [-1, 1].
     */
    private float[] convertToFloat(short[] audioData) {
        float[] audio = new float[audioData.length];
        for (int i = 0; i < audioData.length; i++) {
            audio[i] = audioData[i] / 32768f;
        }
        return audio;
    }
    
    /**
     * Segment audio into overlapping windows.
     */
    private List<float[]> segmentAudio(float[] audio) {
        List<float[]> segments = new ArrayList<>();
        
        int numSegments = Math.max(1, (audio.length - SEGMENT_SAMPLES) / HOP_SAMPLES + 1);
        
        for (int i = 0; i < numSegments; i++) {
            int start = i * HOP_SAMPLES;
            int end = start + SEGMENT_SAMPLES;
            
            float[] segment = new float[SEGMENT_SAMPLES];
            
            if (end <= audio.length) {
                System.arraycopy(audio, start, segment, 0, SEGMENT_SAMPLES);
            } else {
                // Pad last segment
                int available = audio.length - start;
                if (available > 0) {
                    System.arraycopy(audio, start, segment, 0, available);
                }
            }
            
            segments.add(segment);
        }
        
        return segments;
    }
    
    /**
     * Normalize mel spectrogram features.
     */
    private float[][] normalize(float[][] spec) {
        float[][] normalized = new float[spec.length][spec[0].length];
        
        for (int i = 0; i < spec.length; i++) {
            for (int j = 0; j < spec[i].length; j++) {
                normalized[i][j] = (spec[i][j] - normMean) / normStd;
            }
        }
        
        return normalized;
    }
    
    /**
     * Compute anomaly score using TFLite model.
     */
    private float computeAnomalyScore(float[][] melSpec) {
        // Prepare input tensor [1, N_MELS, N_TIME_FRAMES, 1]
        float[][][][] input = new float[1][N_MELS][N_TIME_FRAMES][1];
        for (int i = 0; i < N_MELS; i++) {
            for (int j = 0; j < N_TIME_FRAMES; j++) {
                input[0][i][j][0] = melSpec[i][j];
            }
        }
        
        // Use detector model if available (outputs score directly)
        if (detectorInterpreter != null) {
            float[][] output = new float[1][1];
            detectorInterpreter.run(input, output);
            return output[0][0];
        }
        
        // Otherwise use reconstruction model and compute MSE
        if (interpreter != null) {
            float[][][][] output = new float[1][N_MELS][N_TIME_FRAMES][1];
            interpreter.run(input, output);
            
            // Compute MSE
            float mse = 0f;
            int count = 0;
            for (int i = 0; i < N_MELS; i++) {
                for (int j = 0; j < N_TIME_FRAMES; j++) {
                    float diff = input[0][i][j][0] - output[0][i][j][0];
                    mse += diff * diff;
                    count++;
                }
            }
            return mse / count;
        }
        
        // Fallback
        Log.w(TAG, "No model available for scoring");
        return 0f;
    }
    
    /**
     * Release resources.
     */
    public void close() {
        if (interpreter != null) {
            interpreter.close();
            interpreter = null;
        }
        if (detectorInterpreter != null) {
            detectorInterpreter.close();
            detectorInterpreter = null;
        }
    }
    
    /**
     * Configuration data classes.
     */
    private static class ModelConfig {
        int[] input_shape;
        int sample_rate;
        int n_fft;
        int hop_length;
        int n_mels;
        float f_min;
        float f_max;
        float segment_duration;
        int n_time_frames;
        NormalizationConfig normalization;
        ThresholdConfig threshold;
    }
    
    private static class NormalizationConfig {
        float mean;
        float std;
    }
    
    private static class ThresholdConfig {
        float threshold;
        float mean;
        float std;
        float k;
    }
}

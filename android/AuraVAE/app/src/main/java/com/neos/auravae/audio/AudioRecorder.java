package com.neos.auravae.audio;

import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.util.Log;

import java.util.ArrayList;
import java.util.List;

/**
 * Audio Recorder using AudioRecord API
 * 
 * Records PCM 16-bit audio at 16kHz mono.
 * Designed for vehicle acoustic anomaly detection.
 */
public class AudioRecorder {
    
    private static final String TAG = "AudioRecorder";
    
    // Audio configuration - MUST match Python preprocessing
    public static final int SAMPLE_RATE = 16000;  // 16 kHz
    public static final int CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO;
    public static final int AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT;
    
    // Buffer configuration
    private static final int BUFFER_SIZE_MULTIPLIER = 2;
    
    private AudioRecord audioRecord;
    private boolean isRecording = false;
    private int bufferSize;
    
    // Recorded audio data
    private List<short[]> recordedChunks;
    private Thread recordingThread;
    
    public AudioRecorder() {
        // Calculate minimum buffer size
        bufferSize = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT);
        if (bufferSize == AudioRecord.ERROR || bufferSize == AudioRecord.ERROR_BAD_VALUE) {
            bufferSize = SAMPLE_RATE * 2;  // Fallback to 1 second buffer
        }
        bufferSize *= BUFFER_SIZE_MULTIPLIER;
        
        Log.d(TAG, "Buffer size: " + bufferSize);
    }
    
    /**
     * Start audio recording.
     * Recording continues until stopRecording() is called.
     */
    public void startRecording() throws IllegalStateException {
        if (isRecording) {
            throw new IllegalStateException("Already recording");
        }
        
        // Create AudioRecord instance
        audioRecord = new AudioRecord(
                MediaRecorder.AudioSource.MIC,
                SAMPLE_RATE,
                CHANNEL_CONFIG,
                AUDIO_FORMAT,
                bufferSize
        );
        
        if (audioRecord.getState() != AudioRecord.STATE_INITIALIZED) {
            throw new IllegalStateException("AudioRecord initialization failed");
        }
        
        recordedChunks = new ArrayList<>();
        isRecording = true;
        
        // Start recording
        audioRecord.startRecording();
        
        // Start background thread to read audio data
        recordingThread = new Thread(this::recordAudio, "AudioRecorderThread");
        recordingThread.start();
        
        Log.d(TAG, "Recording started");
    }
    
    /**
     * Background recording loop.
     */
    private void recordAudio() {
        short[] buffer = new short[bufferSize / 2];  // 2 bytes per short
        
        while (isRecording) {
            int read = audioRecord.read(buffer, 0, buffer.length);
            
            if (read > 0) {
                // Copy buffer and store
                short[] chunk = new short[read];
                System.arraycopy(buffer, 0, chunk, 0, read);
                synchronized (recordedChunks) {
                    recordedChunks.add(chunk);
                }
            } else if (read < 0) {
                Log.e(TAG, "Audio read error: " + read);
                break;
            }
        }
    }
    
    /**
     * Stop recording and return the recorded audio data.
     * 
     * @return Recorded audio samples (16-bit PCM)
     */
    public short[] stopRecording() {
        if (!isRecording) {
            return null;
        }
        
        isRecording = false;
        
        // Wait for recording thread to finish
        if (recordingThread != null) {
            try {
                recordingThread.join(1000);
            } catch (InterruptedException e) {
                Log.e(TAG, "Interrupted while waiting for recording thread");
            }
        }
        
        // Stop and release AudioRecord
        if (audioRecord != null) {
            try {
                audioRecord.stop();
                audioRecord.release();
            } catch (Exception e) {
                Log.e(TAG, "Error stopping AudioRecord: " + e.getMessage());
            }
            audioRecord = null;
        }
        
        // Combine all chunks into single array
        short[] audioData = combineChunks();
        
        Log.d(TAG, "Recording stopped. Total samples: " + (audioData != null ? audioData.length : 0));
        
        return audioData;
    }
    
    /**
     * Combine recorded chunks into a single array.
     */
    private short[] combineChunks() {
        if (recordedChunks == null || recordedChunks.isEmpty()) {
            return null;
        }
        
        // Calculate total length
        int totalLength = 0;
        synchronized (recordedChunks) {
            for (short[] chunk : recordedChunks) {
                totalLength += chunk.length;
            }
        }
        
        // Combine chunks
        short[] combined = new short[totalLength];
        int offset = 0;
        synchronized (recordedChunks) {
            for (short[] chunk : recordedChunks) {
                System.arraycopy(chunk, 0, combined, offset, chunk.length);
                offset += chunk.length;
            }
        }
        
        return combined;
    }
    
    /**
     * Check if currently recording.
     */
    public boolean isRecording() {
        return isRecording;
    }
    
    /**
     * Get the recording duration in seconds.
     */
    public float getRecordingDuration() {
        if (recordedChunks == null) {
            return 0;
        }
        
        int totalSamples = 0;
        synchronized (recordedChunks) {
            for (short[] chunk : recordedChunks) {
                totalSamples += chunk.length;
            }
        }
        
        return (float) totalSamples / SAMPLE_RATE;
    }
}

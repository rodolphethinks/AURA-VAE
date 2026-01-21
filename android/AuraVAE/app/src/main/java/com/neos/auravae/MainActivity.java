package com.neos.auravae;

import android.Manifest;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.view.View;
import android.widget.Button;
import android.widget.FrameLayout;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.cardview.widget.CardView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.neos.auravae.audio.AudioRecorder;
import com.neos.auravae.ml.AnomalyDetector;
import com.neos.auravae.ml.DetectionResult;

import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * AURA-VAE Main Activity
 * 
 * Handles:
 * - Audio recording using AudioRecord API
 * - On-device VAE inference for anomaly detection
 * - UI updates for recording and analysis status
 * 
 * Designed for in-car use with large touch targets and clear indicators.
 */
public class MainActivity extends AppCompatActivity {
    
    private static final String TAG = "MainActivity";
    private static final int PERMISSION_REQUEST_RECORD_AUDIO = 200;
    
    // UI Components
    private Button btnStartRecording;
    private Button btnStopAnalyze;
    private TextView txtDuration;
    private TextView txtRecordingStatus;
    private TextView txtAnalysisStatus;
    private TextView txtResultValue;
    private TextView txtAnomalyScore;
    private TextView txtSegmentsAnalyzed;
    private TextView txtLoadingMessage;
    private View recordingIndicator;
    private View analysisIndicator;
    private CardView resultCard;
    private FrameLayout loadingOverlay;
    private ProgressBar confidenceBar;
    
    // Audio recording
    private AudioRecorder audioRecorder;
    private boolean isRecording = false;
    private long recordingStartTime = 0;
    
    // ML inference
    private AnomalyDetector anomalyDetector;
    private ExecutorService executorService;
    
    // UI update handler
    private final Handler uiHandler = new Handler(Looper.getMainLooper());
    private Runnable durationUpdater;
    
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        
        // Initialize views
        initViews();
        
        // Initialize executor for background tasks
        executorService = Executors.newSingleThreadExecutor();
        
        // Initialize audio recorder
        audioRecorder = new AudioRecorder();
        
        // Load ML model
        loadModel();
        
        // Setup button listeners
        setupListeners();
    }
    
    private void initViews() {
        btnStartRecording = findViewById(R.id.btnStartRecording);
        btnStopAnalyze = findViewById(R.id.btnStopAnalyze);
        txtDuration = findViewById(R.id.txtDuration);
        txtRecordingStatus = findViewById(R.id.txtRecordingStatus);
        txtAnalysisStatus = findViewById(R.id.txtAnalysisStatus);
        txtResultValue = findViewById(R.id.txtResultValue);
        txtAnomalyScore = findViewById(R.id.txtAnomalyScore);
        txtSegmentsAnalyzed = findViewById(R.id.txtSegmentsAnalyzed);
        txtLoadingMessage = findViewById(R.id.txtLoadingMessage);
        recordingIndicator = findViewById(R.id.recordingIndicator);
        analysisIndicator = findViewById(R.id.analysisIndicator);
        resultCard = findViewById(R.id.resultCard);
        loadingOverlay = findViewById(R.id.loadingOverlay);
        confidenceBar = findViewById(R.id.confidenceBar);
    }
    
    private void setupListeners() {
        btnStartRecording.setOnClickListener(v -> startRecording());
        btnStopAnalyze.setOnClickListener(v -> stopAndAnalyze());
    }
    
    private void loadModel() {
        showLoading(getString(R.string.loading_model));
        
        executorService.execute(() -> {
            try {
                anomalyDetector = new AnomalyDetector(this);
                
                uiHandler.post(() -> {
                    hideLoading();
                    Toast.makeText(this, "Model loaded successfully", Toast.LENGTH_SHORT).show();
                });
            } catch (Exception e) {
                e.printStackTrace();
                uiHandler.post(() -> {
                    hideLoading();
                    Toast.makeText(this, getString(R.string.error_model) + ": " + e.getMessage(), 
                            Toast.LENGTH_LONG).show();
                });
            }
        });
    }
    
    private void startRecording() {
        // Check permission
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.RECORD_AUDIO},
                    PERMISSION_REQUEST_RECORD_AUDIO);
            return;
        }
        
        // Start recording
        try {
            audioRecorder.startRecording();
            isRecording = true;
            recordingStartTime = System.currentTimeMillis();
            
            // Update UI
            updateRecordingUI(true);
            
            // Start duration updater
            startDurationUpdater();
            
        } catch (Exception e) {
            e.printStackTrace();
            Toast.makeText(this, getString(R.string.error_recording) + ": " + e.getMessage(),
                    Toast.LENGTH_LONG).show();
        }
    }
    
    private void stopAndAnalyze() {
        if (!isRecording) return;
        
        // Stop recording
        short[] audioData = audioRecorder.stopRecording();
        isRecording = false;
        
        // Stop duration updater
        stopDurationUpdater();
        
        // Update UI
        updateRecordingUI(false);
        
        // Analyze audio
        if (audioData != null && audioData.length > 0) {
            analyzeAudio(audioData);
        } else {
            Toast.makeText(this, getString(R.string.error_recording), Toast.LENGTH_SHORT).show();
        }
    }
    
    private void analyzeAudio(short[] audioData) {
        showLoading(getString(R.string.loading_analyzing));
        updateAnalysisStatus(true);
        
        executorService.execute(() -> {
            try {
                // Run anomaly detection
                DetectionResult result = anomalyDetector.detectAnomaly(audioData);
                
                uiHandler.post(() -> {
                    hideLoading();
                    updateAnalysisStatus(false);
                    displayResult(result);
                });
                
            } catch (Exception e) {
                e.printStackTrace();
                uiHandler.post(() -> {
                    hideLoading();
                    updateAnalysisStatus(false);
                    Toast.makeText(this, getString(R.string.error_analysis) + ": " + e.getMessage(),
                            Toast.LENGTH_LONG).show();
                });
            }
        });
    }
    
    private void displayResult(DetectionResult result) {
        // Show result card
        resultCard.setVisibility(View.VISIBLE);
        
        // Set result text and color
        if (result.isAnomaly()) {
            txtResultValue.setText(R.string.result_anomaly);
            txtResultValue.setTextColor(ContextCompat.getColor(this, R.color.result_anomaly));
        } else {
            txtResultValue.setText(R.string.result_normal);
            txtResultValue.setTextColor(ContextCompat.getColor(this, R.color.result_normal));
        }
        
        // Set anomaly score
        txtAnomalyScore.setText(String.format(Locale.US, 
                getString(R.string.score_format), 
                result.getAnomalyScore(), 
                result.getThreshold()));
        
        // Set confidence bar (normalized score vs threshold)
        int confidence = (int) Math.min(100, (result.getAnomalyScore() / result.getThreshold()) * 50);
        confidenceBar.setProgress(confidence);
        
        // Show segments analyzed
        txtSegmentsAnalyzed.setVisibility(View.VISIBLE);
        txtSegmentsAnalyzed.setText(String.format(Locale.US,
                getString(R.string.segments_format),
                result.getSegmentsAnalyzed()));
        
        // Update analysis indicator
        analysisIndicator.setBackgroundResource(R.drawable.status_indicator_success);
        txtAnalysisStatus.setText(R.string.analysis_complete);
    }
    
    private void updateRecordingUI(boolean recording) {
        if (recording) {
            btnStartRecording.setEnabled(false);
            btnStopAnalyze.setEnabled(true);
            recordingIndicator.setBackgroundResource(R.drawable.status_indicator_recording);
            txtRecordingStatus.setText(R.string.status_recording);
            resultCard.setVisibility(View.INVISIBLE);
            txtSegmentsAnalyzed.setVisibility(View.GONE);
        } else {
            btnStartRecording.setEnabled(true);
            btnStopAnalyze.setEnabled(false);
            recordingIndicator.setBackgroundResource(R.drawable.status_indicator_inactive);
            txtRecordingStatus.setText(R.string.status_stopped);
        }
    }
    
    private void updateAnalysisStatus(boolean analyzing) {
        if (analyzing) {
            analysisIndicator.setBackgroundResource(R.drawable.status_indicator_processing);
            txtAnalysisStatus.setText(R.string.analysis_processing);
        } else {
            analysisIndicator.setBackgroundResource(R.drawable.status_indicator_inactive);
            txtAnalysisStatus.setText(R.string.analysis_idle);
        }
    }
    
    private void startDurationUpdater() {
        durationUpdater = new Runnable() {
            @Override
            public void run() {
                if (isRecording) {
                    long elapsed = System.currentTimeMillis() - recordingStartTime;
                    int seconds = (int) (elapsed / 1000);
                    int minutes = seconds / 60;
                    seconds = seconds % 60;
                    txtDuration.setText(String.format(Locale.US, 
                            getString(R.string.duration_format), minutes, seconds));
                    uiHandler.postDelayed(this, 500);
                }
            }
        };
        uiHandler.post(durationUpdater);
    }
    
    private void stopDurationUpdater() {
        if (durationUpdater != null) {
            uiHandler.removeCallbacks(durationUpdater);
        }
    }
    
    private void showLoading(String message) {
        txtLoadingMessage.setText(message);
        loadingOverlay.setVisibility(View.VISIBLE);
    }
    
    private void hideLoading() {
        loadingOverlay.setVisibility(View.GONE);
    }
    
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        
        if (requestCode == PERMISSION_REQUEST_RECORD_AUDIO) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                startRecording();
            } else {
                Toast.makeText(this, getString(R.string.error_permission), Toast.LENGTH_LONG).show();
            }
        }
    }
    
    @Override
    protected void onDestroy() {
        super.onDestroy();
        
        if (isRecording) {
            audioRecorder.stopRecording();
        }
        
        if (anomalyDetector != null) {
            anomalyDetector.close();
        }
        
        if (executorService != null) {
            executorService.shutdown();
        }
        
        stopDurationUpdater();
    }
}

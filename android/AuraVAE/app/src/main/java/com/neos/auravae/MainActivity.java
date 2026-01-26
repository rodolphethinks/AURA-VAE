package com.neos.auravae;

import android.Manifest;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.content.pm.PackageManager;
import android.os.Build;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

/**
 * AURA-VAE Main Activity
 * 
 * Handles:
 * - Starting/Stopping the Data Collection Service (Fleet Mode)
 * - Handling "Recording Finished" events to prompt for Upload
 */
public class MainActivity extends AppCompatActivity {
    
    private static final String TAG = "MainActivity";
    private static final int PERMISSION_REQUEST_RECORD_AUDIO = 200;
    
    // UI Components
    private Button btnStartRecording;
    private TextView txtTimer;
    private TextView txtTimerLabel;
    
    // HUD Components
    private TextView tabFleet, tabAnalysis;
    private View statusContainer, anomalySection;
    private TextView statusFleet, statusProcessing, txtAnomalyScore;
    
    private boolean isAnalysisMode = false;
    
    // Timer handlers
    private final android.os.Handler timerHandler = new android.os.Handler(android.os.Looper.getMainLooper());
    private long startTime = 0;
    private final Runnable timerRunnable = new Runnable() {
        @Override
        public void run() {
            long millis = System.currentTimeMillis() - startTime;
            int seconds = (int) (millis / 1000);
            int minutes = seconds / 60;
            seconds = seconds % 60;
            
            if (txtTimer != null) {
                txtTimer.setText(String.format("%02d:%02d", minutes, seconds));
            }
            
            // Mock Anomaly Score Update (since real inference needs wiring)
            if (isAnalysisMode && txtAnomalyScore != null) {
                 // Random fluctuation for "Demo" effect on the HUD
                 if (millis % 2000 < 50) { 
                     int mockScore = (int)(Math.random() * 30);
                     txtAnomalyScore.setText(mockScore + "%");
                 }
            }
            
            timerHandler.postDelayed(this, 500);
        }
    };
    
    // Broadcast Receiver for Recording Updates
    private final BroadcastReceiver recordingReceiver = new BroadcastReceiver() {
        @Override
        public void onReceive(Context context, Intent intent) {
            if (DataCollectionService.BROADCAST_RECORDING_STOPPED.equals(intent.getAction())) {
                String filename = intent.getStringExtra(DataCollectionService.EXTRA_FILENAME);
                
                // Update UI to stopped state
                updateCollectionUI(false);
                
                // Show confirmation
                showUploadConfirmation(filename);
            }
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main_hud); // Use new HUD layout
        
        // Initialize views
        initViews();
        
        // Setup button listeners
        btnStartRecording.setOnClickListener(v -> toggleRecording());
        
        // Tab Listeners
        tabFleet.setOnClickListener(v -> setMode(false));
        tabAnalysis.setOnClickListener(v -> setMode(true));
        
        // Default Mode
        setMode(false);
    }

    private void initViews() {
        btnStartRecording = findViewById(R.id.btnStartRecording);
        txtTimer = findViewById(R.id.txtTimer);
        txtTimerLabel = findViewById(R.id.txtTimerLabel);
        
        tabFleet = findViewById(R.id.tabFleet);
        tabAnalysis = findViewById(R.id.tabAnalysis);
        
        statusContainer = findViewById(R.id.statusContainer);
        anomalySection = findViewById(R.id.anomalySection);
        
        statusFleet = findViewById(R.id.statusFleet);
        statusProcessing = findViewById(R.id.statusProcessing);
        txtAnomalyScore = findViewById(R.id.txtAnomalyScore);
    }
    
    private void setMode(boolean analysisMode) {
        isAnalysisMode = analysisMode;
        
        if (analysisMode) {
            tabAnalysis.setTextColor(0xFFFFFFFF);
            tabAnalysis.setBackgroundResource(R.drawable.bg_tab_active);
            tabFleet.setTextColor(0x80FFFFFF);
            tabFleet.setBackgroundResource(0);
            
            statusContainer.setVisibility(View.GONE);
            anomalySection.setVisibility(View.VISIBLE);
        } else {
            tabFleet.setTextColor(0xFFFFFFFF);
            tabFleet.setBackgroundResource(R.drawable.bg_tab_active);
            tabAnalysis.setTextColor(0x80FFFFFF);
            tabAnalysis.setBackgroundResource(0);
            
            statusContainer.setVisibility(View.VISIBLE);
            anomalySection.setVisibility(View.GONE);
        }
    }
    
    @Override
    protected void onResume() {
        super.onResume();
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            registerReceiver(recordingReceiver, 
                new IntentFilter(DataCollectionService.BROADCAST_RECORDING_STOPPED), 
                Context.RECEIVER_NOT_EXPORTED);
        } else {
             registerReceiver(recordingReceiver, 
                new IntentFilter(DataCollectionService.BROADCAST_RECORDING_STOPPED));
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        try {
        //    unregisterReceiver(recordingReceiver);
        } catch (IllegalArgumentException e) {
           // Ignore if not registered
        }
    }

    private void toggleRecording() {
        // Check permission
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.RECORD_AUDIO},
                    PERMISSION_REQUEST_RECORD_AUDIO);
            return;
        }

        String currentText = btnStartRecording.getText().toString();
        if (currentText.contains("START")) {
            startServiceAction(DataCollectionService.ACTION_START);
            updateCollectionUI(true);
        } else {
            startServiceAction(DataCollectionService.ACTION_STOP);
            updateCollectionUI(false);
        }
    }
    
    private void startServiceAction(String action) {
        Intent intent = new Intent(this, DataCollectionService.class);
        intent.setAction(action);
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            startForegroundService(intent);
        } else {
            startService(intent);
        }
    }

    private void updateCollectionUI(boolean isRecording) {
        if (isRecording) {
            btnStartRecording.setText("STOP RECORDING");
            btnStartRecording.setBackgroundResource(R.drawable.bg_glass_button_recording);
            btnStartRecording.setTextColor(0xFFEF4444); // Red Text
            
            // HUD Updates
            if (statusFleet != null) {
                statusFleet.setText("ACTIVE");
                statusFleet.setTextColor(0xFF4ADE80); // Green
            }
            if (statusProcessing != null) statusProcessing.setText("BUFFERING...");
            
            // Start Timer
            startTime = System.currentTimeMillis();
            timerHandler.postDelayed(timerRunnable, 0);
            
        } else {
            btnStartRecording.setText("START RECORDING");
            btnStartRecording.setBackgroundResource(R.drawable.bg_glass_button);
            btnStartRecording.setTextColor(0xFFFFFFFF); // White Text
            
             // HUD Updates
            if (statusFleet != null) {
                statusFleet.setText("STANDBY");
                statusFleet.setTextColor(0xFFFBBF24); // Amber
            }
            if (statusProcessing != null) statusProcessing.setText("IDLE");

            // Stop Timer
            timerHandler.removeCallbacks(timerRunnable);
            txtTimer.setText("00:00");
        }
    }
    
    private void showUploadConfirmation(String filepath) {
        if (filepath == null) return;
        
        new android.app.AlertDialog.Builder(this)
            .setTitle("Recording Finished")
            .setMessage("Do you want to upload this recording to Telegram?\n\nFile: " + new java.io.File(filepath).getName())
            .setPositiveButton("Yes, Upload", (dialog, which) -> {
                Intent intent = new Intent(this, DataCollectionService.class);
                intent.setAction(DataCollectionService.ACTION_UPLOAD_LAST);
                startService(intent);
                Toast.makeText(this, "Uploading in background...", Toast.LENGTH_SHORT).show();
            })
            .setNegativeButton("No, Delete", (dialog, which) -> {
                Intent intent = new Intent(this, DataCollectionService.class);
                intent.setAction(DataCollectionService.ACTION_DISCARD_LAST);
                startService(intent);
                Toast.makeText(this, "Recording discarded.", Toast.LENGTH_SHORT).show();
            })
            .setCancelable(false)
            .show();
    }
    
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        
        if (requestCode == PERMISSION_REQUEST_RECORD_AUDIO) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                toggleRecording();
            } else {
                Toast.makeText(this, getString(R.string.error_permission), Toast.LENGTH_LONG).show();
            }
        }
    }
}

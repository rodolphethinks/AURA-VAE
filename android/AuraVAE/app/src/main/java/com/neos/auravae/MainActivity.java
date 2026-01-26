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
    private TextView txtRecordingStatus;
    private TextView txtAnalysisStatus; // Added for feedback
    private TextView txtDuration;
    private View recordingIndicator;
    
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
            
            if (txtDuration != null) {
                txtDuration.setText(String.format("Duration: %02d:%02d", minutes, seconds));
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
        setContentView(R.layout.activity_main);
        
        // Initialize views
        initViews();
        
        // Setup button listeners
        btnStartRecording.setOnClickListener(v -> toggleRecording());
    }

    private void initViews() {
        btnStartRecording = findViewById(R.id.btnStartRecording);
        txtRecordingStatus = findViewById(R.id.txtRecordingStatus);
        txtAnalysisStatus = findViewById(R.id.txtAnalysisStatus); // Init view
        recordingIndicator = findViewById(R.id.recordingIndicator);
        txtDuration = findViewById(R.id.txtDuration);
        
        // Hide unused legacy views if they exist to avoid confusion
        View btnStopAnalyze = findViewById(R.id.btnStopAnalyze);
        if (btnStopAnalyze != null) btnStopAnalyze.setVisibility(View.GONE);
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
            btnStartRecording.setBackgroundTintList(ContextCompat.getColorStateList(this, R.color.button_stop));
            txtRecordingStatus.setText("Recording (Background Service)...");
            if (txtAnalysisStatus != null) txtAnalysisStatus.setText("Collecting Fleet Data..."); // Update status
            if (recordingIndicator != null) recordingIndicator.setVisibility(View.VISIBLE);
            
            // Start Timer
            startTime = System.currentTimeMillis();
            timerHandler.postDelayed(timerRunnable, 0);
            
        } else {
            btnStartRecording.setText("START RECORDING");
            btnStartRecording.setBackgroundTintList(ContextCompat.getColorStateList(this, R.color.button_start)); // Default
            txtRecordingStatus.setText("Ready to Record");
            if (txtAnalysisStatus != null) txtAnalysisStatus.setText("Idle (Data Collection Mode)"); // Update status
             if (recordingIndicator != null) recordingIndicator.setVisibility(View.GONE);
             
            // Stop Timer
            timerHandler.removeCallbacks(timerRunnable);
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

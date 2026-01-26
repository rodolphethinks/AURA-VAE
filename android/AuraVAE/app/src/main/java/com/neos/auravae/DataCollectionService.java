package com.neos.auravae;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Context;
import android.content.Intent;
import android.media.MediaRecorder;
import android.os.Build;
import android.os.Handler;
import android.os.IBinder;
import android.os.Looper;
import android.util.Log;

import androidx.annotation.Nullable;
import androidx.core.app.NotificationCompat;

import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

/**
 * DataCollectionService
 * 
 * Runs in the foreground to collect audio data for model training.
 * Implements the "Fleet" strategy:
 * 1. Auto-starts on boot
 * 2. Records in chunks (to avoid huge files)
 * 3. Saves to local storage (Future: Upload to Cloud)
 */
public class DataCollectionService extends Service {
    
    public static final String ACTION_START = "com.neos.auravae.action.START";
    public static final String ACTION_STOP = "com.neos.auravae.action.STOP";
    public static final String ACTION_UPLOAD_LAST = "com.neos.auravae.action.UPLOAD";
    public static final String ACTION_DISCARD_LAST = "com.neos.auravae.action.DISCARD";
    public static final String EXTRA_FILENAME = "com.neos.auravae.extra.FILENAME";
    public static final String BROADCAST_RECORDING_STOPPED = "com.neos.auravae.broadcast.STOPPED";

    private static final String TAG = "DataCollectionService";
    private static final String CHANNEL_ID = "AuraDataCollectionChannel";
    private static final int NOTIFICATION_ID = 1001;
    
    // Config: 5 minute chunks (Auto-mode only)
    private static final long RECORDING_CHUNK_DURATION_MS = 5 * 60 * 1000;
    private boolean isManualMode = false;

    // Telegram Configuration
    private static final String BOT_TOKEN = BuildConfig.TELEGRAM_BOT_TOKEN; 
    private static final String CHAT_ID = BuildConfig.TELEGRAM_CHAT_ID;
    
    private MediaRecorder mediaRecorder;
    private Handler handler;
    public boolean isRecording = false;
    private File currentFile;
    private File lastRecordedFile;
    private OkHttpClient httpClient;

    @Override
    public void onCreate() {
        super.onCreate();
        createNotificationChannel();
        handler = new Handler(Looper.getMainLooper());
        httpClient = new OkHttpClient();
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        String action = intent != null ? intent.getAction() : null;

        if (ACTION_START.equals(action)) {
            Log.i(TAG, "Manual Start Requested");
            isManualMode = true;
            startForeground(NOTIFICATION_ID, createNotification("Recording Audio..."));
            if (!isRecording) startRecordingChunk();
            
        } else if (ACTION_STOP.equals(action)) {
            Log.i(TAG, "Manual Stop Requested");
            if (isRecording) {
                stopRecording();
                // Notify UI
                Intent broadcast = new Intent(BROADCAST_RECORDING_STOPPED);
                if (lastRecordedFile != null) {
                    broadcast.putExtra(EXTRA_FILENAME, lastRecordedFile.getAbsolutePath());
                }
                sendBroadcast(broadcast);
                stopForeground(true); // Allow service to be killed or go background
            }
            
        } else if (ACTION_UPLOAD_LAST.equals(action)) {
            Log.i(TAG, "Upload Requested for last file");
            if (lastRecordedFile != null && lastRecordedFile.exists()) {
                uploadToCloud(lastRecordedFile);
                lastRecordedFile = null; // Clear ref
            }
            
        } else if (ACTION_DISCARD_LAST.equals(action)) {
            Log.i(TAG, "Discard Requested");
            if (lastRecordedFile != null && lastRecordedFile.exists()) {
                lastRecordedFile.delete();
                lastRecordedFile = null;
            }
            
        } else {
            // Default/Auto-Boot behavior (Fleet Mode)
            Log.i(TAG, "Auto-Start (Fleet Mode)");
            isManualMode = false;
            
            // Start Foreground with Microphone Type specifically for Android 11+
            Notification notification = createNotification("Collecting Fleet Data...");
            if (Build.VERSION.SDK_INT >= 29) { // Build.VERSION_CODES.Q
                startForeground(NOTIFICATION_ID, notification, android.content.pm.ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE);
            } else {
                startForeground(NOTIFICATION_ID, notification);
            }
            
            if (!isRecording) startRecordingChunk();
        }

        return START_NOT_STICKY;
    }

    private void startRecordingChunk() {
        if (isRecording) {
            stopRecording();
        }

        try {
            currentFile = createOutputFile();
            Log.d(TAG, "Starting recording: " + currentFile.getAbsolutePath());

            mediaRecorder = new MediaRecorder();
            // Attempt 1: Standard AAC (Voice Config)
            mediaRecorder.setAudioSource(MediaRecorder.AudioSource.VOICE_RECOGNITION); 
            mediaRecorder.setOutputFormat(MediaRecorder.OutputFormat.MPEG_4);
            mediaRecorder.setAudioEncoder(MediaRecorder.AudioEncoder.AAC);
            mediaRecorder.setAudioEncodingBitRate(64000); // 64kbps (Speech optimized)
            mediaRecorder.setAudioSamplingRate(16000); // 16kHz (Standard wideband speech)
            mediaRecorder.setOutputFile(currentFile.getAbsolutePath());

            try {
                mediaRecorder.prepare();
                mediaRecorder.start();
                isRecording = true;
                Log.d(TAG, "Recording started successfully (VOICE_RECOGNITION / AAC / 16kHz)");
            } catch (Exception e1) {
                Log.e(TAG, "Attempt 1 failed. Retrying with AMR_WB.", e1);
                releaseMediaRecorder();
                
                // Attempt 2: AMR_WB (Wideband Speech)
                try {
                     if (currentFile.exists()) currentFile.delete(); currentFile = createOutputFile();
                     mediaRecorder = new MediaRecorder();
                     mediaRecorder.setAudioSource(MediaRecorder.AudioSource.MIC); // Try generic MIC
                     mediaRecorder.setOutputFormat(MediaRecorder.OutputFormat.THREE_GPP);
                     mediaRecorder.setAudioEncoder(MediaRecorder.AudioEncoder.AMR_WB);
                     mediaRecorder.setOutputFile(currentFile.getAbsolutePath());
                     mediaRecorder.prepare();
                     mediaRecorder.start();
                     isRecording = true;
                     Log.w(TAG, "Recording started (Fallback: AMR_WB)");
                } catch (Exception e2) {
                     Log.e(TAG, "Attempt 2 failed. Retrying with AMR_NB (Lowest Quality).", e2);
                     releaseMediaRecorder();

                     // Attempt 3: AMR_NB (Narrowband - Old School GSM)
                     try {
                         if (currentFile.exists()) currentFile.delete(); currentFile = createOutputFile();
                         mediaRecorder = new MediaRecorder();
                         mediaRecorder.setAudioSource(MediaRecorder.AudioSource.DEFAULT); // Try DEFAULT
                         mediaRecorder.setOutputFormat(MediaRecorder.OutputFormat.THREE_GPP);
                         mediaRecorder.setAudioEncoder(MediaRecorder.AudioEncoder.AMR_NB);
                         mediaRecorder.setOutputFile(currentFile.getAbsolutePath());
                         mediaRecorder.prepare();
                         mediaRecorder.start();
                         isRecording = true;
                         Log.w(TAG, "Recording started (Fallback: AMR_NB)");
                     } catch (Exception e3) {
                         Log.e(TAG, "All recording attempts failed.", e3);
                         isRecording = false;
                         throw e3;
                     }
                }
            }

            // Only schedule rotation in Auto/Fleet mode
            if (!isManualMode) {
                handler.postDelayed(this::rotateFile, RECORDING_CHUNK_DURATION_MS);
            }

        } catch (Exception e) {
            Log.e(TAG, "Error starting recording", e);
            handler.postDelayed(this::startRecordingChunk, 10000);
        }
    }

    private void rotateFile() {
        if (isManualMode) return; // Don't auto-rotate in manual mode
        
        Log.d(TAG, "Rotating file (Fleet Mode)...");
        stopRecording();
        if (lastRecordedFile != null) {
            uploadToCloud(lastRecordedFile);
        }
        startRecordingChunk();
    }

    private void releaseMediaRecorder() {
        if (mediaRecorder != null) {
            mediaRecorder.reset();
            mediaRecorder.release();
            mediaRecorder = null;
        }
    }

    private void stopRecording() {
        if (mediaRecorder != null) {
            try {
                mediaRecorder.stop();
            } catch (RuntimeException e) {
                Log.e(TAG, "Error stopping recorder", e);
            }
            releaseMediaRecorder();
            isRecording = false;
            lastRecordedFile = currentFile; // Store reference for upload/discard
            if (isManualMode) {
                handler.removeCallbacksAndMessages(null); // Clear any rotation tasks
            }
        }
    }

    // Upload to Telegram
    private void uploadToCloud(File file) {
        if (file == null || !file.exists()) return;
        
        if (BOT_TOKEN.contains("YOUR_BOT_TOKEN")) {
            Log.e(TAG, "Cannot upload: BOT_TOKEN not set in DataCollectionService.java");
            return;
        }

        Log.i(TAG, "Uploading file to Telegram: " + file.getName());
        
        // Run in background thread
        new Thread(() -> {
            try {
                RequestBody requestBody = new MultipartBody.Builder()
                        .setType(MultipartBody.FORM)
                        .addFormDataPart("chat_id", CHAT_ID)
                        .addFormDataPart("caption", "Fleet Audio: " + file.getName())
                        .addFormDataPart("audio", file.getName(),
                                RequestBody.create(file, MediaType.parse("audio/mp4")))
                        .build();

                Request request = new Request.Builder()
                        .url("https://api.telegram.org/bot" + BOT_TOKEN + "/sendAudio")
                        .post(requestBody)
                        .build();

                try (Response response = httpClient.newCall(request).execute()) {
                    if (response.isSuccessful()) {
                        Log.i(TAG, "Upload successful: " + response.body().string());
                        // Delete local file to save space
                        if (file.delete()) {
                            Log.d(TAG, "Local file deleted: " + file.getName());
                        }
                    } else {
                        Log.e(TAG, "Upload failed: " + response.code() + " " + response.message());
                        Log.e(TAG, "Response: " + response.body().string());
                    }
                }
            } catch (Exception e) {
                Log.e(TAG, "Error uploading to Telegram", e);
            }
        }).start();
    }

    private File createOutputFile() {
        String timestamp = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(new Date());
        String filename = "Aura_Fleet_" + timestamp + ".m4a";
        File dir = new File(getExternalFilesDir(null), "fleet_recordings");
        if (!dir.exists()) {
            dir.mkdirs();
        }
        return new File(dir, filename);
    }

    private Notification createNotification(String text) {
        Intent notificationIntent = new Intent(this, MainActivity.class);
        PendingIntent pendingIntent = PendingIntent.getActivity(this, 0, notificationIntent, PendingIntent.FLAG_IMMUTABLE);

        return new NotificationCompat.Builder(this, CHANNEL_ID)
                .setContentTitle("AURA-VAE Data Collection")
                .setContentText(text)
                .setSmallIcon(android.R.drawable.ic_btn_speak_now) // Default icon, replace with app icon
                .setContentIntent(pendingIntent)
                .build();
    }

    private void createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            NotificationChannel serviceChannel = new NotificationChannel(
                    CHANNEL_ID,
                    "Aura Collection Channel",
                    NotificationManager.IMPORTANCE_DEFAULT
            );
            NotificationManager manager = getSystemService(NotificationManager.class);
            if (manager != null) {
                manager.createNotificationChannel(serviceChannel);
            }
        }
    }

    @Override
    public void onDestroy() {
        stopRecording();
        handler.removeCallbacksAndMessages(null);
        super.onDestroy();
    }

    @Nullable
    @Override
    public IBinder onBind(Intent intent) {
        return null; // pure start service
    }
}

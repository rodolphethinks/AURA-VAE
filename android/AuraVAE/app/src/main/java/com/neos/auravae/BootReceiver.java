package com.neos.auravae;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.os.Build;
import android.util.Log;

/**
 * BootReceiver
 * 
 * Auto-starts the DataCollectionService when the car/device boots up.
 * This ensures data collection happens automatically without user intervention.
 */
public class BootReceiver extends BroadcastReceiver {
    private static final String TAG = "BootReceiver";

    @Override
    public void onReceive(Context context, Intent intent) {
        if (intent == null || intent.getAction() == null) return;

        String action = intent.getAction();
        Log.d(TAG, "Received action: " + action);

        if (Intent.ACTION_BOOT_COMPLETED.equals(action) || 
            "android.intent.action.QUICKBOOT_POWERON".equals(action) ||
            "android.intent.action.LOCKED_BOOT_COMPLETED".equals(action)) {
            
            Log.i(TAG, "Car powered on / Boot completed. Starting DataCollectionService...");
            
            Intent serviceIntent = new Intent(context, DataCollectionService.class);
            
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                context.startForegroundService(serviceIntent);
            } else {
                context.startService(serviceIntent);
            }
        }
    }
}

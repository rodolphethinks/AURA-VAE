package com.neos.auravae.audio;

import android.media.MediaCodec;
import android.media.MediaExtractor;
import android.media.MediaFormat;
import android.util.Log;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Utility to decode audio files (M4A/AAC/MP3) to raw PCM 16-bit short array.
 * Useful for processing recorded files with the VAE model.
 */
public class AudioFileDecoder {

    private static final String TAG = "AudioFileDecoder";
    private static final long TIMEOUT_US = 5000;

    /**
     * Decode audio file to PCM 16-bit short array.
     * 
     * @param audioPath Path to audio file
     * @return short[] containing raw audio samples (mono usually)
     * @throws IOException If decoding fails
     */
    public static short[] decode(String audioPath) throws IOException {
        MediaExtractor extractor = new MediaExtractor();
        extractor.setDataSource(audioPath);

        int audioTrackIndex = -1;
        MediaFormat format = null;

        // Find audio track
        for (int i = 0; i < extractor.getTrackCount(); i++) {
            MediaFormat trackFormat = extractor.getTrackFormat(i);
            String mime = trackFormat.getString(MediaFormat.KEY_MIME);
            if (mime.startsWith("audio/")) {
                audioTrackIndex = i;
                format = trackFormat;
                break;
            }
        }

        if (audioTrackIndex == -1) {
            extractor.release();
            throw new IOException("No audio track found in " + audioPath);
        }

        extractor.selectTrack(audioTrackIndex);
        String mime = format.getString(MediaFormat.KEY_MIME);
        MediaCodec codec = MediaCodec.createDecoderByType(mime);
        
        Log.d(TAG, "Decoding " + audioPath + " (" + mime + ")");

        codec.configure(format, null, null, 0);
        codec.start();

        ByteArrayOutputStream outputBuffer = new ByteArrayOutputStream();
        MediaCodec.BufferInfo info = new MediaCodec.BufferInfo();
        boolean inputDone = false;
        boolean outputDone = false;

        while (!outputDone) {
            // Feed input
            if (!inputDone) {
                int inputBufferId = codec.dequeueInputBuffer(TIMEOUT_US);
                if (inputBufferId >= 0) {
                    ByteBuffer inputBuffer = codec.getInputBuffer(inputBufferId);
                    int sampleSize = extractor.readSampleData(inputBuffer, 0);
                    if (sampleSize < 0) {
                        codec.queueInputBuffer(inputBufferId, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM);
                        inputDone = true;
                    } else {
                        codec.queueInputBuffer(inputBufferId, 0, sampleSize, extractor.getSampleTime(), 0);
                        extractor.advance();
                    }
                }
            }

            // Read output
            int outputBufferId = codec.dequeueOutputBuffer(info, TIMEOUT_US);
            if (outputBufferId >= 0) {
                if ((info.flags & MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0) {
                    outputDone = true;
                }
                if (info.size > 0) {
                    ByteBuffer buffer = codec.getOutputBuffer(outputBufferId);
                    byte[] chunk = new byte[info.size];
                    buffer.get(chunk);
                    outputBuffer.write(chunk);
                }
                codec.releaseOutputBuffer(outputBufferId, false);
            } else if (outputBufferId == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                // Format changed, not critical here since we just want raw bytes
                MediaFormat newFormat = codec.getOutputFormat();
                Log.d(TAG, "Output format changed: " + newFormat);
            }
        }

        codec.stop();
        codec.release();
        extractor.release();

        byte[] rawBytes = outputBuffer.toByteArray();
        
        // Convert byte[] to short[] (Little Endian)
        short[] shorts = new short[rawBytes.length / 2];
        ByteBuffer.wrap(rawBytes).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer().get(shorts);
        
        Log.d(TAG, "Decoded " + shorts.length + " samples");
        return shorts;
    }
}

package com.neos.auravae.ml;

/**
 * Detection Result Container
 * 
 * Holds the result of anomaly detection including:
 * - Anomaly classification (normal/anomaly)
 * - Anomaly score
 * - Threshold used
 * - Number of segments analyzed
 */
public class DetectionResult {
    
    private final boolean isAnomaly;
    private final float anomalyScore;
    private final float threshold;
    private final int segmentsAnalyzed;
    private final float[] segmentScores;
    
    public DetectionResult(boolean isAnomaly, float anomalyScore, float threshold, 
                           int segmentsAnalyzed, float[] segmentScores) {
        this.isAnomaly = isAnomaly;
        this.anomalyScore = anomalyScore;
        this.threshold = threshold;
        this.segmentsAnalyzed = segmentsAnalyzed;
        this.segmentScores = segmentScores;
    }
    
    public boolean isAnomaly() {
        return isAnomaly;
    }
    
    public float getAnomalyScore() {
        return anomalyScore;
    }
    
    public float getThreshold() {
        return threshold;
    }
    
    public int getSegmentsAnalyzed() {
        return segmentsAnalyzed;
    }
    
    public float[] getSegmentScores() {
        return segmentScores;
    }
    
    /**
     * Get confidence level (0-1) based on score distance from threshold.
     */
    public float getConfidence() {
        if (isAnomaly) {
            // Higher score = higher confidence for anomaly
            return Math.min(1.0f, anomalyScore / threshold);
        } else {
            // Lower score = higher confidence for normal
            return 1.0f - (anomalyScore / threshold);
        }
    }
    
    @Override
    public String toString() {
        return String.format("DetectionResult{isAnomaly=%s, score=%.6f, threshold=%.6f, segments=%d}",
                isAnomaly, anomalyScore, threshold, segmentsAnalyzed);
    }
}

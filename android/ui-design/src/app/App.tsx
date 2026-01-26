import { useState, useEffect } from 'react';
import { motion } from 'motion/react';
import { Circle, Mic, Send, Play, Pause } from 'lucide-react';

export default function App() {
  const [isRecording, setIsRecording] = useState(false);
  const [timer, setTimer] = useState(14);
  const [anomalyScore, setAnomalyScore] = useState(23);
  const [activeTab, setActiveTab] = useState<'analytics' | 'collection'>('analytics');
  const [audioLevel, setAudioLevel] = useState(0);
  const [isSent, setIsSent] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [hasScore, setHasScore] = useState(true);
  const [hasRecording, setHasRecording] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackProgress, setPlaybackProgress] = useState(0);
  const [recordedDuration, setRecordedDuration] = useState(0);

  useEffect(() => {
    if (isRecording) {
      // Reset score when recording starts
      setHasScore(false);
      setAnomalyScore(0);
      
      const interval = setInterval(() => {
        setTimer((prev) => prev + 1);
        // Simulate audio level fluctuation when recording
        setAudioLevel(Math.random() * 100);
      }, 100);
      return () => clearInterval(interval);
    } else {
      setAudioLevel(0);
      
      // If we just stopped recording (timer > 0 means we were recording)
      if (timer > 0 && !hasRecording) {
        setHasRecording(true);
        setRecordedDuration(timer);
      }
      
      // Start processing when recording stops (only if we don't have a score)
      if (!hasScore && timer > 0) {
        setIsProcessing(true);
        
        // Processing animation for 3 seconds
        const processingTimeout = setTimeout(() => {
          setIsProcessing(false);
          // Generate random score between 15-35
          const newScore = Math.floor(Math.random() * 20) + 15;
          setAnomalyScore(newScore);
          setHasScore(true);
        }, 3000);
        
        return () => clearTimeout(processingTimeout);
      }
    }
  }, [isRecording, hasScore, timer, hasRecording]);

  // Playback progress simulation
  useEffect(() => {
    if (isPlaying) {
      const interval = setInterval(() => {
        setPlaybackProgress((prev) => {
          if (prev >= recordedDuration) {
            setIsPlaying(false);
            return 0;
          }
          return prev + 0.1;
        });
      }, 100);
      return () => clearInterval(interval);
    }
  }, [isPlaying, recordedDuration]);

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
  };

  return (
    <div className="w-full h-screen overflow-hidden dark">
      {/* Deep blurred abstract mesh gradient background */}
      <div className="absolute inset-0 bg-gradient-to-br from-[#0a0a0f] via-[#13131a] to-[#1a1a24]">
        {/* Animated mesh gradient layers for depth */}
        <motion.div
          className="absolute inset-0 opacity-40"
          animate={{
            background: [
              'radial-gradient(circle at 20% 30%, rgba(59, 130, 246, 0.15) 0%, transparent 50%)',
              'radial-gradient(circle at 80% 70%, rgba(139, 92, 246, 0.15) 0%, transparent 50%)',
              'radial-gradient(circle at 40% 60%, rgba(59, 130, 246, 0.15) 0%, transparent 50%)',
            ],
          }}
          transition={{ duration: 8, repeat: Infinity, ease: 'easeInOut' }}
        />
        <motion.div
          className="absolute inset-0 opacity-30"
          animate={{
            background: [
              'radial-gradient(circle at 80% 20%, rgba(139, 92, 246, 0.12) 0%, transparent 60%)',
              'radial-gradient(circle at 30% 80%, rgba(59, 130, 246, 0.12) 0%, transparent 60%)',
              'radial-gradient(circle at 60% 40%, rgba(139, 92, 246, 0.12) 0%, transparent 60%)',
            ],
          }}
          transition={{ duration: 10, repeat: Infinity, ease: 'easeInOut' }}
        />
      </div>

      {/* Main content container - 1800x720 centered */}
      <div className="relative flex items-center justify-center w-full h-full p-8">
        <div className="w-[1800px] h-[720px] flex gap-0 relative">
          
          {/* Left Panel - Controls */}
          <div className="flex-1 flex flex-col items-start justify-between p-16">
            {/* App Title */}
            <motion.div
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, ease: 'easeOut' }}
            >
              <h1 className="text-7xl tracking-wider text-white/95 mb-2" style={{ fontWeight: 300, letterSpacing: '0.15em' }}>
                AURA
              </h1>
              <div className="h-[2px] w-24 bg-gradient-to-r from-blue-400 to-transparent rounded-full" />
            </motion.div>

            {/* Frosted Glass Panel with Button and Timer */}
            <motion.div
              className="relative"
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.8, delay: 0.2 }}
            >
              {/* Glass panel background */}
              <div className="relative backdrop-blur-2xl bg-white/[0.03] border border-white/[0.15] rounded-[40px] p-12 shadow-2xl">
                {/* Subtle inner glow */}
                <div className="absolute inset-0 rounded-[40px] bg-gradient-to-br from-white/[0.05] to-transparent pointer-events-none" />
                
                {/* Primary Button */}
                <motion.button
                  onClick={() => setIsRecording(!isRecording)}
                  className="relative px-20 py-8 rounded-full overflow-hidden group"
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  transition={{ type: 'spring', stiffness: 400, damping: 17 }}
                >
                  {/* Button background with frosted glass */}
                  <div className={`absolute inset-0 backdrop-blur-2xl border transition-all duration-500 ${
                    isRecording 
                      ? 'bg-white/[0.03] border-red-400/30 shadow-[0_8px_32px_rgba(239,68,68,0.2)]' 
                      : 'bg-white/[0.03] border-white/[0.15] shadow-[0_8px_32px_rgba(255,255,255,0.05)]'
                  }`} />
                  
                  {/* Soft inner glow */}
                  <div className={`absolute inset-0 bg-gradient-to-br pointer-events-none ${
                    isRecording 
                      ? 'from-red-400/10 to-transparent' 
                      : 'from-white/[0.05] to-transparent'
                  }`} />
                  
                  {/* Animated shimmer effect */}
                  <motion.div
                    className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent"
                    animate={{ x: ['-200%', '200%'] }}
                    transition={{ duration: 3, repeat: Infinity, ease: 'linear' }}
                  />
                  
                  {/* Button text */}
                  <span className={`relative text-2xl tracking-[0.2em] uppercase transition-colors duration-500 ${
                    isRecording ? 'text-red-300' : 'text-white/90'
                  }`} style={{ fontWeight: 500 }}>
                    {isRecording ? 'STOP RECORDING' : 'START RECORDING'}
                  </span>
                </motion.button>

                {/* Timer Display */}
                <motion.div
                  className="mt-8 text-center"
                  animate={{ opacity: [0.7, 1, 0.7] }}
                  transition={{ duration: 2, repeat: Infinity, ease: 'easeInOut' }}
                >
                  <div className="text-6xl text-white/90 tracking-widest font-light tabular-nums">
                    {formatTime(timer)}
                  </div>
                  <div className="text-sm text-white/40 tracking-[0.3em] uppercase mt-2">
                    Elapsed Time
                  </div>
                </motion.div>

                {/* Audio Playback Section */}
                {hasRecording && !isRecording ? (
                  <motion.div
                    className="mt-8 pt-6 border-t border-white/[0.08]"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ duration: 0.3 }}
                  >
                    <div className="flex items-center gap-3">
                      {/* Play/Pause Button */}
                      <button
                        onClick={() => {
                          if (isPlaying) {
                            setIsPlaying(false);
                          } else {
                            if (playbackProgress >= recordedDuration) {
                              setPlaybackProgress(0);
                            }
                            setIsPlaying(true);
                          }
                        }}
                        className="w-10 h-10 rounded-full bg-white/[0.08] border border-white/[0.12] flex items-center justify-center text-white/70 hover:bg-white/[0.12] transition-colors"
                      >
                        {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4 ml-0.5" />}
                      </button>
                      
                      {/* Time and waveform */}
                      <div className="flex-1">
                        <div className="text-xs text-white/50 mb-1 tabular-nums">
                          {formatTime(Math.floor(playbackProgress))} / {formatTime(recordedDuration)}
                        </div>
                        <div className="h-1 bg-white/[0.05] rounded-full overflow-hidden">
                          <div 
                            className="h-full bg-blue-400/60 transition-all duration-100"
                            style={{ width: `${(playbackProgress / recordedDuration) * 100}%` }}
                          />
                        </div>
                      </div>
                    </div>
                  </motion.div>
                ) : isRecording ? (
                  <motion.div
                    className="mt-8 pt-6 border-t border-white/[0.08]"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ duration: 0.3 }}
                  >
                    <div className="flex items-center gap-3">
                      {/* Recording indicator */}
                      <div className="w-10 h-10 rounded-full bg-red-400/10 border border-red-400/30 flex items-center justify-center">
                        <motion.div
                          className="w-3 h-3 rounded-full bg-red-400"
                          animate={{ opacity: [1, 0.3, 1] }}
                          transition={{ duration: 1.5, repeat: Infinity, ease: 'easeInOut' }}
                        />
                      </div>
                      
                      {/* Recording progress */}
                      <div className="flex-1">
                        <div className="text-xs text-red-400/70 mb-1 uppercase tracking-wider">
                          Recording in progress...
                        </div>
                        <div className="h-1 bg-white/[0.05] rounded-full overflow-hidden">
                          <motion.div 
                            className="h-full bg-red-400/60"
                            animate={{ width: ['0%', '100%'] }}
                            transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
                          />
                        </div>
                      </div>
                    </div>
                  </motion.div>
                ) : (
                  <div className="mt-8 pt-6 border-t border-white/[0.08] text-center">
                    <div className="text-sm text-white/30 tracking-wider">
                      No recording available
                    </div>
                  </div>
                )}
              </div>
            </motion.div>

            <div /> {/* Spacer for bottom alignment */}
          </div>

          {/* Vertical Divider */}
          <div className="w-[1px] bg-gradient-to-b from-transparent via-white/20 to-transparent my-16" />

          {/* Right Panel - Status & Analytics */}
          <div className="flex-1 flex flex-col items-center justify-center p-16">
            <motion.div
              className="w-full max-w-2xl"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.8, delay: 0.4 }}
            >
              {/* Tab Switcher */}
              <motion.div
                className="relative mb-8 p-1.5 backdrop-blur-2xl bg-white/[0.03] border border-white/[0.12] rounded-full inline-flex"
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.5 }}
              >
                {/* Sliding background indicator */}
                <motion.div
                  className="absolute top-1.5 h-[calc(100%-12px)] bg-white/[0.08] rounded-full backdrop-blur-xl border border-white/[0.15]"
                  animate={{
                    left: activeTab === 'analytics' ? '6px' : '50%',
                    width: activeTab === 'analytics' ? 'calc(50% - 9px)' : 'calc(50% - 9px)',
                  }}
                  transition={{ type: 'spring', stiffness: 300, damping: 30 }}
                >
                  <div className="absolute inset-0 bg-gradient-to-br from-white/[0.08] to-transparent rounded-full" />
                </motion.div>

                {/* Tab buttons */}
                <button
                  onClick={() => setActiveTab('analytics')}
                  className="relative z-10 px-8 py-3 text-sm uppercase tracking-[0.2em] transition-colors duration-300"
                  style={{ fontWeight: 500 }}
                >
                  <span className={activeTab === 'analytics' ? 'text-white/95' : 'text-white/40'}>
                    Fleet Analytics
                  </span>
                </button>
                <button
                  onClick={() => setActiveTab('collection')}
                  className="relative z-10 px-8 py-3 text-sm uppercase tracking-[0.2em] transition-colors duration-300"
                  style={{ fontWeight: 500 }}
                >
                  <span className={activeTab === 'collection' ? 'text-white/95' : 'text-white/40'}>
                    Live Collection
                  </span>
                </button>
              </motion.div>

              {/* HUD Information Card */}
              <div className="backdrop-blur-2xl bg-white/[0.03] border border-white/[0.12] rounded-3xl p-12 shadow-2xl">
                {/* Subtle inner glow */}
                <div className="absolute inset-0 rounded-3xl bg-gradient-to-br from-white/[0.03] to-transparent pointer-events-none" />
                
                {activeTab === 'analytics' ? (
                  <>
                    {/* Header */}
                    <div className="mb-10">
                      <h2 className="text-3xl text-white/90 tracking-wide mb-1" style={{ fontWeight: 300 }}>
                        System Status
                      </h2>
                      <div className="h-[1px] w-16 bg-gradient-to-r from-blue-400/60 to-transparent rounded-full" />
                    </div>

                    {/* Status Indicators */}
                    <div className="space-y-6 mb-12">
                      {/* Fleet Collection Status */}
                      <motion.div
                        className="flex items-center gap-4 group"
                        whileHover={{ x: 4 }}
                        transition={{ type: 'spring', stiffness: 300 }}
                      >
                        <div className="relative">
                          <Circle className="w-3 h-3 fill-green-400 text-green-400" />
                          <motion.div
                            className="absolute inset-0 rounded-full bg-green-400/40 blur-md"
                            animate={{ scale: [1, 1.5, 1], opacity: [0.5, 0, 0.5] }}
                            transition={{ duration: 2, repeat: Infinity, ease: 'easeInOut' }}
                          />
                        </div>
                        <span className="text-white/70 text-xl tracking-wide">Fleet Collection</span>
                        <span className="ml-auto text-green-400/90 text-lg uppercase tracking-wider" style={{ fontWeight: 500 }}>
                          Active
                        </span>
                      </motion.div>

                      {/* System Normal Status */}
                      <motion.div
                        className="flex items-center gap-4 group"
                        whileHover={{ x: 4 }}
                        transition={{ type: 'spring', stiffness: 300 }}
                      >
                        <div className="relative">
                          <Circle className="w-3 h-3 fill-green-400 text-green-400" />
                          <motion.div
                            className="absolute inset-0 rounded-full bg-green-400/40 blur-md"
                            animate={{ scale: [1, 1.5, 1], opacity: [0.5, 0, 0.5] }}
                            transition={{ duration: 2, repeat: Infinity, ease: 'easeInOut', delay: 0.5 }}
                          />
                        </div>
                        <span className="text-white/70 text-xl tracking-wide">System Normal</span>
                        <span className="ml-auto text-green-400/90 text-lg uppercase tracking-wider" style={{ fontWeight: 500 }}>
                          Nominal
                        </span>
                      </motion.div>

                      {/* Data Processing Status */}
                      <motion.div
                        className="flex items-center gap-4 group"
                        whileHover={{ x: 4 }}
                        transition={{ type: 'spring', stiffness: 300 }}
                      >
                        <div className="relative">
                          <Circle className="w-3 h-3 fill-amber-400 text-amber-400" />
                          <motion.div
                            className="absolute inset-0 rounded-full bg-amber-400/40 blur-md"
                            animate={{ scale: [1, 1.5, 1], opacity: [0.5, 0, 0.5] }}
                            transition={{ duration: 2, repeat: Infinity, ease: 'easeInOut', delay: 1 }}
                          />
                        </div>
                        <span className="text-white/70 text-xl tracking-wide">Data Processing</span>
                        <span className="ml-auto text-amber-400/90 text-lg uppercase tracking-wider" style={{ fontWeight: 500 }}>
                          Standby
                        </span>
                      </motion.div>
                    </div>

                    {/* Divider */}
                    <div className="h-[1px] bg-gradient-to-r from-transparent via-white/10 to-transparent mb-10" />

                    {/* Anomaly Score Visualization */}
                    <div>
                      <div className="flex items-end justify-between mb-2">
                        <div>
                          <h3 className="text-white/60 text-sm uppercase tracking-[0.2em] mb-0.5">
                            Anomaly Score
                          </h3>
                          <div className="text-4xl text-white/95 tabular-nums font-light">
                            {!hasScore ? (isProcessing ? '---' : '---') : `${anomalyScore}%`}
                          </div>
                        </div>
                        <div className="text-white/40 text-sm uppercase tracking-wider">
                          {isProcessing ? 'Processing...' : hasScore ? 'Normal Range' : 'Awaiting Data'}
                        </div>
                      </div>

                      {/* Processing Animation or Confidence Bar */}
                      {isProcessing ? (
                        <>
                          {/* Processing Bars Animation */}
                          <div className="space-y-2 mb-2">
                            {[0, 1, 2, 3, 4].map((index) => (
                              <div key={index} className="relative h-2 bg-white/[0.05] rounded-full overflow-hidden border border-white/[0.08]">
                                <motion.div
                                  className="absolute left-0 top-0 h-full rounded-full bg-gradient-to-r from-blue-400 via-blue-500 to-cyan-500"
                                  animate={{ 
                                    x: ['-100%', '100%'],
                                  }}
                                  transition={{ 
                                    duration: 1.5,
                                    repeat: Infinity,
                                    ease: 'linear',
                                    delay: index * 0.2
                                  }}
                                >
                                  <motion.div
                                    className="absolute inset-0 bg-gradient-to-r from-transparent via-white/30 to-transparent"
                                    animate={{ x: ['-100%', '200%'] }}
                                    transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
                                  />
                                  <div className="absolute inset-0 shadow-[0_0_15px_rgba(59,130,246,0.5)]" />
                                </motion.div>
                              </div>
                            ))}
                          </div>

                          {/* Processing Status Text */}
                          <motion.div 
                            className="text-center text-blue-400/80 text-sm uppercase tracking-wider h-5 flex items-center justify-center mt-3"
                            animate={{ opacity: [0.5, 1, 0.5] }}
                            transition={{ duration: 1.5, repeat: Infinity, ease: 'easeInOut' }}
                          >
                            AI Analysis in Progress
                          </motion.div>
                        </>
                      ) : hasScore ? (
                        <>
                          {/* Confidence Bar */}
                          <div className="mb-2">
                            <div className="relative h-3 bg-white/[0.05] rounded-full overflow-hidden border border-white/[0.08]">
                              {/* Background gradient track */}
                              <div className="absolute inset-0 bg-gradient-to-r from-green-500/20 via-yellow-500/20 to-red-500/20" />
                              
                              {/* Animated fill */}
                              <motion.div
                                className="absolute left-0 top-0 h-full rounded-full bg-gradient-to-r from-green-400 via-green-500 to-emerald-500"
                                initial={{ width: 0 }}
                                animate={{ width: `${anomalyScore}%` }}
                                transition={{ duration: 1.5, ease: 'easeOut', delay: 0.3 }}
                              >
                                {/* Shimmer effect */}
                                <motion.div
                                  className="absolute inset-0 bg-gradient-to-r from-transparent via-white/30 to-transparent"
                                  animate={{ x: ['-100%', '200%'] }}
                                  transition={{ duration: 2, repeat: Infinity, ease: 'linear', delay: 1 }}
                                />
                                
                                {/* Glow effect */}
                                <div className="absolute inset-0 shadow-[0_0_20px_rgba(74,222,128,0.6)]" />
                              </motion.div>

                              {/* Indicator dot */}
                              <motion.div
                                className="absolute top-1/2 -translate-y-1/2 w-5 h-5 bg-white rounded-full shadow-lg shadow-green-400/50 border-2 border-green-400"
                                initial={{ left: 0 }}
                                animate={{ left: `calc(${anomalyScore}% - 10px)` }}
                                transition={{ duration: 1.5, ease: 'easeOut', delay: 0.3 }}
                              />
                            </div>
                          </div>

                          {/* Scale markers */}
                          <div className="flex justify-between text-xs text-white/30 uppercase tracking-wider px-1 h-5 items-center mt-1">
                            <span>0%</span>
                            <span>25%</span>
                            <span>50%</span>
                            <span>75%</span>
                            <span>100%</span>
                          </div>
                        </>
                      ) : (
                        <>
                          {/* Empty state bars */}
                          <div className="space-y-2 mb-2">
                            {[0, 1, 2, 3, 4].map((index) => (
                              <div key={index} className="relative h-2 bg-white/[0.05] rounded-full overflow-hidden border border-white/[0.08]">
                                {/* Empty bar */}
                              </div>
                            ))}
                          </div>

                          {/* Awaiting message */}
                          <div className="text-center text-white/30 text-sm uppercase tracking-wider h-5 flex items-center justify-center mt-3">
                            No Analysis Available
                          </div>
                        </>
                      )}
                    </div>
                  </>
                ) : (
                  <>
                    {/* Live Collection Tab Content - Audio Recording */}
                    <div className="mb-10">
                      <h2 className="text-3xl text-white/90 tracking-wide mb-1 flex items-center gap-3" style={{ fontWeight: 300 }}>
                        <Mic className="w-7 h-7" />
                        Audio Collection
                      </h2>
                      <div className="h-[1px] w-16 bg-gradient-to-r from-purple-400/60 to-transparent rounded-full" />
                    </div>

                    {/* Microphone Status */}
                    <div className="space-y-6 mb-12">
                      <motion.div
                        className="flex items-center gap-4 group"
                        whileHover={{ x: 4 }}
                        transition={{ type: 'spring', stiffness: 300 }}
                      >
                        <div className="relative">
                          <Circle className={`w-3 h-3 ${isRecording ? 'fill-red-400 text-red-400' : 'fill-green-400 text-green-400'}`} />
                          <motion.div
                            className={`absolute inset-0 rounded-full blur-md ${isRecording ? 'bg-red-400/40' : 'bg-green-400/40'}`}
                            animate={{ scale: [1, 1.5, 1], opacity: [0.5, 0, 0.5] }}
                            transition={{ duration: 2, repeat: Infinity, ease: 'easeInOut' }}
                          />
                        </div>
                        <span className="text-white/70 text-xl tracking-wide">Built-in Microphone</span>
                        <span className={`ml-auto text-lg uppercase tracking-wider ${isRecording ? 'text-red-400/90' : 'text-green-400/90'}`} style={{ fontWeight: 500 }}>
                          {isRecording ? 'Recording' : 'Ready'}
                        </span>
                      </motion.div>
                    </div>

                    {/* Divider */}
                    <div className="h-[1px] bg-gradient-to-r from-transparent via-white/10 to-transparent mb-10" />

                    {/* Audio Level Visualization */}
                    <div>
                      <div className="flex items-end justify-between mb-4">
                        <div>
                          <h3 className="text-white/60 text-sm uppercase tracking-[0.2em] mb-1">
                            Input Level
                          </h3>
                          <div className="text-4xl text-white/95 tabular-nums font-light">
                            {Math.round(audioLevel)}%
                          </div>
                        </div>
                        <div className="text-white/40 text-sm uppercase tracking-wider">
                          {isRecording ? 'Live' : 'Monitoring'}
                        </div>
                      </div>

                      {/* Live Recording Waveform Bars */}
                      <div className="space-y-2 mb-8">
                        {[0, 1, 2, 3, 4].map((index) => {
                          const randomLevel = isRecording ? (Math.random() * 80 + 20) : 0;
                          return (
                            <div key={index} className="relative h-2 bg-white/[0.05] rounded-full overflow-hidden border border-white/[0.08]">
                              <motion.div
                                className="absolute left-0 top-0 h-full rounded-full bg-gradient-to-r from-purple-400 via-purple-500 to-pink-500"
                                animate={{ 
                                  width: isRecording ? `${randomLevel}%` : '0%',
                                }}
                                transition={{ 
                                  duration: 0.1,
                                  ease: 'easeOut'
                                }}
                              >
                                <motion.div
                                  className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent"
                                  animate={{ x: ['-100%', '200%'] }}
                                  transition={{ duration: 1.5, repeat: Infinity, ease: 'linear', delay: index * 0.2 }}
                                />
                                <div className="absolute inset-0 shadow-[0_0_15px_rgba(192,132,252,0.5)]" />
                              </motion.div>
                            </div>
                          );
                        })}
                      </div>

                      {/* Send to Telegram Button */}
                      <motion.button
                        onClick={() => {
                          if (!isRecording) {
                            setIsSent(true);
                          }
                        }}
                        disabled={isRecording}
                        className="w-full relative px-8 py-4 rounded-full overflow-hidden group"
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        whileHover={{ scale: isRecording ? 1 : 1.02 }}
                        whileTap={{ scale: isRecording ? 1 : 0.98 }}
                        transition={{ type: 'spring', stiffness: 400, damping: 17 }}
                      >
                        {/* Button background */}
                        <div className={`absolute inset-0 backdrop-blur-2xl border transition-all duration-500 ${
                          isRecording
                            ? 'bg-white/[0.02] border-white/[0.08] shadow-[0_8px_32px_rgba(255,255,255,0.02)]' 
                            : 'bg-white/[0.03] border-cyan-400/30 shadow-[0_8px_32px_rgba(34,211,238,0.2)]'
                        }`} />
                        
                        {/* Inner glow */}
                        <div className={`absolute inset-0 bg-gradient-to-br pointer-events-none ${
                          isRecording
                            ? 'from-white/[0.02] to-transparent' 
                            : 'from-cyan-400/10 to-transparent'
                        }`} />
                        
                        {/* Shimmer effect */}
                        {!isRecording && (
                          <motion.div
                            className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent"
                            animate={{ x: ['-200%', '200%'] }}
                            transition={{ duration: 3, repeat: Infinity, ease: 'linear' }}
                          />
                        )}
                        
                        {/* Button text */}
                        <span className={`relative flex items-center justify-center gap-3 text-xl tracking-[0.2em] uppercase transition-colors duration-500 ${
                          isRecording ? 'text-white/30' : 'text-cyan-300'
                        }`} style={{ fontWeight: 500 }}>
                          <Send className="w-5 h-5" />
                          Send to Telegram
                        </span>
                      </motion.button>
                    </div>
                  </>
                )}
              </div>

              {/* Bottom metadata */}
              <motion.div
                className="mt-8 text-center text-white/30 text-sm tracking-[0.3em] uppercase"
                animate={{ opacity: [0.3, 0.5, 0.3] }}
                transition={{ duration: 3, repeat: Infinity, ease: 'easeInOut' }}
              >
                Automotive HUD System v2.4
              </motion.div>
            </motion.div>
          </div>
        </div>
      </div>
    </div>
  );
}
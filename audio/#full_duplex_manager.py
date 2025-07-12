# audio/full_duplex_manager.py - BALANCED FIXED VAD (Syntax Corrected)
# Date: 2025-01-07 09:00:15
# FIXES: Balanced VAD with smart timing, all syntax errors fixed

import threading
import time
import queue
import numpy as np
from collections import deque
from config import *

from audio.smart_aec import smart_aec

class FullDuplexManager:
    def __init__(self):
        self.input_queue = queue.Queue(maxsize=100)
        self.output_queue = queue.Queue(maxsize=50)
        self.processed_queue = queue.Queue(maxsize=50)

        self.listening = False
        self.processing = False
        self.buddy_interrupted = False
        
        # ✅ BALANCED: Enhanced state tracking
        self.buddy_is_speaking = False
        self.buddy_speech_start_time = 0
        self.buddy_speech_lock = threading.Lock()

        # ✅ BALANCED: Optimal buffers
        self.mic_buffer = deque(maxlen=8000)
        self.speech_buffer = deque(maxlen=240000)
        self.pre_speech_buffer = deque(maxlen=32000)

        # ✅ BALANCED: Use config values directly
        self.vad_threshold = VAD_THRESHOLD
        self.speech_frames = 0
        self.silence_frames = 0
        self.min_speech_frames = MIN_SPEECH_FRAMES
        self.max_silence_frames = MAX_SILENCE_FRAMES

        # ✅ BALANCED: Use config interrupt threshold
        self.interrupt_threshold = INTERRUPT_THRESHOLD
        self.interrupt_frames = 0
        self.min_interrupt_frames = 8  # Reasonable number

        # ✅ BALANCED: Add dynamic thresholds from config
        self.vad_threshold_buddy_silent = getattr(globals().get('config', {}), 'VAD_THRESHOLD_BUDDY_SILENT', 800)
        self.vad_threshold_buddy_speaking = getattr(globals().get('config', {}), 'VAD_THRESHOLD_BUDDY_SPEAKING', 3500)
        self.interrupt_grace_period_threshold = getattr(globals().get('config', {}), 'INTERRUPT_GRACE_PERIOD_THRESHOLD', 4500)
        self.aec_sync_delay = getattr(globals().get('config', {}), 'AEC_SYNC_DELAY', 0.1)
        self.buddy_speech_grace_period = getattr(globals().get('config', {}), 'BUDDY_SPEECH_GRACE_PERIOD', 0.5)
        self.interrupt_ramp_up_time = getattr(globals().get('config', {}), 'INTERRUPT_RAMP_UP_TIME', 0.3)

        self.noise_baseline = 150
        self.noise_samples = deque(maxlen=200)
        self.noise_calibrated = False
        self.adaptive_threshold_history = deque(maxlen=50)

        self.speech_quality_buffer = deque(maxlen=32)
        self.consecutive_good_frames = 0
        self.last_good_speech_time = 0

        # Stats
        self.interrupts_detected = 0
        self.speeches_processed = 0
        self.buddy_speeches_rejected = 0
        self.false_detections = 0
        self.enhanced_captures = 0
        self.quality_improvements = 0

        self.running = False
        self.threads = []

        print(f"[FullDuplex] 🌟 BALANCED Full Duplex Manager - Smart Timing")
        print(f"[FullDuplex] 🎯 Buddy Silent VAD: {self.vad_threshold_buddy_silent}")
        print(f"[FullDuplex] 🎯 Buddy Speaking VAD: {self.vad_threshold_buddy_speaking}")
        print(f"[FullDuplex] 🎯 Normal Interrupt: {self.interrupt_threshold}")
        print(f"[FullDuplex] 🎯 Grace Period: {self.buddy_speech_grace_period}s")

    def start(self):
        """Start the full duplex manager"""
        if self.running:
            return
        self.running = True
        self.listening = True
        
        self.threads = [
            threading.Thread(target=self._audio_input_worker, daemon=True),
            threading.Thread(target=self._balanced_vad_processor, daemon=True),
            threading.Thread(target=self._speech_processor, daemon=True),
            threading.Thread(target=self._balanced_interrupt_detector, daemon=True),
            threading.Thread(target=self._noise_tracker, daemon=True),
            threading.Thread(target=self._pre_buffer_manager, daemon=True),
        ]
        
        for thread in self.threads:
            thread.start()
        print("[FullDuplex] ✅ BALANCED Full duplex with smart timing started")

    def stop(self):
        """Stop the full duplex manager"""
        self.running = False
        self.listening = False
        
        with self.buddy_speech_lock:
            self.buddy_is_speaking = False
        
        while not self.input_queue.empty():
            try:
                self.input_queue.get_nowait()
            except:
                break
        print("[FullDuplex] 🛑 BALANCED Full duplex stopped")

    def add_audio_input(self, audio_chunk):
        """Add audio input with enhanced processing"""
        if not self.listening:
            return
        try:
            # Always add to pre-buffer first
            self.pre_speech_buffer.extend(audio_chunk)
            
            # ✅ BALANCED: Smart AEC when Buddy speaking
            with self.buddy_speech_lock:
                if self.buddy_is_speaking:
                    processed_chunk = smart_aec.process_microphone_input(audio_chunk)
                else:
                    processed_chunk = audio_chunk
            
            if not self.input_queue.full():
                self.input_queue.put(processed_chunk)
            self.mic_buffer.extend(processed_chunk)
            
        except Exception as e:
            if DEBUG:
                print(f"[FullDuplex] Audio input error: {e}")

    def notify_buddy_speaking(self, audio_data):
        """Notify that Buddy started speaking"""
        try:
            with self.buddy_speech_lock:
                self.buddy_is_speaking = True
                self.buddy_speech_start_time = time.time()
            
            if AEC_ENABLED and not getattr(self, 'aec_emergency_disabled', False):
                smart_aec.update_reference(audio_data)
                
            print("[FullDuplex] 🌟 BALANCED: Buddy STARTED speaking - Smart timing activated")
        except Exception as e:
            if DEBUG:
                print(f"[FullDuplex] Notify buddy speaking error: {e}")

    def notify_buddy_stopped_speaking(self):
        """Notify that Buddy stopped speaking"""
        try:
            with self.buddy_speech_lock:
                self.buddy_is_speaking = False
                
            print("[FullDuplex] 🌟 BALANCED: Buddy STOPPED speaking - Normal VAD resumed")
        except Exception as e:
            if DEBUG:
                print(f"[FullDuplex] Notify buddy stopped error: {e}")

    def _is_buddy_speaking(self):
        """Thread-safe check if Buddy is speaking"""
        with self.buddy_speech_lock:
            return self.buddy_is_speaking

    def _pre_buffer_manager(self):
        """Continuously maintain pre-speech buffer"""
        while self.running:
            try:
                if len(self.pre_speech_buffer) > 0:
                    recent_audio = np.array(list(self.pre_speech_buffer)[-1600:]) if len(self.pre_speech_buffer) >= 1600 else np.array(list(self.pre_speech_buffer))
                    
                    if len(recent_audio) > 0:
                        volume = np.abs(recent_audio).mean()
                        if volume > self.vad_threshold * 0.5:
                            self.adaptive_threshold_history.append(volume)
                
                time.sleep(0.05)
            except Exception as e:
                if DEBUG:
                    print(f"[FullDuplex] Pre-buffer manager error: {e}")

    def _audio_input_worker(self):
        """Process audio input with better buffering"""
        while self.running:
            try:
                audio_chunk = self.input_queue.get(timeout=0.1)
                self.speech_buffer.extend(audio_chunk)
                
                if getattr(self, 'processing', False):
                    if not hasattr(self, '_captured_speech'):
                        self._captured_speech = []
                    self._captured_speech.extend(audio_chunk)
            except queue.Empty:
                continue
            except Exception as e:
                if DEBUG:
                    print(f"[FullDuplex] Input worker error: {e}")

    def _balanced_vad_processor(self):
        """✅ BALANCED FIX: Smart timing-based VAD instead of complete blackout"""
        min_silence_frames = 12  # Reasonable number
        consecutive_silence_frames = 0
        last_debug_time = 0
        buddy_speech_start_time = 0
        last_log_time = 0

        print("[FullDuplex] 🌟 BALANCED VAD PROCESSOR: Smart timing-based approach")

        while self.running:
            try:
                if len(self.speech_buffer) < 160:
                    time.sleep(0.01)
                    continue
                
                chunk = np.array(list(self.speech_buffer)[-160:])
                volume = np.abs(chunk).mean()
                peak = np.max(np.abs(chunk))
                
                buddy_speaking = self._is_buddy_speaking()
                current_time = time.time()
                
                # ✅ BALANCED: Smart threshold selection based on timing
                if buddy_speaking:
                    if buddy_speech_start_time == 0:
                        buddy_speech_start_time = current_time
                        print(f"[FullDuplex] 🌟 SMART TIMING: Buddy started, using progressive thresholds")
                    
                    # ✅ PROGRESSIVE THRESHOLDS: Start high, then lower over time
                    time_since_buddy_started = current_time - buddy_speech_start_time
                    
                    if time_since_buddy_started < self.aec_sync_delay:
                        # Very brief initial delay for AEC sync
                        threshold = 5000  # Very high for AEC sync
                        context = "AEC_SYNC"
                    elif time_since_buddy_started < self.buddy_speech_grace_period:
                        # Grace period with high threshold
                        threshold = self.interrupt_grace_period_threshold
                        context = "GRACE"
                    elif time_since_buddy_started < (self.buddy_speech_grace_period + self.interrupt_ramp_up_time):
                        # Ramping down to normal interrupt threshold
                        progress = (time_since_buddy_started - self.buddy_speech_grace_period) / self.interrupt_ramp_up_time
                        progress = min(1.0, max(0.0, progress))  # Clamp between 0 and 1
                        threshold = self.interrupt_grace_period_threshold - (self.interrupt_grace_period_threshold - self.interrupt_threshold) * progress
                        context = "RAMP"
                    else:
                        # Normal interrupt threshold for natural interrupts
                        threshold = self.interrupt_threshold
                        context = "NORMAL"
                    
                    # ✅ INTERRUPT DETECTION: Allow natural interrupts after ramp-up
                    if volume > threshold or peak > threshold * 1.2:
                        self.interrupt_frames += 1
                        
                        if DEBUG and current_time - last_log_time > 0.5:
                            print(f"[VAD] {context} Interrupt: {self.interrupt_frames}/{self.min_interrupt_frames} (vol:{volume:.0f}>{threshold:.0f})")
                            last_log_time = current_time
                        
                        # ✅ BALANCED: Reasonable interrupt frame requirement
                        required_frames = 6 if context == "NORMAL" else 10
                        if self.interrupt_frames >= required_frames:
                            print(f"\n[FullDuplex] ⚡ {context} INTERRUPT: vol:{volume:.0f} > {threshold:.0f}")
                            self._handle_interrupt()
                            buddy_speech_start_time = 0
                            self.interrupt_frames = 0
                    else:
                        self.interrupt_frames = max(0, self.interrupt_frames - 1)
                    
                    time.sleep(0.01)  # Fast processing for responsiveness
                    continue
                else:
                    # ✅ Buddy not speaking - reset timer and use normal VAD
                    if buddy_speech_start_time > 0:
                        print(f"[FullDuplex] ✅ SMART TIMING: Buddy stopped - normal VAD resumed")
                    buddy_speech_start_time = 0
                    self.interrupt_frames = 0
                
                # ✅ NORMAL VAD: When Buddy is not speaking
                if not self.processing:
                    # ✅ BALANCED: Use normal threshold when Buddy silent
                    speech_start_thresh = self.vad_threshold_buddy_silent
                    
                    if (volume > speech_start_thresh or peak > speech_start_thresh * 1.3):
                        self.speech_frames += 1
                        consecutive_silence_frames = 0
                        
                        if DEBUG and current_time - last_debug_time > 0.5:
                            print(f"[VAD] NORMAL Speech building: {self.speech_frames}/{self.min_speech_frames} (vol:{volume:.0f}>{speech_start_thresh:.0f})")
                            last_debug_time = current_time
                        
                        if self.speech_frames >= self.min_speech_frames:
                            print(f"\n[FullDuplex] 🎤 USER SPEECH DETECTED!")
                            print(f"[FullDuplex] Vol:{volume:.0f} > {speech_start_thresh:.0f}")
                            self._start_enhanced_speech_capture()
                    else:
                        self.speech_frames = max(0, self.speech_frames - 1)
                        
                else:
                    # ✅ SPEECH ENDING: Normal threshold for ending
                    speech_end_thresh = max(300, speech_start_thresh * 0.4)
                    if volume < speech_end_thresh and peak < speech_end_thresh * 1.1:
                        consecutive_silence_frames += 1
                        
                        if DEBUG and current_time - last_debug_time > 0.3:
                            print(f"[VAD] Silence: {consecutive_silence_frames}/{min_silence_frames} (vol:{volume:.0f}<{speech_end_thresh:.0f})")
                            last_debug_time = current_time
                            
                        if consecutive_silence_frames >= min_silence_frames:
                            print(f"\n[FullDuplex] ⏸️ Speech ENDED ({consecutive_silence_frames} silence frames)")
                            self._end_enhanced_speech_capture()
                            consecutive_silence_frames = 0
                            self.speech_frames = 0
                    else:
                        consecutive_silence_frames = 0
                        self.speech_frames += 1
                
                time.sleep(0.01)
                
            except Exception as e:
                if DEBUG:
                    print(f"[FullDuplex] BALANCED VAD error: {e}")

    def _balanced_interrupt_detector(self):
        """✅ BALANCED: Reasonable interrupt detection"""
        last_interrupt_time = 0
        
        print("[FullDuplex] 🌟 BALANCED INTERRUPT DETECTOR: Natural interrupt thresholds")
        
        while self.running:
            try:
                if not INTERRUPT_DETECTION:
                    time.sleep(0.1)
                    continue
                
                # Only detect interrupts when Buddy is actually speaking
                if not self._is_buddy_speaking():
                    time.sleep(0.05)
                    self.interrupt_frames = 0
                    continue
                
                # ✅ BALANCED COOLDOWN: Prevent rapid re-interrupts
                current_time = time.time()
                cooldown = getattr(globals().get('config', {}), 'INTERRUPT_COOLDOWN_TIME', 2.0)
                if current_time - last_interrupt_time < cooldown:
                    time.sleep(0.1)
                    continue
                
                if len(self.mic_buffer) < 160:
                    time.sleep(0.01)
                    continue
                
                chunk = np.array(list(self.mic_buffer)[-160:])
                volume = np.abs(chunk).mean()
                peak = np.max(np.abs(chunk))
                
                # ✅ BALANCED: Natural interrupt threshold
                if volume > self.interrupt_threshold and peak > self.interrupt_threshold * 1.3:
                    self.interrupt_frames += 1
                    
                    if DEBUG:
                        print(f"[INT] BALANCED Interrupt: {self.interrupt_frames}/{self.min_interrupt_frames} (vol:{volume:.0f}>{self.interrupt_threshold:.0f})")
                    
                    if self.interrupt_frames >= self.min_interrupt_frames:
                        print(f"\n[FullDuplex] ⚡ BALANCED CONFIRMED INTERRUPT!")
                        self._handle_interrupt()
                        last_interrupt_time = current_time
                        self.interrupt_frames = 0
                else:
                    self.interrupt_frames = max(0, self.interrupt_frames - 1)
                
                time.sleep(0.01)
                
            except Exception as e:
                if DEBUG:
                    print(f"[FullDuplex] BALANCED interrupt detector error: {e}")

    def _handle_interrupt(self):
        """✅ BALANCED: Handle interrupt"""
        self.interrupts_detected += 1
        self.buddy_interrupted = True
        
        print(f"[FullDuplex] ⚡ BALANCED INTERRUPT #{self.interrupts_detected} - STOPPING BUDDY")
        
        try:
            # Stop audio playback
            from audio.output import current_audio_playback, stop_audio_playback
            
            if current_audio_playback and current_audio_playback.is_playing():
                current_audio_playback.stop()
                print(f"[FullDuplex] 🛑 Audio playback STOPPED")
            
            # Backup stop
            stop_audio_playback()
            
            # Clear Buddy speaking state
            with self.buddy_speech_lock:
                self.buddy_is_speaking = False
            
            print(f"[FullDuplex] ✅ Interrupt successful")
                
        except Exception as e:
            if DEBUG:
                print(f"[FullDuplex] Interrupt handling error: {e}")

    def _start_enhanced_speech_capture(self):
        """Enhanced speech capture"""
        self.processing = True
        self.speech_start_time = time.time()
        self.last_good_speech_time = time.time()
        
        # Pre-context
        pre_context_frames = int(SPEECH_PADDING_START * SAMPLE_RATE * 1.5)
        
        if len(self.pre_speech_buffer) >= pre_context_frames:
            pre_context = list(self.pre_speech_buffer)[-pre_context_frames:]
            print(f"[FullDuplex] 📝 BALANCED: Added {len(pre_context)} pre-speech samples ({len(pre_context)/SAMPLE_RATE:.1f}s)")
        else:
            pre_context = list(self.pre_speech_buffer)
            print(f"[FullDuplex] 📝 BALANCED: Added ALL {len(pre_context)} pre-buffer samples")
        
        self.speech_buffer.clear()
        if pre_context:
            self.speech_buffer.extend(pre_context)
        
        self._captured_speech = []
        print("🔴 BALANCED CAPTURING", end="", flush=True)

    def _end_enhanced_speech_capture(self):
        """Enhanced speech capture ending"""
        if not self.processing:
            return
        
        self.processing = False
        time.sleep(SPEECH_PADDING_END)
        
        # Get captured audio
        total_context_frames = int(WHISPER_CONTEXT_PADDING * SAMPLE_RATE)
        audio_data = np.array(getattr(self, '_captured_speech', []), dtype=np.int16)
        
        if len(audio_data) > total_context_frames:
            audio_data = audio_data[-total_context_frames:]
        
        # Quality checks
        duration = len(audio_data) / SAMPLE_RATE
        volume = np.abs(audio_data).mean()
        
        min_duration = max(MIN_SPEECH_DURATION, 0.3)
        volume_threshold = max(self.noise_baseline * 2, 400)
        
        if duration >= min_duration and volume > volume_threshold:
            print(f"\n[FullDuplex] ✅ BALANCED ACCEPTED speech: {duration:.1f}s (vol:{volume:.0f})")
            
            if len(audio_data) > SAMPLE_RATE * 0.5:
                audio_data = self._enhance_audio_quality(audio_data)
                self.quality_improvements += 1
            
            self.processed_queue.put(audio_data)
            self.speeches_processed += 1
            self.enhanced_captures += 1
        else:
            print(f"\n[FullDuplex] ❌ BALANCED REJECTED speech: {duration:.1f}s (vol:{volume:.0f})")
            print(f"[FullDuplex] Required: >{min_duration:.1f}s, >{volume_threshold:.0f}")
            self.false_detections += 1
        
        # Clean up
        self._captured_speech = []
        
        # Keep buffer for context
        keep_frames = int(1.5 * SAMPLE_RATE)
        if len(self.speech_buffer) > keep_frames:
            kept_audio = list(self.speech_buffer)[-keep_frames:]
            self.speech_buffer.clear()
            self.speech_buffer.extend(kept_audio)
        else:
            self.speech_buffer.clear()

    def _enhance_audio_quality(self, audio_data):
        """Gentle audio enhancement"""
        try:
            audio_float = audio_data.astype(np.float32)
            
            # Very gentle noise gate
            abs_audio = np.abs(audio_float)
            noise_level = np.percentile(abs_audio, 15)
            noise_gate = noise_level * 0.3
            mask = abs_audio > noise_gate
            audio_float = np.where(mask, audio_float, audio_float * 0.5)
            
            # Gentle normalization
            max_val = np.max(np.abs(audio_float))
            if max_val > 0:
                target_max = 16000
                audio_float = audio_float * (target_max / max_val)
            
            return audio_float.astype(np.int16)
        except Exception as e:
            if DEBUG:
                print(f"[FullDuplex] Audio enhancement error: {e}")
            return audio_data

    def _noise_tracker(self):
        """Track noise levels for adaptive thresholds"""
        calibration_samples = 0
        while self.running:
            try:
                if len(self.mic_buffer) >= 160:
                    chunk = np.array(list(self.mic_buffer)[-160:])
                    volume = np.abs(chunk).mean()
                    
                    if not self.processing and not self._is_buddy_speaking():
                        self.noise_samples.append(volume)
                        calibration_samples += 1
                        
                        if not self.noise_calibrated and calibration_samples >= 30:
                            self.noise_baseline = np.percentile(self.noise_samples, 70)
                            self.noise_calibrated = True
                            print(f"[FullDuplex] 🎯 BALANCED Noise calibrated: {self.noise_baseline:.1f}")
                            
                        elif len(self.noise_samples) >= 50:
                            old_baseline = self.noise_baseline
                            new_baseline = np.percentile(self.noise_samples, 70)
                            self.noise_baseline = old_baseline * 0.8 + new_baseline * 0.2
                            
                            if DEBUG and len(self.noise_samples) % 50 == 0:
                                print(f"[FullDuplex] 📊 BALANCED Noise updated: {self.noise_baseline:.1f}")
                
                time.sleep(0.1)
            except Exception as e:
                if DEBUG:
                    print(f"[FullDuplex] Noise tracker error: {e}")

    def _speech_processor(self):
        """Process captured speech"""
        while self.running:
            try:
                audio_data = self.processed_queue.get(timeout=0.5)
                
                def transcribe_and_handle():
                    try:
                        from ai.speech import transcribe_audio
                        
                        if DEBUG:
                            duration = len(audio_data) / SAMPLE_RATE
                            volume = np.abs(audio_data).mean()
                            print(f"[FullDuplex] 🎙️ BALANCED Transcribing: {duration:.1f}s, vol:{volume:.1f}")
                        
                        text = transcribe_audio(audio_data)
                        
                        if text and len(text.strip()) > 0:
                            print(f"[FullDuplex] 📝 BALANCED Transcribed: '{text}'")
                            self._handle_transcribed_text(text, audio_data)
                        else:
                            print(f"[FullDuplex] ❌ BALANCED Empty transcription")
                    except Exception as e:
                        if DEBUG:
                            print(f"[FullDuplex] BALANCED Transcription error: {e}")
                
                threading.Thread(target=transcribe_and_handle, daemon=True).start()
                
            except queue.Empty:
                continue
            except Exception as e:
                if DEBUG:
                    print(f"[FullDuplex] BALANCED Speech processor error: {e}")

    def _handle_transcribed_text(self, text, audio_data):
        """Handle transcribed text"""
        self.last_transcription = (text, audio_data)

    def get_next_speech(self, timeout=0.1):
        """Get next transcribed speech"""
        try:
            if hasattr(self, 'last_transcription'):
                result = self.last_transcription
                delattr(self, 'last_transcription')
                return result
        except:
            pass
        return None

    # Utility methods
    def emergency_disable_aec(self):
        """Emergency disable AEC"""
        self.aec_emergency_disabled = True
        print("[FullDuplex] 🚨 AEC EMERGENCY DISABLED")

    def emergency_enable_aec(self):
        """Emergency enable AEC"""
        self.aec_emergency_disabled = False
        print("[FullDuplex] ✅ AEC EMERGENCY ENABLED")

    def set_raw_audio_mode(self, enabled=True):
        """Set raw audio mode"""
        self.force_raw_audio = enabled
        print(f"[FullDuplex] {'🎙️ RAW AUDIO MODE' if enabled else '🔧 PROCESSING MODE'}")

    def get_stats(self):
        """Get detailed statistics"""
        try:
            try:
                aec_stats = smart_aec.get_stats()
            except:
                aec_stats = {"aec_not_available": True}
            
            return {
                "running": self.running,
                "listening": self.listening,
                "processing": self.processing,
                "buddy_is_speaking": self._is_buddy_speaking(),
                "interrupts_detected": self.interrupts_detected,
                "speeches_processed": self.speeches_processed,
                "enhanced_captures": self.enhanced_captures,
                "quality_improvements": self.quality_improvements,
                "buddy_rejections": self.buddy_speeches_rejected,
                "false_detections": self.false_detections,
                "noise_calibrated": self.noise_calibrated,
                "aec_stats": aec_stats,
                "current_vad_threshold": self.vad_threshold,
                "current_interrupt_threshold": self.interrupt_threshold,
                "noise_baseline": self.noise_baseline,
                "pre_buffer_size": len(self.pre_speech_buffer),
                "balanced_mode": True,
                "config_values": {
                    "VAD_THRESHOLD": VAD_THRESHOLD,
                    "INTERRUPT_THRESHOLD": INTERRUPT_THRESHOLD,
                    "MIN_SPEECH_FRAMES": MIN_SPEECH_FRAMES,
                    "AEC_ENABLED": AEC_ENABLED,
                },
                "buffer_sizes": {
                    "input": self.input_queue.qsize(),
                    "processed": self.processed_queue.qsize(),
                    "mic_buffer": len(self.mic_buffer),
                    "speech_buffer": len(self.speech_buffer),
                }
            }
        except Exception as e:
            return {
                "error": str(e),
                "running": self.running,
                "listening": self.listening,
                "buddy_is_speaking": getattr(self, 'buddy_is_speaking', False),
                "balanced_mode": True
            }

# Create global instance
try:
    full_duplex_manager = FullDuplexManager()
    print("[FullDuplex] ✅ BALANCED Global full duplex manager created - Smart Timing Mode")
except Exception as e:
    print(f"[FullDuplex] ❌ Error creating BALANCED manager: {e}")
    full_duplex_manager = None
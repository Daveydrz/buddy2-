# audio/voice_analyzer.py - Advanced Voice Quality Analysis
# Date: 2025-07-05 09:42:45
# ADVANCED: Multi-layer voice detection to reject background noise

import numpy as np
import scipy.signal as signal
from scipy.fft import fft, fftfreq
from collections import deque
import time
from config import *

class AdvancedVoiceAnalyzer:
    def __init__(self):
        self.noise_baseline = 100
        self.noise_samples = deque(maxlen=500)
        self.voice_samples = deque(maxlen=100)
        self.environment_calibrated = False
        self.calibration_start_time = time.time()
        
        # Voice detection history
        self.recent_voice_scores = deque(maxlen=20)
        self.recent_spectral_scores = deque(maxlen=20)
        self.recent_temporal_scores = deque(maxlen=20)
        
        print("[VoiceAnalyzer] 🧠 Advanced Voice Analyzer initialized")
        print(f"[VoiceAnalyzer] 🎯 Voice Quality Threshold: {USER_SPEECH_QUALITY_THRESHOLD}")
        print(f"[VoiceAnalyzer] 📊 Spectral Threshold: {USER_SPEECH_SPECTRAL_THRESHOLD}")

    def analyze_audio_chunk(self, audio_chunk, is_buddy_speaking=False):
        """
        Advanced multi-layer analysis to determine if audio contains human voice
        Returns: (is_voice, quality_score, detailed_scores)
        """
        try:
            # ✅ FIXED: Comprehensive input validation
            if audio_chunk is None:
                return False, 0.0, {'error': 'audio_chunk is None'}
            
            # Convert to numpy array and ensure proper data type
            try:
                audio_chunk = np.array(audio_chunk, dtype=np.int16)
            except (ValueError, TypeError) as e:
                if DEBUG:
                    print(f"[VoiceAnalyzer] ❌ Invalid audio data type: {e}")
                return False, 0.0, {'error': 'invalid_data_type'}
            
            # Check for empty or too small chunks
            if len(audio_chunk) == 0:
                return False, 0.0, {'error': 'empty_chunk'}
            
            if len(audio_chunk) < 40:  # Minimum viable chunk size
                return False, 0.0, {'error': 'chunk_too_small', 'length': len(audio_chunk)}
            
            # ✅ FIXED: Clean and validate audio data
            # Remove any NaN, inf, or invalid values
            audio_chunk = np.nan_to_num(audio_chunk, nan=0, posinf=0, neginf=0)
            
            # Clip to valid int16 range
            audio_chunk = np.clip(audio_chunk, -32768, 32767)
            
            # Convert to float for analysis with proper scaling
            try:
                audio_float = audio_chunk.astype(np.float32)
                max_val = np.max(np.abs(audio_float))
                if max_val > 1.0:
                    audio_float = audio_float / 32768.0
                
                # Final validation of float array
                if np.any(np.isnan(audio_float)) or np.any(np.isinf(audio_float)):
                    audio_float = np.nan_to_num(audio_float, nan=0, posinf=0, neginf=0)
                    
            except Exception as e:
                if DEBUG:
                    print(f"[VoiceAnalyzer] ❌ Float conversion error: {e}")
                return False, 0.0, {'error': 'float_conversion_failed'}
            
            # ✅ FIXED: Calculate basic metrics with error handling
            try:
                volume = float(np.abs(audio_chunk).mean())
                peak = float(np.max(np.abs(audio_chunk)))
                
                # Validate basic metrics
                if np.isnan(volume) or np.isinf(volume):
                    volume = 0.0
                if np.isnan(peak) or np.isinf(peak):
                    peak = 0.0
                    
            except Exception as e:
                if DEBUG:
                    print(f"[VoiceAnalyzer] ❌ Basic metrics error: {e}")
                return False, 0.0, {'error': 'basic_metrics_failed'}
            
            # ✅ FIXED: Early exit for very quiet audio to avoid processing errors
            if volume < 5.0:  # Very quiet, likely silence
                return False, 0.0, {
                    'volume': volume,
                    'peak': peak,
                    'combined': 0.0,
                    'reason': 'too_quiet'
                }
            
            # ✅ FIXED: Run analysis layers with individual error handling
            spectral_score = 0.0
            temporal_score = 0.0
            quality_score = 0.0
            harmonic_score = 0.0
            noise_score = 0.5  # Default neutral score
            
            # Spectral analysis with error handling
            try:
                spectral_score = self._spectral_voice_analysis(audio_float)
                if np.isnan(spectral_score) or np.isinf(spectral_score):
                    spectral_score = 0.0
                spectral_score = max(0.0, min(1.0, float(spectral_score)))
            except Exception as e:
                if DEBUG and "non-printable character" not in str(e):
                    print(f"[VoiceAnalyzer] Spectral analysis error: {e}")
                spectral_score = 0.0
            
            # Temporal analysis with error handling
            try:
                temporal_score = self._temporal_pattern_analysis(audio_float)
                if np.isnan(temporal_score) or np.isinf(temporal_score):
                    temporal_score = 0.0
                temporal_score = max(0.0, min(1.0, float(temporal_score)))
            except Exception as e:
                if DEBUG and "non-printable character" not in str(e):
                    print(f"[VoiceAnalyzer] Temporal analysis error: {e}")
                temporal_score = 0.0
            
            # Quality analysis with error handling
            try:
                quality_score = self._voice_quality_analysis(audio_float)
                if np.isnan(quality_score) or np.isinf(quality_score):
                    quality_score = 0.0
                quality_score = max(0.0, min(1.0, float(quality_score)))
            except Exception as e:
                if DEBUG and "non-printable character" not in str(e):
                    print(f"[VoiceAnalyzer] Quality analysis error: {e}")
                quality_score = 0.0
            
            # Harmonic analysis with error handling
            try:
                harmonic_score = self._harmonic_analysis(audio_float)
                if np.isnan(harmonic_score) or np.isinf(harmonic_score):
                    harmonic_score = 0.0
                harmonic_score = max(0.0, min(1.0, float(harmonic_score)))
            except Exception as e:
                if DEBUG and "non-printable character" not in str(e):
                    print(f"[VoiceAnalyzer] Harmonic analysis error: {e}")
                harmonic_score = 0.0
            
            # Noise analysis with error handling
            try:
                noise_score = self._noise_analysis(audio_float, volume)
                if np.isnan(noise_score) or np.isinf(noise_score):
                    noise_score = 0.5
                noise_score = max(0.0, min(1.0, float(noise_score)))
            except Exception as e:
                if DEBUG and "non-printable character" not in str(e):
                    print(f"[VoiceAnalyzer] Noise analysis error: {e}")
                noise_score = 0.5
            
            # ✅ FIXED: Combine scores with validation
            try:
                combined_score = (
                    spectral_score * 0.25 +
                    temporal_score * 0.20 +
                    quality_score * 0.25 +
                    harmonic_score * 0.15 +
                    noise_score * 0.15
                )
                
                # Validate combined score
                if np.isnan(combined_score) or np.isinf(combined_score):
                    combined_score = 0.0
                combined_score = max(0.0, min(1.0, float(combined_score)))
                
            except Exception as e:
                if DEBUG:
                    print(f"[VoiceAnalyzer] Score combination error: {e}")
                combined_score = 0.0
            
            # ✅ FIXED: Safe threshold determination
            try:
                if is_buddy_speaking:
                    # Much stricter when Buddy is speaking (interrupts)
                    voice_threshold = getattr(self, 'buddy_interrupt_quality_threshold', 
                                            getattr(globals(), 'BUDDY_INTERRUPT_QUALITY_THRESHOLD', 0.8))
                    volume_threshold = getattr(self, 'buddy_interrupt_threshold',
                                             getattr(globals(), 'BUDDY_INTERRUPT_THRESHOLD', 2500))
                else:
                    # Normal thresholds when waiting for user
                    voice_threshold = getattr(self, 'user_speech_quality_threshold',
                                            getattr(globals(), 'USER_SPEECH_QUALITY_THRESHOLD', 0.6))
                    volume_threshold = getattr(self, 'user_speech_threshold',
                                             getattr(globals(), 'USER_SPEECH_THRESHOLD', 800))
                
                spectral_threshold = getattr(self, 'user_speech_spectral_threshold',
                                           getattr(globals(), 'USER_SPEECH_SPECTRAL_THRESHOLD', 0.5))
                
            except Exception as e:
                if DEBUG:
                    print(f"[VoiceAnalyzer] Threshold determination error: {e}")
                voice_threshold = 0.6
                volume_threshold = 800
                spectral_threshold = 0.5
            
            # ✅ FIXED: Safe voice detection decision
            try:
                passes_noise_gate = self._passes_noise_gate(audio_float, volume)
            except Exception as e:
                if DEBUG and "non-printable character" not in str(e):
                    print(f"[VoiceAnalyzer] Noise gate error: {e}")
                passes_noise_gate = volume > volume_threshold * 0.5  # Fallback
            
            is_voice = (
                volume > volume_threshold and
                combined_score > voice_threshold and
                spectral_score > spectral_threshold and
                passes_noise_gate
            )
            
            # ✅ FIXED: Safe storage for adaptation
            try:
                if is_voice:
                    if hasattr(self, 'voice_samples') and hasattr(self, 'recent_voice_scores'):
                        self.voice_samples.append(combined_score)
                        self.recent_voice_scores.append(combined_score)
                else:
                    if hasattr(self, 'noise_samples'):
                        self.noise_samples.append(volume)
                
                # Update environment calibration
                if hasattr(self, '_update_environment_calibration'):
                    self._update_environment_calibration()
                    
            except Exception as e:
                if DEBUG:
                    print(f"[VoiceAnalyzer] Storage/calibration error: {e}")
            
            # ✅ FIXED: Build detailed scores dictionary safely
            detailed_scores = {
                'volume': float(volume),
                'peak': float(peak),
                'spectral': float(spectral_score),
                'temporal': float(temporal_score),
                'quality': float(quality_score),
                'harmonic': float(harmonic_score),
                'noise': float(noise_score),
                'combined': float(combined_score),
                'threshold': float(voice_threshold),
                'is_voice': bool(is_voice),
                'analysis_success': True
            }
            
            # ✅ FIXED: Safe debug output
            try:
                if (DEBUG and 
                    getattr(globals(), 'SHOW_VOICE_QUALITY_SCORES', True) and
                    (is_voice or combined_score > 0.3)):
                    print(f"[VoiceAnalyzer] {'✅' if is_voice else '❌'} "
                          f"Vol:{volume:.0f} Combined:{combined_score:.2f} "
                          f"Spec:{spectral_score:.2f} Qual:{quality_score:.2f}")
            except Exception as e:
                # Silently ignore debug output errors
                pass
            
            return bool(is_voice), float(combined_score), detailed_scores
            
        except Exception as e:
            if DEBUG:
                print(f"[VoiceAnalyzer] ❌ Critical analysis error: {e}")
            return False, 0.0, {
                'error': 'critical_failure',
                'error_message': str(e),
                'volume': 0.0,
                'combined': 0.0
            }

    def _spectral_voice_analysis(self, audio_float):
        """Analyze frequency content to detect voice characteristics"""
        try:
            if len(audio_float) < 160:
                return 0.0
            
            # FFT analysis
            window = np.hanning(len(audio_float))
            windowed = audio_float * window
            fft_result = fft(windowed)
            freqs = fftfreq(len(audio_float), 1/SAMPLE_RATE)
            magnitude = np.abs(fft_result)
            
            # Focus on positive frequencies
            positive_freq_mask = freqs > 0
            freqs = freqs[positive_freq_mask]
            magnitude = magnitude[positive_freq_mask]
            
            # Voice frequency ranges
            voice_mask = (freqs >= VOICE_FREQUENCY_MIN) & (freqs <= VOICE_FREQUENCY_MAX)
            formant1_mask = (freqs >= 200) & (freqs <= 1000)   # First formant
            formant2_mask = (freqs >= 1000) & (freqs <= 3000)  # Second formant
            formant3_mask = (freqs >= 2000) & (freqs <= 4000)  # Third formant
            
            if not np.any(voice_mask):
                return 0.0
            
            # Calculate energy in different bands
            total_energy = np.sum(magnitude**2)
            if total_energy == 0:
                return 0.0
                
            voice_energy = np.sum(magnitude[voice_mask]**2)
            formant1_energy = np.sum(magnitude[formant1_mask]**2)
            formant2_energy = np.sum(magnitude[formant2_mask]**2)
            formant3_energy = np.sum(magnitude[formant3_mask]**2)
            
            # Voice characteristics
            voice_ratio = voice_energy / total_energy
            formant_ratio = (formant1_energy + formant2_energy + formant3_energy) / total_energy
            
            # Spectral centroid (brightness)
            spectral_centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
            centroid_score = 1.0 if 200 <= spectral_centroid <= 2000 else 0.5
            
            # Combine scores
            spectral_score = (voice_ratio * 0.4 + formant_ratio * 0.4 + centroid_score * 0.2)
            
            return min(1.0, spectral_score)
            
        except Exception as e:
            if DEBUG:
                print(f"[VoiceAnalyzer] Spectral analysis error: {e}")
            return 0.0

    def _temporal_pattern_analysis(self, audio_float):
        """Analyze temporal patterns typical of human speech"""
        try:
            if len(audio_float) < 320:  # Need at least 20ms
                return 0.0
            
            # Zero crossing rate
            zcr = self._calculate_zcr(audio_float)
            zcr_score = 1.0 if MIN_ZERO_CROSSING_RATE <= zcr <= MAX_ZERO_CROSSING_RATE else 0.0
            
            # Energy variance (speech has varying energy)
            frame_size = 160  # 10ms frames
            frames = [audio_float[i:i+frame_size] for i in range(0, len(audio_float)-frame_size, frame_size)]
            if len(frames) < 2:
                return zcr_score * 0.5
            
            frame_energies = [np.sum(frame**2) for frame in frames]
            if np.mean(frame_energies) == 0:
                return 0.0
                
            energy_variance = np.var(frame_energies) / np.mean(frame_energies)
            variance_score = min(1.0, energy_variance / 0.5)  # Normalize
            
            # Temporal regularity (speech has some regularity but not too much)
            regularity_score = 1.0 - min(1.0, abs(energy_variance - 0.3) / 0.3)
            
            temporal_score = (zcr_score * 0.4 + variance_score * 0.3 + regularity_score * 0.3)
            
            return temporal_score
            
        except Exception as e:
            if DEBUG:
                print(f"[VoiceAnalyzer] Temporal analysis error: {e}")
            return 0.0

    def _voice_quality_analysis(self, audio_float):
        """Analyze overall voice quality indicators"""
        try:
            if len(audio_float) < 160:
                return 0.0
            
            # Signal-to-noise ratio estimation
            signal_power = np.mean(audio_float**2)
            if signal_power == 0:
                return 0.0
                
            # Estimate noise floor (bottom 10% of signal)
            sorted_power = np.sort(audio_float**2)
            noise_floor = np.mean(sorted_power[:len(sorted_power)//10])
            
            if noise_floor == 0:
                snr_score = 1.0
            else:
                snr = signal_power / noise_floor
                snr_score = min(1.0, np.log10(snr) / 2.0)  # Normalize SNR
            
            # Spectral flatness (voice is not flat)
            fft_result = fft(audio_float)
            magnitude = np.abs(fft_result)
            magnitude = magnitude[magnitude > 0]  # Remove zeros
            
            if len(magnitude) == 0:
                flatness_score = 0.0
            else:
                geometric_mean = np.exp(np.mean(np.log(magnitude)))
                arithmetic_mean = np.mean(magnitude)
                spectral_flatness = geometric_mean / arithmetic_mean
                flatness_score = 1.0 - min(1.0, spectral_flatness)  # Voice should not be flat
            
            # Combine quality indicators
            quality_score = (snr_score * 0.6 + flatness_score * 0.4)
            
            return quality_score
            
        except Exception as e:
            if DEBUG:
                print(f"[VoiceAnalyzer] Quality analysis error: {e}")
            return 0.0

    def _harmonic_analysis(self, audio_float):
        """Detect harmonic content typical of voice"""
        try:
            if len(audio_float) < 512:
                return 0.0
            
            # Autocorrelation for pitch detection
            correlation = np.correlate(audio_float, audio_float, mode='full')
            correlation = correlation[len(correlation)//2:]
            
            # Look for peaks in autocorrelation (pitch periods)
            min_pitch_samples = SAMPLE_RATE // MAX_PITCH_HZ
            max_pitch_samples = SAMPLE_RATE // MIN_PITCH_HZ
            
            if len(correlation) < max_pitch_samples:
                return 0.0
            
            search_range = correlation[min_pitch_samples:min(max_pitch_samples, len(correlation))]
            if len(search_range) == 0:
                return 0.0
            
            max_correlation = np.max(search_range)
            mean_correlation = np.mean(search_range)
            
            if mean_correlation == 0:
                harmonic_score = 0.0
            else:
                harmonic_score = (max_correlation - mean_correlation) / mean_correlation
                harmonic_score = min(1.0, harmonic_score / 2.0)  # Normalize
            
            return harmonic_score
            
        except Exception as e:
            if DEBUG:
                print(f"[VoiceAnalyzer] Harmonic analysis error: {e}")
            return 0.0

    def _noise_analysis(self, audio_float, volume):
        """Analyze noise characteristics"""
        try:
            # Compare to learned noise baseline
            if len(self.noise_samples) > 10:
                current_noise_baseline = np.percentile(self.noise_samples, 70)
            else:
                current_noise_baseline = self.noise_baseline
            
            # Noise ratio
            if current_noise_baseline == 0:
                noise_ratio = 1.0
            else:
                noise_ratio = min(1.0, volume / (current_noise_baseline * 3))
            
            # Adaptive scoring based on environment
            if self.environment_calibrated:
                # Well-calibrated environment
                noise_score = noise_ratio
            else:
                # Still learning environment - be more permissive
                noise_score = min(1.0, noise_ratio * 1.5)
            
            return noise_score
            
        except Exception as e:
            if DEBUG:
                print(f"[VoiceAnalyzer] Noise analysis error: {e}")
            return 0.5

    def _passes_noise_gate(self, audio_float, volume):
        """Advanced noise gate with adaptive threshold"""
        try:
            if not NOISE_GATE_ENABLED:
                return True
            
            # Adaptive noise gate threshold
            if len(self.noise_samples) > 10:
                noise_threshold = np.percentile(self.noise_samples, 80) * (1 + NOISE_GATE_THRESHOLD)
            else:
                noise_threshold = self.noise_baseline * (1 + NOISE_GATE_THRESHOLD)
            
            # Apply environment adaptation
            if self.environment_calibrated:
                if len(self.noise_samples) > 50:
                    avg_noise = np.mean(list(self.noise_samples)[-50:])
                    if avg_noise < 50:  # Quiet environment
                        noise_threshold *= QUIET_ENVIRONMENT_BONUS
                    elif avg_noise > 200:  # Noisy environment
                        noise_threshold *= NOISY_ENVIRONMENT_PENALTY
            
            return volume > noise_threshold
            
        except Exception as e:
            if DEBUG:
                print(f"[VoiceAnalyzer] Noise gate error: {e}")
            return True

    def _calculate_zcr(self, audio_float):
        """Calculate zero crossing rate"""
        try:
            signs = np.sign(audio_float)
            sign_changes = np.diff(signs)
            zero_crossings = np.sum(np.abs(sign_changes)) / 2
            zcr = zero_crossings / len(audio_float)
            return zcr
        except:
            return 0.0

    def _update_environment_calibration(self):
        """Update environment calibration"""
        try:
            # Calibrate for initial period
            if not self.environment_calibrated:
                elapsed = time.time() - self.calibration_start_time
                if elapsed > ENVIRONMENT_CALIBRATION_TIME or len(self.noise_samples) > 100:
                    self.environment_calibrated = True
                    if len(self.noise_samples) > 0:
                        self.noise_baseline = np.percentile(self.noise_samples, 70)
                    print(f"[VoiceAnalyzer] ✅ Environment calibrated: noise baseline {self.noise_baseline:.1f}")
            
            # Continuous adaptation
            if ADAPTIVE_NOISE_FLOOR and len(self.noise_samples) > 20:
                recent_noise = np.mean(list(self.noise_samples)[-20:])
                self.noise_baseline = (self.noise_baseline * (1 - NOISE_ADAPTATION_RATE) + 
                                     recent_noise * NOISE_ADAPTATION_RATE)
                
        except Exception as e:
            if DEBUG:
                print(f"[VoiceAnalyzer] Calibration error: {e}")

    def get_environment_stats(self):
        """Get environment and analysis statistics"""
        try:
            return {
                'noise_baseline': self.noise_baseline,
                'environment_calibrated': self.environment_calibrated,
                'noise_samples_count': len(self.noise_samples),
                'voice_samples_count': len(self.voice_samples),
                'recent_voice_avg': np.mean(self.recent_voice_scores) if self.recent_voice_scores else 0,
                'recent_noise_avg': np.mean(list(self.noise_samples)[-10:]) if len(self.noise_samples) >= 10 else self.noise_baseline,
                'calibration_time': time.time() - self.calibration_start_time
            }
        except:
            return {}

# Create global instance
try:
    voice_analyzer = AdvancedVoiceAnalyzer()
    print("[VoiceAnalyzer] ✅ Global voice analyzer created")
except Exception as e:
    print(f"[VoiceAnalyzer] ❌ Error creating voice analyzer: {e}")
    voice_analyzer = None
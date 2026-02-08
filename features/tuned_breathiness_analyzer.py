#!/usr/bin/env python3
"""
Breathiness Analyzer - TUNED VERSION based on your test results
"""

import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import threading
import time

class TunedBreathinessAnalyzer:
    def __init__(self, sample_rate=16000, buffer_size=2048):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.audio_buffer = deque(maxlen=buffer_size * 2)
        
        # Based on your test data, we'll tune these weights
        self.breathiness = 50
        self.debug_info = {}
        
        # History for smoothing
        self.breathiness_history = deque(maxlen=20)
        
        # Settings
        self.amplitude_threshold = 0.01
        self.voice_active = False
        
        self.lock = threading.Lock()
        
    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"Audio: {status}")
        
        audio_data = indata[:, 0] if len(indata.shape) > 1 else indata
        self.audio_buffer.extend(audio_data)
    
    def detect_pitch(self, audio):
        """Improved pitch detection"""
        # Normalize
        max_amp = np.max(np.abs(audio))
        if max_amp < 1e-10:
            return 0
            
        audio_norm = audio / max_amp
        
        # Autocorrelation with pre-emphasis (emphasize high frequencies)
        # This helps with breathy voice detection
        audio_pre = np.diff(audio_norm)
        audio_pre = np.append(audio_pre, 0)
        
        corr = np.correlate(audio_pre, audio_pre, mode='full')
        corr = corr[len(corr)//2:]
        
        # Voice range
        min_lag = int(self.sample_rate / 400)
        max_lag = int(self.sample_rate / 80)
        
        if max_lag >= len(corr):
            return 0
        
        corr_slice = corr[min_lag:max_lag]
        if len(corr_slice) == 0:
            return 0
        
        # Find peaks
        peak_idx = np.argmax(corr_slice)
        peak_val = corr_slice[peak_idx]
        
        # Lower threshold for breathy voice (whisper has weaker correlation)
        if peak_val < 0.15:  # Reduced from 0.2
            return 0
        
        freq = self.sample_rate / (peak_idx + min_lag)
        
        # Slightly wider range for breathy voice
        if 70 <= freq <= 450:
            return freq
        
        return 0
    
    def calculate_hnr_fixed(self, audio, pitch):
        """Fixed HNR calculation"""
        if pitch == 0:
            return 0
        
        # Normalize
        audio_norm = audio / (np.max(np.abs(audio)) + 1e-10)
        
        # Calculate autocorrelation
        corr = np.correlate(audio_norm, audio_norm, mode='full')
        corr = corr[len(corr)//2:]
        
        # Find period in samples
        period = self.sample_rate / pitch
        
        if period < 10 or period >= len(corr):
            return 0
        
        # Look for correlation peak near the period
        start = int(period * 0.8)
        end = min(len(corr), int(period * 1.2))
        
        if end <= start:
            return 0
        
        # Find maximum in this range
        corr_peak = np.max(corr[start:end])
        corr_0 = corr[0]
        
        if corr_0 > 0 and corr_peak > 0:
            # Harmonic energy is the peak, noise is the difference
            harmonic_energy = corr_peak
            noise_energy = corr_0 - corr_peak
            
            if noise_energy > 0:
                hnr_db = 10 * np.log10(harmonic_energy / noise_energy)
                return max(-10, min(30, hnr_db))
        
        return 0
    
    def analyze_breathiness_tuned(self, audio, pitch):
        """Tuned based on your test data"""
        if pitch == 0:
            return 50, {}
        
        # Compute spectrum
        windowed = audio * np.hamming(len(audio))
        fft = np.fft.rfft(windowed)
        fft_mag = np.abs(fft)
        freqs = np.fft.rfftfreq(len(audio), 1/self.sample_rate)
        
        # 1. HARMONIC STRENGTH (your data shows this is most important)
        # Your clear voice: 3.978, whisper: 0.003
        harmonic_energy = 0
        for h in range(1, 6):
            harmonic_freq = pitch * h
            if harmonic_freq > self.sample_rate / 2:
                break
            
            idx = np.argmin(np.abs(freqs - harmonic_freq))
            if idx < len(fft_mag):
                harmonic_energy += fft_mag[idx]
        
        # Normalize by number of harmonics found
        harmonic_strength = harmonic_energy / 5 if harmonic_energy > 0 else 0
        
        # 2. NOISE RATIO (high vs low frequency)
        high_mask = (freqs >= 2000) & (freqs <= 5000)  # Adjusted range
        low_mask = (freqs >= 100) & (freqs <= 1000)
        
        high_energy = np.mean(fft_mag[high_mask]) if np.any(high_mask) else 0
        low_energy = np.mean(fft_mag[low_mask]) if np.any(low_mask) else 1e-10
        
        noise_ratio = high_energy / low_energy
        
        # 3. SPECTRAL TILT
        valid = (freqs >= 100) & (freqs <= 4000)
        if np.sum(valid) > 10:
            fft_db = 20 * np.log10(fft_mag[valid] + 1e-10)
            slope = np.polyfit(freqs[valid], fft_db, 1)[0]
            spectral_tilt = slope * 1000  # dB/kHz
        else:
            spectral_tilt = 0
        
        # 4. HNR (fixed calculation)
        hnr = self.calculate_hnr_fixed(audio, pitch)
        
        # 5. SPECTRAL FLATNESS (noisiness measure)
        # Breathy voice has flatter spectrum (more like noise)
        if len(fft_mag) > 10:
            geometric_mean = np.exp(np.mean(np.log(fft_mag + 1e-10)))
            arithmetic_mean = np.mean(fft_mag)
            if arithmetic_mean > 0:
                spectral_flatness = geometric_mean / arithmetic_mean
            else:
                spectral_flatness = 0
        else:
            spectral_flatness = 0
        
        # SCORING - TUNED FOR YOUR VOICE BASED ON TEST DATA
        
        # 1. Harmonic strength (MOST IMPORTANT based on your data)
        # Clear: ~4.0, Whisper: ~0.003
        # Map: 0 → 100, 4 → 0
        if harmonic_strength > 0:
            score_harmonic = max(0, 100 - (harmonic_strength * 25))
        else:
            score_harmonic = 100
        
        # 2. Noise ratio (important but less than harmonic strength)
        # Clear: 0.068, Whisper: 0.361
        # Map: 0 → 0, 0.5 → 100
        score_noise = min(100, noise_ratio * 200)
        
        # 3. Spectral tilt
        # Clear: -1.4, Whisper: -2.1, "Airy": -7.0
        # More negative = more breathy
        score_tilt = max(0, min(100, -spectral_tilt * 10))
        
        # 4. HNR (when fixed)
        # Typically: -10 to 30 dB, lower = more breathy
        score_hnr = max(0, min(100, (15 - hnr) * 4))
        
        # 5. Spectral flatness (0-1, higher = more breathy)
        score_flatness = spectral_flatness * 100
        
        # Debug info
        self.debug_info = {
            'harmonic_strength': harmonic_strength,
            'noise_ratio': noise_ratio,
            'spectral_tilt': spectral_tilt,
            'hnr': hnr,
            'spectral_flatness': spectral_flatness,
            'pitch': pitch
        }
        
        # WEIGHTS TUNED FOR YOUR VOICE:
        # Based on your test, harmonic strength is the best differentiator
        # Your whisper had harmonic_strength 0.003 vs clear 3.978
        weights = [0.40, 0.25, 0.15, 0.10, 0.10]  # Sum to 1.0
        scores = [score_harmonic, score_noise, score_tilt, score_hnr, score_flatness]
        
        breathiness = sum(w * s for w, s in zip(weights, scores))
        
        # Store individual scores for debugging
        self.debug_info['scores'] = scores
        self.debug_info['weights'] = weights
        
        return breathiness, self.debug_info
    
    def analyze(self):
        """Main analysis"""
        if len(self.audio_buffer) < self.buffer_size:
            return None
        
        audio = np.array(list(self.audio_buffer)[-self.buffer_size:])
        amplitude = np.max(np.abs(audio))
        
        if amplitude < self.amplitude_threshold:
            self.voice_active = False
            return None
        
        self.voice_active = True
        
        # Detect pitch
        pitch = self.detect_pitch(audio)
        
        if pitch == 0:
            return None
        
        # Analyze breathiness
        breathiness, debug = self.analyze_breathiness_tuned(audio, pitch)
        
        # Get spectrum for display
        windowed = audio * np.hamming(len(audio))
        fft = np.fft.rfft(windowed)
        fft_mag = np.abs(fft)
        freqs = np.fft.rfftfreq(len(audio), 1/self.sample_rate)
        
        # Update with exponential smoothing
        self.breathiness_history.append(breathiness)
        
        with self.lock:
            if len(self.breathiness_history) > 0:
                # Use weighted average (recent samples more important)
                weights = np.exp(np.linspace(0, 1, len(self.breathiness_history)))
                weights = weights / np.sum(weights)
                self.breathiness = np.average(list(self.breathiness_history), weights=weights)
        
        return {
            'pitch': pitch,
            'breathiness': self.breathiness,
            'amplitude': amplitude,
            'debug': debug,
            'fft_mag': fft_mag,
            'freqs': freqs
        }

class TunedGUI:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(16, 12))
        
        # Create layout
        gs = self.fig.add_gridspec(4, 2, height_ratios=[1.2, 1, 1, 1], hspace=0.3, wspace=0.3)
        
        # Main breathiness display
        self.ax_main = self.fig.add_subplot(gs[0, :])
        self.ax_main.axis('off')
        self.text_breathiness = self.ax_main.text(0.5, 0.7, "", fontsize=36, 
                                                 ha='center', va='center', 
                                                 fontweight='bold')
        self.text_pitch = self.ax_main.text(0.5, 0.4, "", fontsize=24, 
                                           ha='center', va='center')
        self.text_instruction = self.ax_main.text(0.5, 0.1, "", fontsize=16,
                                                 ha='center', va='center')
        
        # Audio waveform
        self.ax_audio = self.fig.add_subplot(gs[1, 0])
        self.line_audio, = self.ax_audio.plot([], [], 'cyan', linewidth=1)
        self.ax_audio.set_title('Audio Waveform')
        self.ax_audio.set_xlabel('Samples')
        self.ax_audio.set_ylabel('Amplitude')
        self.ax_audio.set_ylim(-0.3, 0.3)
        self.ax_audio.grid(True, alpha=0.2)
        
        # Spectrum
        self.ax_spectrum = self.fig.add_subplot(gs[1, 1])
        self.line_spectrum, = self.ax_spectrum.plot([], [], 'yellow', linewidth=1)
        self.ax_spectrum.set_title('Frequency Spectrum')
        self.ax_spectrum.set_xlabel('Frequency (Hz)')
        self.ax_spectrum.set_ylabel('Magnitude')
        self.ax_spectrum.set_xlim(0, 6000)
        self.ax_spectrum.axvspan(2000, 5000, color='red', alpha=0.2, label='Breathiness range')
        self.ax_spectrum.legend(fontsize=8)
        self.ax_spectrum.grid(True, alpha=0.2)
        
        # Breathiness history
        self.ax_history = self.fig.add_subplot(gs[2, 0])
        self.breath_data = deque(maxlen=50)
        self.line_breath, = self.ax_history.plot([], [], 'r', linewidth=2)
        self.ax_history.set_title('Breathiness Over Time')
        self.ax_history.set_xlabel('Time')
        self.ax_history.set_ylabel('Score')
        self.ax_history.set_ylim(0, 100)
        self.ax_history.axhspan(0, 30, color='green', alpha=0.1, label='Clear')
        self.ax_history.axhspan(30, 70, color='yellow', alpha=0.1, label='Moderate')
        self.ax_history.axhspan(70, 100, color='red', alpha=0.1, label='Breathy')
        self.ax_history.legend(loc='upper right', fontsize=8)
        self.ax_history.grid(True, alpha=0.2)
        
        # Component bars
        self.ax_components = self.fig.add_subplot(gs[2, 1])
        self.ax_components.set_title('Breathiness Components')
        self.ax_components.set_ylabel('Score (0-100)')
        self.ax_components.set_ylim(0, 100)
        self.bar_components = None
        self.component_labels = ['Harmonic', 'Noise', 'Tilt', 'HNR', 'Flatness']
        
        # Debug info
        self.ax_debug = self.fig.add_subplot(gs[3, :])
        self.ax_debug.axis('off')
        self.text_debug = self.ax_debug.text(0.02, 0.95, "", fontsize=9,
                                           fontfamily='monospace',
                                           verticalalignment='top')
        
        plt.tight_layout()
    
    def update_plot(self, frame):
        result = self.analyzer.analyze()
        
        # Update audio waveform
        if len(self.analyzer.audio_buffer) >= self.analyzer.buffer_size:
            audio = np.array(list(self.analyzer.audio_buffer)[-self.analyzer.buffer_size:])
            self.line_audio.set_data(range(len(audio)), audio)
            self.ax_audio.set_xlim(0, len(audio))
        
        if result:
            # Update spectrum
            mask = result['freqs'] <= 6000
            self.line_spectrum.set_data(result['freqs'][mask], result['fft_mag'][mask])
            if len(result['fft_mag'][mask]) > 0:
                self.ax_spectrum.set_ylim(0, np.max(result['fft_mag'][mask]) * 1.1)
            
            # Update breathiness
            breathiness = result['breathiness']
            self.breath_data.append(breathiness)
            self.line_breath.set_data(range(len(self.breath_data)), list(self.breath_data))
            self.ax_history.set_xlim(0, max(50, len(self.breath_data)))
            
            # Update main display
            if breathiness < 30:
                color = 'green'
                quality = 'CLEAR'
                instruction = "Try whispering to increase breathiness"
            elif breathiness < 70:
                color = 'yellow'
                quality = 'MODERATE'
                instruction = "Try clear voice (lower) or whisper (higher)"
            else:
                color = 'red'
                quality = 'BREATHY'
                instruction = "Good! Try clear voice to decrease breathiness"
            
            self.text_breathiness.set_text(f"BREATHINESS: {breathiness:.1f}")
            self.text_breathiness.set_color(color)
            
            self.text_pitch.set_text(f"Pitch: {result['pitch']:.1f} Hz | Amplitude: {result['amplitude']:.3f}")
            self.text_instruction.set_text(instruction)
            
            # Update component bars
            if 'debug' in result and 'scores' in result['debug']:
                scores = result['debug']['scores']
                
                if self.bar_components is None:
                    x_pos = np.arange(len(self.component_labels))
                    self.bar_components = self.ax_components.bar(x_pos, scores, 
                                                               color=['red', 'orange', 'yellow', 'cyan', 'purple'],
                                                               alpha=0.7)
                    self.ax_components.set_xticks(x_pos)
                    self.ax_components.set_xticklabels(self.component_labels, rotation=45, fontsize=9)
                else:
                    for bar, score in zip(self.bar_components, scores):
                        bar.set_height(score)
                
                self.ax_components.set_ylim(0, 100)
                
                # Update debug info
                debug = result['debug']
                debug_text = f"""
                Harmonic strength: {debug['harmonic_strength']:.3f} (score: {scores[0]:.1f})
                Noise ratio:       {debug['noise_ratio']:.3f} (score: {scores[1]:.1f})
                Spectral tilt:     {debug['spectral_tilt']:.1f} dB/kHz (score: {scores[2]:.1f})
                HNR:               {debug['hnr']:.1f} dB (score: {scores[3]:.1f})
                Spectral flatness: {debug['spectral_flatness']:.3f} (score: {scores[4]:.1f})
                """
                self.text_debug.set_text(debug_text)
        
        else:
            # No voice
            if self.analyzer.voice_active:
                self.text_breathiness.set_text("ANALYZING...")
                self.text_breathiness.set_color('yellow')
                self.text_instruction.set_text("Hold steady for analysis")
            else:
                self.text_breathiness.set_text("SPEAK OR SING")
                self.text_breathiness.set_color('gray')
                self.text_instruction.set_text("To see breathiness analysis")
            
            self.text_pitch.set_text("")
            self.text_debug.set_text("")
        
        return (self.text_breathiness, self.text_pitch, self.text_instruction,
                self.line_audio, self.line_spectrum, self.line_breath)

def quick_verification():
    """Quick test to verify the tuning works"""
    analyzer = TunedBreathinessAnalyzer()
    
    print("\n" + "="*60)
    print("QUICK VERIFICATION TEST")
    print("="*60)
    print("We'll test whisper vs clear voice with tuned algorithm")
    print("\nPress Enter when ready for WHISPER test...")
    input()
    
    # Test whisper
    print("Recording whisper...")
    duration = 3
    whisper = sd.rec(int(duration * 16000), samplerate=16000, channels=1, dtype='float32')
    sd.wait()
    
    # Take middle chunk
    audio = whisper.flatten()
    if len(audio) > 2048:
        start = len(audio) // 2 - 1024
        audio = audio[start:start + 2048]
    
    pitch = analyzer.detect_pitch(audio)
    if pitch > 0:
        breathiness, debug = analyzer.analyze_breathiness_tuned(audio, pitch)
        print(f"\nWHISPER:")
        print(f"  Pitch: {pitch:.1f} Hz")
        print(f"  Breathiness: {breathiness:.1f}")
        print(f"  Harmonic strength: {debug['harmonic_strength']:.3f}")
        whisper_score = breathiness
    else:
        print("No pitch detected in whisper")
        whisper_score = 0
    
    print("\nPress Enter when ready for CLEAR VOICE test...")
    input()
    
    # Test clear voice
    print("Recording clear voice...")
    clear = sd.rec(int(duration * 16000), samplerate=16000, channels=1, dtype='float32')
    sd.wait()
    
    audio = clear.flatten()
    if len(audio) > 2048:
        start = len(audio) // 2 - 1024
        audio = audio[start:start + 2048]
    
    pitch = analyzer.detect_pitch(audio)
    if pitch > 0:
        breathiness, debug = analyzer.analyze_breathiness_tuned(audio, pitch)
        print(f"\nCLEAR VOICE:")
        print(f"  Pitch: {pitch:.1f} Hz")
        print(f"  Breathiness: {breathiness:.1f}")
        print(f"  Harmonic strength: {debug['harmonic_strength']:.3f}")
        clear_score = breathiness
    else:
        print("No pitch detected in clear voice")
        clear_score = 100  # Arbitrary high
    
    print("\n" + "="*60)
    print("RESULTS:")
    print(f"  Whisper breathiness: {whisper_score:.1f}")
    print(f"  Clear voice breathiness: {clear_score:.1f}")
    print(f"  Difference: {whisper_score - clear_score:.1f}")
    
    if whisper_score > clear_score + 20:
        print("✓ SUCCESS: Whisper correctly detected as more breathy!")
        print("\nStarting real-time analysis...")
        return True
    else:
        print("⚠ Need more tuning. Let's adjust...")
        return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Tuned Breathiness Analyzer')
    parser.add_argument('--verify', action='store_true', help='Run verification test first')
    parser.add_argument('--threshold', type=float, default=0.01, help='Amplitude threshold')
    
    args = parser.parse_args()
    
    if args.verify:
        if not quick_verification():
            print("\nThe algorithm needs more tuning for your voice.")
            print("We'll proceed anyway, but results may not be optimal.")
            input("Press Enter to continue...")
    
    print("\n" + "="*60)
    print("TUNED BREATHINESS ANALYZER")
    print("="*60)
    print("Algorithm tuned based on your previous test results")
    print("\nKey improvements:")
    print("1. Fixed HNR calculation (was always 0)")
    print("2. Adjusted frequency ranges for your mic")
    print("3. Emphasis on harmonic strength (best differentiator)")
    print("4. Added spectral flatness measure")
    print("="*60)
    print("\nWatch the component bars to see what changes:")
    print("• 'Harmonic' should be LOW for breathy voice")
    print("• 'Noise' should be HIGH for breathy voice")
    print("• 'Tilt' should be HIGH for breathy voice")
    print("="*60)
    
    analyzer = TunedBreathinessAnalyzer()
    analyzer.amplitude_threshold = args.threshold
    
    stream = sd.InputStream(
        channels=1,
        samplerate=16000,
        blocksize=1024,
        callback=analyzer.audio_callback,
        dtype='float32'
    )
    
    try:
        stream.start()
        gui = TunedGUI(analyzer)
        ani = FuncAnimation(gui.fig, gui.update_plot, interval=100, blit=False)
        plt.show()
        
    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        stream.stop()
        stream.close()

if __name__ == '__main__':
    main()
"""
Real-time Vocal Register Detection System

This script provides real-time chest vs head voice detection from:
1. Live microphone input
2. Pre-recorded audio files
3. Streaming audio processing

Usage:
    # Live microphone detection
    python realtime_detection.py --mode live
    
    # Process audio file
    python realtime_detection.py --mode file --input song.mp3
    
    # Streaming with visualization
    python realtime_detection.py --mode live --visualize
"""

import numpy as np
import librosa
import sounddevice as sd
import queue
import threading
import argparse
import time
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pickle

class RealtimeVocalRegisterDetector:
    """
    Real-time vocal register detection using trained SVM model
    """
    
    def __init__(self, model_path='vocal_register_svm.pkl', 
                 sample_rate=22050,
                 window_size=2.0,
                 hop_size=0.5):
        """
        Initialize real-time detector
        
        Args:
            model_path: Path to trained model
            sample_rate: Audio sample rate (Hz)
            window_size: Analysis window size (seconds)
            hop_size: How often to analyze (seconds)
        """
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.hop_size = hop_size
        self.window_samples = int(window_size * sample_rate)
        self.hop_samples = int(hop_size * sample_rate)
        
        # Load trained model
        self.load_model(model_path)
        
        # Audio buffer for real-time processing
        self.audio_buffer = deque(maxlen=self.window_samples * 2)
        self.audio_queue = queue.Queue()
        
        # Results storage
        self.results_history = deque(maxlen=100)
        self.current_prediction = None
        self.current_confidence = 0.0
        
        # Threading
        self.is_running = False
        self.processing_thread = None
        
    def load_model(self, model_path):
        """Load trained SVM model"""
        print(f"Loading model from {model_path}...")
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.svm = model_data['svm']
        self.scaler = model_data['scaler']
        self.img_height = model_data['img_height']
        self.img_width = model_data['img_width']
        print("Model loaded successfully!")
        
    def audio_to_melspec(self, audio):
        """Convert audio to mel-spectrogram"""
        mel_spec = librosa.feature.melspectrogram(
            y=audio, 
            sr=self.sample_rate,
            n_mels=128,
            fmax=8000
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db
    
    def preprocess_spectrogram(self, mel_spec_db):
        """Preprocess mel-spectrogram to fixed size image"""
        import cv2
        
        # Normalize to 0-255 range
        mel_spec_norm = ((mel_spec_db - mel_spec_db.min()) / 
                        (mel_spec_db.max() - mel_spec_db.min()) * 255).astype(np.uint8)
        
        # Resize to target dimensions
        img_resized = cv2.resize(mel_spec_norm, (self.img_width, self.img_height))
        
        return img_resized
    
    def extract_features(self, img):
        """Flatten image to feature vector"""
        return img.flatten()
    
    def predict(self, audio_segment):
        """
        Predict vocal register for audio segment by choosing the class with highest probability.

        Returns:
            (label, confidence): Prediction and confidence score
        """
        # Convert to mel-spectrogram
        mel_spec = self.audio_to_melspec(audio_segment)

        # Preprocess
        img = self.preprocess_spectrogram(mel_spec)

        # Extract features
        features = self.extract_features(img).reshape(1, -1)

        # Scale
        features_scaled = self.scaler.transform(features)

        # Get probability for each class
        probabilities = self.svm.predict_proba(features_scaled)[0]

        # Choose the class with the highest probability
        pred_index = np.argmax(probabilities)
        confidence = probabilities[pred_index]

        # Map predicted class to label
        # You can adapt this mapping to your SVM's class names if needed
        class_name = self.svm.classes_[pred_index]
        if str(class_name).lower() in ['1', 'head', 'falsetto', 'f_falsetto', 'm_falsetto']:
            label = "Head Voice"
        else:
            label = "Chest Voice"

        return label, confidence

    
    def audio_callback(self, indata, frames, time_info, status):
        """
        Callback for sounddevice stream (called for each audio chunk)
        """
        if status:
            print(f"Audio status: {status}")
        
        # Add audio to queue for processing
        self.audio_queue.put(indata.copy())
    
    def process_audio_stream(self):
        """
        Process audio from queue in separate thread
        """
        samples_since_last_prediction = 0
        
        while self.is_running:
            try:
                # Get audio chunk from queue
                audio_chunk = self.audio_queue.get(timeout=0.1)
                
                # Add to buffer
                self.audio_buffer.extend(audio_chunk.flatten())
                samples_since_last_prediction += len(audio_chunk)
                
                # Check if we should make a prediction
                if samples_since_last_prediction >= self.hop_samples:
                    # Get analysis window
                    if len(self.audio_buffer) >= self.window_samples:
                        # Get most recent window_samples
                        audio_window = np.array(list(self.audio_buffer)[-self.window_samples:])
                        
                        # Make prediction
                        label, confidence = self.predict(audio_window)
                        
                        # Store results
                        self.current_prediction = label
                        self.current_confidence = confidence
                        self.results_history.append({
                            'time': time.time(),
                            'label': label,
                            'confidence': confidence
                        })
                        
                        # Print result
                        print(f"\r{label:12s} | Confidence: {confidence:6.1%}", end='', flush=True)
                        
                        samples_since_last_prediction = 0
                        
            except queue.Empty:
                continue
            except Exception as e:
                print(f"\nError in processing: {e}")
    
    def start_live_detection(self, device=None, visualize=False):
        """
        Start real-time detection from microphone
        
        Args:
            device: Audio input device (None for default)
            visualize: Show live visualization
        """
        print("\n" + "=" * 60)
        print("STARTING LIVE VOCAL REGISTER DETECTION")
        print("=" * 60)
        print(f"Sample rate: {self.sample_rate} Hz")
        print(f"Window size: {self.window_size} seconds")
        print(f"Update rate: {self.hop_size} seconds")
        print("\nSpeak or sing into your microphone...")
        print("Press Ctrl+C to stop\n")
        
        self.is_running = True
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self.process_audio_stream)
        self.processing_thread.start()
        
        # Start audio stream
        try:
            with sd.InputStream(
                device=device,
                channels=1,
                samplerate=self.sample_rate,
                blocksize=int(self.sample_rate * 0.1),  # 100ms blocks
                callback=self.audio_callback
            ):
                if visualize:
                    self.visualize_live()
                else:
                    # Just keep running
                    while self.is_running:
                        time.sleep(0.1)
                        
        except KeyboardInterrupt:
            print("\n\nStopping detection...")
        finally:
            self.stop()
    
    def process_file(self, filepath, visualize=False):
        """
        Process pre-recorded audio file
        
        Args:
            filepath: Path to audio file
            visualize: Show visualization of results
        """
        print("\n" + "=" * 60)
        print("PROCESSING AUDIO FILE")
        print("=" * 60)
        print(f"File: {filepath}")
        
        # Load audio
        audio, sr = librosa.load(filepath, sr=self.sample_rate)
        duration = len(audio) / sr
        print(f"Duration: {duration:.2f} seconds")
        
        # Process in segments
        results = []
        num_windows = (len(audio) - self.window_samples) // self.hop_samples + 1
        
        print(f"\nAnalyzing {num_windows} segments...\n")
        
        for i in range(0, len(audio) - self.window_samples, self.hop_samples):
            segment = audio[i:i + self.window_samples]
            label, confidence = self.predict(segment)
            
            time_start = i / sr
            time_end = (i + self.window_samples) / sr
            
            results.append({
                'time_start': time_start,
                'time_end': time_end,
                'label': label,
                'confidence': confidence,
                'is_chest': label == "Chest Voice"
            })
            
            print(f"{time_start:6.2f}s - {time_end:6.2f}s: {label:12s} | {confidence:6.1%}")
        
        # Summary statistics
        chest_count = sum(1 for r in results if r['is_chest'])
        head_count = len(results) - chest_count
        
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Total segments: {len(results)}")
        print(f"Chest voice: {chest_count} ({chest_count/len(results)*100:.1f}%)")
        print(f"Head voice: {head_count} ({head_count/len(results)*100:.1f}%)")
        print(f"Average confidence: {np.mean([r['confidence'] for r in results]):.1%}")
        
        if visualize:
            self.visualize_file_results(results, duration)
        
        return results
    
    def visualize_file_results(self, results, duration):
        """
        Visualize analysis results for audio file
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Timeline visualization
        times = [(r['time_start'] + r['time_end']) / 2 for r in results]
        colors = ['blue' if r['is_chest'] else 'red' for r in results]
        confidences = [r['confidence'] for r in results]
        
        ax1.scatter(times, [1]*len(times), c=colors, s=100, alpha=0.6)
        ax1.set_ylim(0.5, 1.5)
        ax1.set_xlim(0, duration)
        ax1.set_xlabel('Time (seconds)', fontsize=12)
        ax1.set_ylabel('Register', fontsize=12)
        ax1.set_yticks([1])
        ax1.set_yticklabels([''])
        ax1.grid(True, alpha=0.3)
        ax1.set_title('Vocal Register Timeline', fontsize=14, fontweight='bold')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='blue', label='Chest Voice'),
            Patch(facecolor='red', label='Head Voice')
        ]
        ax1.legend(handles=legend_elements, loc='upper right')
        
        # Confidence over time
        ax2.plot(times, confidences, 'g-', linewidth=2, label='Confidence')
        ax2.fill_between(times, confidences, alpha=0.3, color='green')
        ax2.set_xlim(0, duration)
        ax2.set_ylim(0, 1)
        ax2.set_xlabel('Time (seconds)', fontsize=12)
        ax2.set_ylabel('Confidence', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.set_title('Prediction Confidence Over Time', fontsize=14, fontweight='bold')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    def visualize_live(self):
        """
        Live visualization of detection results
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        def update(frame):
            if not self.is_running:
                return
            
            # Clear axes
            ax1.clear()
            ax2.clear()
            
            # Display current prediction
            if self.current_prediction:
                color = 'blue' if self.current_prediction == "Chest Voice" else 'red'
                ax1.barh([0], [1], color=color, alpha=0.7)
                ax1.set_xlim(0, 1)
                ax1.set_ylim(-0.5, 0.5)
                ax1.set_yticks([0])
                ax1.set_yticklabels([self.current_prediction])
                ax1.set_xlabel('Confidence')
                ax1.set_title(f'Current: {self.current_prediction} ({self.current_confidence:.1%})', 
                            fontsize=14, fontweight='bold')
                
                # Plot confidence bar
                ax1.axvline(self.current_confidence, color='black', linewidth=2, linestyle='--')
            
            # Plot history
            if len(self.results_history) > 0:
                recent_results = list(self.results_history)[-50:]  # Last 50 predictions
                
                chest_confidences = [r['confidence'] if r['label'] == "Chest Voice" else 0 
                                   for r in recent_results]
                head_confidences = [r['confidence'] if r['label'] == "Head Voice" else 0 
                                  for r in recent_results]
                
                x = range(len(recent_results))
                ax2.fill_between(x, chest_confidences, label='Chest Voice', 
                               color='blue', alpha=0.5)
                ax2.fill_between(x, head_confidences, label='Head Voice', 
                               color='red', alpha=0.5)
                ax2.set_ylim(0, 1)
                ax2.set_xlabel('Recent Predictions')
                ax2.set_ylabel('Confidence')
                ax2.set_title('Prediction History', fontsize=12, fontweight='bold')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
        
        ani = FuncAnimation(fig, update, interval=100, cache_frame_data=False)
        plt.tight_layout()
        
        try:
            plt.show()
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()
    
    def stop(self):
        """Stop detection"""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join()
        print("\nDetection stopped.")


# ============== COMMAND LINE INTERFACE ==============

def main():
    parser = argparse.ArgumentParser(
        description='Real-time Vocal Register Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--mode', choices=['live', 'file'], default='live',
                       help='Detection mode: live microphone or audio file')
    parser.add_argument('--input', type=str,
                       help='Input audio file (required for file mode)')
    parser.add_argument('--model', type=str, default='vocal_register_svm.pkl',
                       help='Path to trained model file')
    parser.add_argument('--window', type=float, default=2.0,
                       help='Analysis window size in seconds')
    parser.add_argument('--hop', type=float, default=0.5,
                       help='Time between predictions in seconds')
    parser.add_argument('--visualize', action='store_true',
                       help='Show live visualization')
    parser.add_argument('--device', type=int, default=None,
                       help='Audio input device ID (use --list-devices to see options)')
    parser.add_argument('--list-devices', action='store_true',
                       help='List available audio devices')
    
    args = parser.parse_args()
    
    # List devices if requested
    if args.list_devices:
        print("\nAvailable audio devices:")
        print(sd.query_devices())
        return
    
    # Validate arguments
    if args.mode == 'file' and not args.input:
        parser.error("--input is required for file mode")
    
    # Create detector
    detector = RealtimeVocalRegisterDetector(
        model_path=args.model,
        window_size=args.window,
        hop_size=args.hop
    )
    
    # Run detection
    if args.mode == 'live':
        detector.start_live_detection(device=args.device, visualize=args.visualize)
    else:
        detector.process_file(args.input, visualize=args.visualize)


if __name__ == "__main__":
    main()
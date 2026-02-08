"""
Flask Backend for Vocal Register Detection Web App

This server provides API endpoints for the web frontend to:
- Upload and analyze audio files
- Process real-time microphone audio
- Get prediction results

Usage:
    python backend_server.py
    Then open frontend.html in your browser
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
import librosa
import io
import base64
import pickle
import cv2
import os
from datetime import datetime
import argparse

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

parser = argparse.ArgumentParser(description='Run the backend server')
parser.add_argument('--port', type=int, default=8080, help='Port to run the server on')
args = parser.parse_args()

class VocalRegisterPredictor:
    """Simple predictor class for the backend"""
    
    def __init__(self, model_path='vocal_register_svm.pkl'):
        self.load_model(model_path)
        
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
        
    def audio_to_melspec(self, audio, sr=22050):
        """Convert audio to mel-spectrogram"""
        mel_spec = librosa.feature.melspectrogram(
            y=audio, 
            sr=sr,
            n_mels=128,
            fmax=8000
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db
    
    def preprocess_spectrogram(self, mel_spec_db):
        """Preprocess mel-spectrogram to fixed size image"""
        # Normalize to 0-255 range
        mel_spec_norm = ((mel_spec_db - mel_spec_db.min()) / 
                        (mel_spec_db.max() - mel_spec_db.min()) * 255).astype(np.uint8)
        
        # Resize to target dimensions
        img_resized = cv2.resize(mel_spec_norm, (self.img_width, self.img_height))
        
        return img_resized
    
    def extract_features(self, img):
        """Flatten image to feature vector"""
        return img.flatten()
    
    def predict(self, audio, sr=22050):
        """
        Predict vocal register for audio segment
        
        Returns:
            dict with label, confidence, and probabilities
        """
        audio = np.asarray(audio, dtype=np.float64)
        if sr != 22050:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=22050)
            sr = 22050
        if len(audio) < 1024:
            raise ValueError(f"Audio too short: {len(audio)} samples (need at least ~0.05s)")
        # Convert to mel-spectrogram
        mel_spec = self.audio_to_melspec(audio, sr)
        
        # Preprocess
        img = self.preprocess_spectrogram(mel_spec)
        
        # Extract features
        features = self.extract_features(img).reshape(1, -1)
        
        # Scale
        features_scaled = self.scaler.transform(features)
        
        # Predict
        probabilities = self.svm.predict_proba(features_scaled)[0]  # [0.569, 0.431]
        prediction = np.argmax(probabilities)
        confidence = probabilities[prediction]
        
        label = "Head Voice/Falsetto" if prediction == 1 else "Chest Voice"
        
        return {
            'label': label,
            'confidence': float(confidence),
            'chest_probability': float(probabilities[0]),
            'head_probability': float(probabilities[1]),
            'prediction_code': int(prediction)
        }

# Initialize predictor
try:
    predictor = VocalRegisterPredictor()
    model_loaded = True
except Exception as e:
    print(f"Warning: Could not load model: {e}")
    print("Server will run but predictions will fail until model is available")
    predictor = None
    model_loaded = False


@app.route('/')
def index():
    """Serve the frontend HTML"""
    try:
        return send_file('frontend.html')
    except FileNotFoundError:
        return """
        <html>
        <body>
            <h1>Vocal Register Detection Server</h1>
            <p>Server is running, but frontend.html not found in current directory.</p>
            <p>Please make sure frontend.html is in the same folder as backend_server.py</p>
            <hr>
            <p>API Endpoints:</p>
            <ul>
                <li>GET /api/health - Health check</li>
                <li>POST /api/predict_audio - Single prediction</li>
                <li>POST /api/predict_segments - Timeline analysis</li>
                <li>POST /api/predict_realtime - Real-time microphone</li>
            </ul>
        </body>
        </html>
        """, 200


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/predict_audio', methods=['POST'])
def predict_audio():
    """
    Predict vocal register from uploaded audio file or base64 audio data
    
    Expects JSON with either:
    - 'audio_base64': base64 encoded audio data
    - 'sample_rate': sample rate of audio
    
    Or form data with:
    - 'file': audio file upload
    """
    if not model_loaded:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Check if it's JSON with base64 audio
        if request.is_json:
            data = request.get_json()
            audio_base64 = data.get('audio_base64')
            sample_rate = data.get('sample_rate', 22050)
            
            if not audio_base64:
                return jsonify({'error': 'No audio_base64 provided'}), 400
            
            # Decode base64 audio
            audio_bytes = base64.b64decode(audio_base64.split(',')[1] if ',' in audio_base64 else audio_base64)
            audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
            
        # Check if it's file upload
        elif 'file' in request.files:
            file = request.files['file']
            
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            # Read file
            audio_bytes = file.read()
            if len(audio_bytes) == 0:
                return jsonify({'error': 'File is empty'}), 400
            
            try:
                audio_array, sample_rate = librosa.load(io.BytesIO(audio_bytes), sr=22050)
            except Exception as load_err:
                return jsonify({
                    'error': f'Could not decode audio file. Use WAV or MP3. ({load_err})'
                }), 400
            
        else:
            return jsonify({'error': 'No audio data provided'}), 400
        
        # Make prediction
        result = predictor.predict(audio_array, sample_rate)
        
        return jsonify({
            'success': True,
            'prediction': result
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/predict_segments', methods=['POST'])
def predict_segments():
    """
    Predict vocal register for multiple segments of audio
    Returns timeline of predictions
    """
    if not model_loaded:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Get file
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        window_size = float(request.form.get('window_size', 2.0))
        hop_size = float(request.form.get('hop_size', 0.5))
        
        # Load audio
        audio_bytes = file.read()
        audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=22050)
        
        duration = len(audio) / sr
        window_samples = int(window_size * sr)
        hop_samples = int(hop_size * sr)
        
        # Process segments
        results = []
        for i in range(0, len(audio) - window_samples, hop_samples):
            segment = audio[i:i + window_samples]
            prediction = predictor.predict(segment, sr)
            
            results.append({
                'time_start': float(i / sr),
                'time_end': float((i + window_samples) / sr),
                'label': prediction['label'],
                'confidence': prediction['confidence'],
                'chest_probability': prediction['chest_probability'],
                'head_probability': prediction['head_probability']
            })
        
        # Calculate statistics
        chest_count = sum(1 for r in results if r['label'] == 'Chest Voice')
        head_count = len(results) - chest_count
        avg_confidence = np.mean([r['confidence'] for r in results])
        
        return jsonify({
            'success': True,
            'duration': float(duration),
            'num_segments': len(results),
            'segments': results,
            'statistics': {
                'chest_count': chest_count,
                'head_count': head_count,
                'chest_percentage': float(chest_count / len(results) * 100) if results else 0,
                'head_percentage': float(head_count / len(results) * 100) if results else 0,
                'average_confidence': float(avg_confidence)
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/predict_realtime', methods=['POST'])
def predict_realtime():
    """
    Predict from real-time microphone audio chunk
    Expects base64 encoded Float32Array from browser
    """
    if not model_loaded:
        return jsonify({'success': False, 'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json() or {}
        audio_base64 = data.get('audio_base64')
        sample_rate = int(data.get('sample_rate', 22050))
        
        if not audio_base64:
            return jsonify({'success': False, 'error': 'No audio data provided'}), 400
        
        # Decode base64 to float32 array
        audio_bytes = base64.b64decode(audio_base64)
        audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
        
        if len(audio_array) < 1024:
            return jsonify({
                'success': False,
                'error': f'Audio too short: {len(audio_array)} samples (record at least ~0.5s)'
            }), 400
        
        # Make prediction (predictor resamples to 22050 if needed)
        result = predictor.predict(audio_array, sample_rate)
        
        return jsonify({
            'success': True,
            'prediction': result,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"[predict_realtime] Error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("VOCAL REGISTER DETECTION SERVER")
    print("=" * 60)
    print("\nServer starting...")
    print("\n🌐 OPEN IN BROWSER:")
    print(f"   http://localhost:{args.port}")
    print("\nAPI Endpoints:")
    print("  - GET  /               - Web interface")
    print("  - GET  /api/health     - Health check")
    print("  - POST /api/predict_audio - Single prediction")
    print("  - POST /api/predict_segments - Timeline analysis")
    print("  - POST /api/predict_realtime - Real-time microphone")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 60 + "\n")
    
    app.run(host='0.0.0.0', port=args.port, debug=True)
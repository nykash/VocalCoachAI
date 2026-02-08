# test_model.py
import pickle
import librosa
import numpy as np
import cv2
import sys

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "vocal_register_svm.pkl"  # path to your trained model
AUDIO_PATH = "test_audio.wav"        # path to audio file to test

# -----------------------------
# LOAD MODEL
# -----------------------------
with open(MODEL_PATH, "rb") as f:
    model_data = pickle.load(f)

svm = model_data['svm']
scaler = model_data['scaler']
img_height = model_data['img_height']
img_width = model_data['img_width']

print("Model loaded successfully!")

# -----------------------------
# LOAD AUDIO
# -----------------------------
try:
    audio, sr = librosa.load(AUDIO_PATH, sr=22050)
except Exception as e:
    print(f"Error loading audio file: {e}")
    sys.exit(1)

# -----------------------------
# PROCESS AUDIO TO FEATURES
# -----------------------------
# Convert to mel-spectrogram
mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=8000)
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

# Normalize and resize like training
mel_spec_norm = ((mel_spec_db - mel_spec_db.min()) /
                 (mel_spec_db.max() - mel_spec_db.min()) * 255).astype(np.uint8)
img_resized = cv2.resize(mel_spec_norm, (img_width, img_height))
features = img_resized.flatten().reshape(1, -1)

# Scale
features_scaled = scaler.transform(features)

# -----------------------------
# PREDICT
# -----------------------------
label = svm.predict(features_scaled)[0]
probabilities = svm.predict_proba(features_scaled)[0]
label_str = "Head Voice/Falsetto" if label == 1 else "Chest Voice"
confidence = probabilities[label]

print(f"\nPrediction: {label_str}")
print(f"Confidence: {confidence:.1%}")

"""
Real-time vocal register service: Chest vs Head vs Falsetto.

Uses the trained Head_Chest_Voice SVM for chest vs (head+falsetto), then
a head-vs-falsetto discriminator based on acoustic features (breathiness,
spectral tilt, energy). The head/falsetto classifier is feature-based and
can be replaced with a trained model when labeled data is available.
"""

from __future__ import annotations

import base64
import os
import pickle
from typing import Any, Dict, Optional

import numpy as np
import librosa

# Optional: cv2 for same preprocessing as Head_Chest_Voice; fallback to numpy
try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_HEAD_CHEST_VOICE_DIR = os.path.join(_ROOT, "Head_Chest_Voice")
_DEFAULT_MODEL_PATH = os.path.join(_HEAD_CHEST_VOICE_DIR, "vocal_register_svm.pkl")

TARGET_SR = 22050
MIN_SAMPLES = 1024


def _resize_numpy(img: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize 2D array to (height, width) using nearest-neighbor."""
    h, w = img.shape
    yi = np.clip(np.linspace(0, h - 1, height).astype(np.intp), 0, h - 1)
    xi = np.clip(np.linspace(0, w - 1, width).astype(np.intp), 0, w - 1)
    return img[np.ix_(yi, xi)]


class ChestHeadFalsettoPredictor:
    """
    Two-stage predictor:
    1) Chest vs Head/Falsetto (trained SVM from Head_Chest_Voice).
    2) Head vs Falsetto (acoustic-feature discriminator).
    """

    def __init__(self, model_path: Optional[str] = None):
        model_path = model_path or _DEFAULT_MODEL_PATH
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Vocal register model not found: {model_path}")

        with open(model_path, "rb") as f:
            data = pickle.load(f)

        self.svm = data["svm"]
        self.scaler = data["scaler"]
        self.img_height = int(data["img_height"])
        self.img_width = int(data["img_width"])

        # Head vs Falsetto: feature scaler and classifier (trained on prototype features)
        self._head_falsetto_scaler = StandardScaler()
        self._head_falsetto_clf = LogisticRegression(max_iter=500, random_state=42)
        self._fit_head_falsetto_prior()

    def _fit_head_falsetto_prior(self) -> None:
        """
        Fit head-vs-falsetto classifier on prototype feature vectors.
        Head: lower spectral flatness (clearer), higher rms, lower centroid.
        Falsetto: higher flatness (breathier), lower rms, higher centroid.
        Replace with real labeled data when available.
        """
        np.random.seed(42)
        # Feature order: spectral_flatness_mean, rms_mean, spectral_centroid_mean,
        #                spectral_rolloff_mean, spectral_bandwidth_mean, zcr_mean
        head_prototype = np.array([0.15, 0.6, 0.4, 0.45, 0.5, 0.35])
        falsetto_prototype = np.array([0.55, 0.25, 0.65, 0.7, 0.6, 0.45])
        head_points = head_prototype + np.random.randn(40, 6) * 0.08
        falsetto_points = falsetto_prototype + np.random.randn(40, 6) * 0.08
        X = np.vstack([head_points, falsetto_points])
        y = np.array([0] * 40 + [1] * 40)  # 0 = head, 1 = falsetto
        self._head_falsetto_scaler.fit(X)
        Xs = self._head_falsetto_scaler.transform(X)
        self._head_falsetto_clf.fit(Xs, y)

    def _audio_to_melspec(self, audio: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
        mel = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_mels=128, fmax=8000
        )
        return librosa.power_to_db(mel, ref=np.max)

    def _preprocess_spectrogram(self, mel_db: np.ndarray) -> np.ndarray:
        mel_norm = (
            (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8) * 255
        ).astype(np.uint8)
        if _HAS_CV2:
            return cv2.resize(mel_norm, (self.img_width, self.img_height))
        return _resize_numpy(mel_norm, self.img_width, self.img_height).astype(np.uint8)

    def _extract_head_falsetto_features(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Features that tend to differ between head (clearer) and falsetto (breathier)."""
        flatness = librosa.feature.spectral_flatness(y=audio)
        rms = librosa.feature.rms(y=audio)
        cent = librosa.feature.spectral_centroid(y=audio, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
        bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(audio)

        # Normalize to 0–1 scale per feature (using typical ranges)
        flatness_n = np.clip(np.median(flatness), 0, 1)
        rms_n = np.clip(np.median(rms) * 10, 0, 1)
        cent_n = np.clip(np.median(cent) / 4000, 0, 1)
        rolloff_n = np.clip(np.median(rolloff) / 8000, 0, 1)
        bandwidth_n = np.clip(np.median(bandwidth) / 4000, 0, 1)
        zcr_n = np.clip(np.median(zcr) * 100, 0, 1)

        return np.array(
            [[flatness_n, rms_n, cent_n, rolloff_n, bandwidth_n, zcr_n]],
            dtype=np.float64,
        )

    def predict(self, audio: np.ndarray, sr: int = TARGET_SR) -> Dict[str, Any]:
        """
        Predict vocal register: chest, head, or falsetto.

        Returns dict with:
          - label: "Chest Voice" | "Head Voice" | "Falsetto"
          - confidence: float
          - chest_probability, head_probability, falsetto_probability
          - prediction_code: 0=chest, 1=head, 2=falsetto
        """
        audio = np.asarray(audio, dtype=np.float64)
        if sr != TARGET_SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
            sr = TARGET_SR
        if len(audio) < MIN_SAMPLES:
            raise ValueError(
                f"Audio too short: {len(audio)} samples (need at least {MIN_SAMPLES})"
            )

        # Stage 1: Chest vs Head/Falsetto
        mel = self._audio_to_melspec(audio, sr)
        img = self._preprocess_spectrogram(mel)
        features = img.flatten().reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        probs_2 = self.svm.predict_proba(features_scaled)[0]
        chest_prob = float(probs_2[0])
        head_falsetto_prob = float(probs_2[1])

        if chest_prob >= head_falsetto_prob:
            return {
                "label": "Chest Voice",
                "confidence": float(chest_prob),
                "chest_probability": chest_prob,
                "head_probability": 0.0,
                "falsetto_probability": 0.0,
                "prediction_code": 0,
            }

        # Stage 2: Head vs Falsetto
        hf_features = self._extract_head_falsetto_features(audio, sr)
        hf_scaled = self._head_falsetto_scaler.transform(hf_features)
        probs_hf = self._head_falsetto_clf.predict_proba(hf_scaled)[0]
        head_prob = float(probs_hf[0])
        falsetto_prob = float(probs_hf[1])
        # Scale by stage-1 probability so that chest+head+falsetto sum sensibly
        head_prob *= head_falsetto_prob
        falsetto_prob *= head_falsetto_prob

        winner = 1 if head_prob >= falsetto_prob else 2
        label = "Head Voice" if winner == 1 else "Falsetto"
        confidence = head_prob if winner == 1 else falsetto_prob

        return {
            "label": label,
            "confidence": float(confidence),
            "chest_probability": chest_prob,
            "head_probability": head_prob,
            "falsetto_probability": falsetto_prob,
            "prediction_code": winner,
        }


# Lazy-loaded singleton
_predictor: Optional[ChestHeadFalsettoPredictor] = None


def get_predictor() -> Optional[ChestHeadFalsettoPredictor]:
    global _predictor
    if _predictor is None:
        try:
            _predictor = ChestHeadFalsettoPredictor()
        except Exception as e:
            print(f"Vocal register model not loaded: {e}")
    return _predictor


def predict_realtime_from_base64(
    audio_base64: str, sample_rate: int = TARGET_SR
) -> Dict[str, Any]:
    """
    Decode base64 float32 audio and return chest/head/falsetto prediction.
    """
    raw = base64.b64decode(audio_base64)
    audio = np.frombuffer(raw, dtype=np.float32)
    predictor = get_predictor()
    if predictor is None:
        return {
            "success": False,
            "error": "Vocal register model not loaded",
        }
    if len(audio) < MIN_SAMPLES:
        return {
            "success": False,
            "error": f"Audio too short: {len(audio)} samples (need at least {MIN_SAMPLES})",
        }
    try:
        result = predictor.predict(audio, sr=sample_rate)
        return {"success": True, "prediction": result}
    except Exception as e:
        return {"success": False, "error": str(e)}

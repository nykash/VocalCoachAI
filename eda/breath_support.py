"""
Breath support score from audio (file or live).

Acoustic correlates of good breath support:
- Stable amplitude, strong harmonics, low noise, smooth onset, stable pitch, sustained energy.

Features: HNR, spectral tilt, RMS stability, decay rate, pitch stability, onset breathiness.
Output: 0–100 breath support score + per-feature breakdown.

Usage (use the project venv):
  cd eda && source venv/bin/activate
  python breath_support.py path/to/audio.mp3   # or .wav, .mp4
  python breath_support.py --live              # real-time microphone
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
import tempfile
import threading
import time
from collections import deque

import numpy as np

try:
    import librosa
except ImportError:
    raise ImportError("breath_support requires librosa. Install: pip install librosa")

# Optional: Parselmouth (Praat) for accurate HNR and jitter
try:
    import parselmouth
    from parselmouth.praat import call as praat_call
    PRAAT_AVAILABLE = True
except ImportError:
    PRAAT_AVAILABLE = False

# Optional: live audio
try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants (tune for singing; match analyze.py where relevant)
# ---------------------------------------------------------------------------
SR = 22050
FRAME_MS = 30
HOP_MS = 10
N_FFT = 2048
FMIN = 80.0
FMAX = 1000.0
ONSET_MS = 200
MIN_PHRASE_DURATION_SEC = 0.3
SILENCE_THRESH_DB = -40
DECAY_FIT_MIN_DURATION_SEC = 0.2
# Live mode: only show score when chunk has at least this much energy (RMS)
MIN_ENERGY_RMS_FOR_LIVE = 0.001
# Calibration: minimum voiced chunk duration (sec) to extract features
MIN_CHUNK_DURATION_CALIBRATION = 1.0

# Score: each feature contributes weight * (normalized - 0.5); all weights positive (higher norm = better).
# Sum of weights ~40 so raw in [-20, 20]; score = 50 + raw * 2.5 -> [0, 100].
W_HNR = 6.0
W_RMS_STABILITY = 8.0
W_PITCH_STABILITY = 6.0
W_TILT = 4.0
W_DECAY = 4.0
W_ONSET = 4.0
W_ACI = 6.0


def _frame_length_hop(sr: int):
    frame_length = int(FRAME_MS * sr / 1000)
    hop_length = int(HOP_MS * sr / 1000)
    return frame_length, hop_length


def load_audio(path: str, sr: int = SR, mono: bool = True):
    """Load audio from file. Supports wav, mp3, m4a, flac; for mp4 uses pydub if needed."""
    ext = (os.path.splitext(path)[1] or "").lower()
    if ext == ".mp4":
        try:
            y, sr_out = librosa.load(path, sr=sr, mono=mono)
            return y, sr_out
        except Exception:
            try:
                from pydub import AudioSegment
                seg = AudioSegment.from_file(path, format="mp4")
                seg = seg.set_channels(1)
                seg = seg.set_frame_rate(sr)
                samples = np.array(seg.get_array_of_samples(), dtype=np.float32) / (1 << 15)
                return samples, sr
            except ImportError:
                # Fallback: write wav via ffmpeg then load
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    wav_path = f.name
                try:
                    import subprocess
                    subprocess.run([
                        "ffmpeg", "-y", "-i", path, "-vn", "-acodec", "pcm_s16le",
                        "-ar", str(sr), "-ac", "1", wav_path
                    ], check=True, capture_output=True)
                    y, sr_out = librosa.load(wav_path, sr=sr, mono=True)
                    return y, sr_out
                finally:
                    try:
                        os.unlink(wav_path)
                    except OSError:
                        pass
    y, sr_out = librosa.load(path, sr=sr, mono=mono)
    return y, sr_out


# ---------------------------------------------------------------------------
# (A) Harmonic-to-Noise Ratio
# ---------------------------------------------------------------------------
def _hnr_parselmouth(y: np.ndarray, sr: int, frame_length_ms: float = 25.0):
    """Per-frame HNR using Praat (parselmouth). Returns times and HNR in dB."""
    sound = parselmouth.Sound(y, sampling_frequency=sr)
    point_process = praat_call(sound, "To PointProcess (periodic, cc)", 75, 500)
    hnr = praat_call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    # Get HNR per frame (Praat uses 0.01 s step by default)
    num_frames = hnr.get_number_of_frames()
    times = [hnr.get_time_from_frame_number(i + 1) for i in range(num_frames)]
    values = [hnr.get_value_in_frame(i + 1) for i in range(num_frames)]
    # Praat returns HNR in dB; NaN where unvoiced
    return np.array(times), np.array(values)


def _hnr_autocorr(y: np.ndarray, sr: int, frame_length: int, hop_length: int):
    """
    Fallback HNR via autocorrelation: harmonic energy ≈ peak at period, noise ≈ rest.
    Returns per-frame HNR in dB (one value per hop).
    """
    n_frames = 1 + (len(y) - frame_length) // hop_length
    hnr_db = np.full(n_frames, np.nan)
    for i in range(n_frames):
        start = i * hop_length
        frame = y[start : start + frame_length]
        if np.all(frame == 0):
            continue
        frame = frame - np.mean(frame)
        corr = np.correlate(frame, frame, mode="full")
        corr = corr[len(corr) // 2 :]
        if corr[0] <= 0:
            continue
        corr = corr / corr[0]
        # Find first peak after lag corresponding to FMAX (min period)
        min_lag = max(1, int(sr / FMAX))
        max_lag = min(len(corr) - 1, int(sr / FMIN))
        if max_lag <= min_lag:
            continue
        peak_idx = min_lag + np.argmax(corr[min_lag : max_lag + 1])
        period = peak_idx
        harmonic_energy = 0.0
        for k in range(1, min(20, len(corr) // max(1, period))):
            idx = k * period
            if idx < len(corr):
                harmonic_energy += corr[idx] ** 2
        # Noise: total energy minus harmonic part (simplified: use 1 - peak)
        E_total = 1.0
        E_harmonic = min(1.0, harmonic_energy * 0.5)
        E_noise = max(1e-10, E_total - E_harmonic)
        hnr_db[i] = 10 * math.log10(E_harmonic / E_noise + 1e-10)
    return hnr_db


def compute_hnr(y: np.ndarray, sr: int):
    """Return (times or frame indices), (HNR array in dB), and mean HNR (voiced only)."""
    frame_length, hop_length = _frame_length_hop(sr)
    if PRAAT_AVAILABLE:
        try:
            times, hnr = _hnr_parselmouth(y, sr, frame_length_ms=FRAME_MS)
            valid = np.isfinite(hnr) & (hnr > -50)
            mean_hnr = float(np.mean(hnr[valid])) if np.any(valid) else np.nan
            return times, hnr, mean_hnr
        except Exception:
            pass
    hnr = _hnr_autocorr(y, sr, frame_length, hop_length)
    times = np.arange(len(hnr)) * (hop_length / sr)
    valid = np.isfinite(hnr) & (hnr > -50)
    mean_hnr = float(np.mean(hnr[valid])) if np.any(valid) else np.nan
    return times, hnr, mean_hnr


# ---------------------------------------------------------------------------
# (B) Spectral Tilt (log magnitude vs log freq slope)
# ---------------------------------------------------------------------------
def compute_spectral_tilt(y: np.ndarray, sr: int):
    """Per-frame slope of log(magnitude) vs log(freq). Steep negative = good support."""
    frame_length, hop_length = _frame_length_hop(sr)
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=hop_length, win_length=frame_length))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    log_f = np.log(freqs + 1e-6)
    tilts = []
    for frame in S.T:
        mag = frame + 1e-10
        log_mag = np.log(mag)
        slope = np.polyfit(log_f, log_mag, 1)[0]
        tilts.append(slope)
    tilts = np.array(tilts)
    return tilts, float(np.nanmean(tilts))


# ---------------------------------------------------------------------------
# (C) Amplitude stability (RMS CV)
# ---------------------------------------------------------------------------
def compute_rms_stability(y: np.ndarray, sr: int):
    """RMS per frame; return CV = std/mean (lower = better breath control)."""
    frame_length, hop_length = _frame_length_hop(sr)
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    rms = rms[rms > 1e-10]
    if len(rms) == 0:
        return np.array([]), np.nan
    cv = float(np.std(rms) / (np.mean(rms) + 1e-10))
    return rms, cv


# ---------------------------------------------------------------------------
# (D) Sustained note decay rate
# ---------------------------------------------------------------------------
def _segment_voiced_regions(y: np.ndarray, sr: int, frame_length: int, hop_length: int, thresh_db: float):
    """Simple energy-based voiced segments (start, end) in seconds."""
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    ref = np.max(rms) + 1e-10
    db = 20 * np.log10(rms / ref + 1e-10)
    voiced = db > thresh_db
    segments = []
    in_seg = False
    start = 0.0
    for i, t in enumerate(times):
        if voiced[i] and not in_seg:
            in_seg = True
            start = t
        elif not voiced[i] and in_seg:
            in_seg = False
            if t - start >= MIN_PHRASE_DURATION_SEC:
                segments.append((start, t))
    if in_seg and times[-1] - start >= MIN_PHRASE_DURATION_SEC:
        segments.append((start, times[-1]))
    return segments


def compute_decay_rate(y: np.ndarray, sr: int):
    """
    Fit A(t) = A0 * exp(-k*t) to amplitude envelope in sustained segments.
    Return mean k (higher = faster decay = worse support).
    """
    frame_length, hop_length = _frame_length_hop(sr)
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    segments = _segment_voiced_regions(y, sr, frame_length, hop_length, SILENCE_THRESH_DB)
    decay_rates = []
    min_frames = max(3, int(DECAY_FIT_MIN_DURATION_SEC * sr / hop_length))
    for s0, s1 in segments:
        idx = (times >= s0) & (times <= s1)
        t_seg = times[idx]
        r_seg = rms[idx]
        if len(t_seg) < min_frames or np.max(r_seg) <= 0:
            continue
        t_rel = t_seg - t_seg[0]
        log_r = np.log(r_seg + 1e-10)
        try:
            slope, _ = np.polyfit(t_rel, log_r, 1)
            k = -float(slope)
            if k > 0 and k < 50:
                decay_rates.append(k)
        except Exception:
            pass
    mean_decay = float(np.mean(decay_rates)) if decay_rates else 0.0
    return decay_rates, mean_decay


# ---------------------------------------------------------------------------
# (E) Pitch stability (jitter / variance)
# ---------------------------------------------------------------------------
def compute_pitch_stability(y: np.ndarray, sr: int):
    """F0 variance and optional jitter (parselmouth). Lower = better support."""
    frame_length, hop_length = _frame_length_hop(sr)
    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=FMIN, fmax=FMAX,
        hop_length=hop_length, frame_length=frame_length
    )
    voiced = np.isfinite(f0) & voiced_flag
    f0_voiced = f0[voiced]
    pitch_var = float(np.nanvar(f0_voiced)) if np.sum(voiced) > 2 else np.nan
    jitter = np.nan
    if PRAAT_AVAILABLE and np.any(voiced):
        try:
            sound = parselmouth.Sound(y, sampling_frequency=sr)
            point_process = praat_call(sound, "To PointProcess (periodic, cc)", 75, 500)
            jitter = praat_call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        except Exception:
            pass
    return f0, voiced_flag, pitch_var, jitter


# ---------------------------------------------------------------------------
# (F) Onset breathiness (first ~200 ms)
# ---------------------------------------------------------------------------
def compute_onset_breathiness(y: np.ndarray, sr: int):
    """HNR and noise in first ONSET_MS ms of phonation (after silence)."""
    frame_length, hop_length = _frame_length_hop(sr)
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    ref = np.max(rms) + 1e-10
    db = 20 * np.log10(rms / ref + 1e-10)
    # Find first sustained onset
    onset_frames = np.where(db > SILENCE_THRESH_DB)[0]
    if len(onset_frames) == 0:
        return np.nan, np.nan
    start_frame = onset_frames[0]
    start_time = times[start_frame]
    onset_duration_sec = ONSET_MS / 1000.0
    end_time = start_time + onset_duration_sec
    end_sample = min(len(y), int((end_time + 0.1) * sr))
    start_sample = int(start_time * sr)
    y_onset = y[max(0, start_sample) : end_sample]
    if len(y_onset) < frame_length:
        return np.nan, np.nan
    _, _, hnr_onset = compute_hnr(y_onset, sr)
    # Tilt in onset (flat = breathy)
    tilts, tilt_onset = compute_spectral_tilt(y_onset, sr)
    return hnr_onset, tilt_onset


# ---------------------------------------------------------------------------
# Phrase segmentation and phrase-level metrics
# ---------------------------------------------------------------------------
def get_phrase_segments(y: np.ndarray, sr: int):
    """Return list of (start_sec, end_sec) for each phrase (silence-segmented)."""
    frame_length, hop_length = _frame_length_hop(sr)
    segments = _segment_voiced_regions(y, sr, frame_length, hop_length, SILENCE_THRESH_DB)
    return segments


def compute_phrase_level_scores(y: np.ndarray, sr: int, segments: list):
    """
    Per phrase: energy slope, pitch sag, noise increase.
    Return mean energy slope (negative = decay), mean pitch sag, mean noise increase.
    """
    frame_length, hop_length = _frame_length_hop(sr)
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    f0, voiced_flag, _, _ = compute_pitch_stability(y, sr)
    energy_slopes = []
    pitch_sags = []
    for s0, s1 in segments:
        idx = (times >= s0) & (times <= s1)
        if np.sum(idx) < 3:
            continue
        t_seg = times[idx]
        r_seg = rms[idx]
        f_seg = f0[idx]
        try:
            slope_r = np.polyfit(t_seg - t_seg[0], r_seg, 1)[0]
            energy_slopes.append(slope_r)
        except Exception:
            pass
        f_voiced = f_seg[np.isfinite(f_seg)]
        if len(f_voiced) >= 2:
            pitch_sag = float(f_voiced[0] - f_voiced[-1])
            pitch_sags.append(pitch_sag)
    mean_energy_slope = float(np.mean(energy_slopes)) if energy_slopes else 0.0
    mean_pitch_sag = float(np.mean(pitch_sags)) if pitch_sags else 0.0
    return mean_energy_slope, mean_pitch_sag


# ---------------------------------------------------------------------------
# Airflow Consistency Index (ACI)
# ---------------------------------------------------------------------------
def compute_aci(y: np.ndarray, sr: int):
    """
    ACI = 1 - normalized variance of (RMS, HNR, spectral tilt) over time.
    Stable airflow -> low variance -> high ACI.
    """
    frame_length, hop_length = _frame_length_hop(sr)
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    tilts, _ = compute_spectral_tilt(y, sr)
    _, hnr_frames, _ = compute_hnr(y, sr)
    # Align lengths (rms and tilts are per STFT frame; HNR may be different length)
    n = min(len(rms), len(tilts))
    rms = rms[:n]
    tilts = tilts[:n]
    if len(hnr_frames) >= n:
        hnr_frames = hnr_frames[:n]
    else:
        if len(hnr_frames) > 0:
            x_old = np.linspace(0, 1, len(hnr_frames))
            x_new = np.linspace(0, 1, n)
            hnr_frames = np.interp(x_new, x_old, np.nan_to_num(hnr_frames, nan=0.0))
        else:
            hnr_frames = np.zeros(n)
    hnr_frames = np.nan_to_num(hnr_frames, nan=0.0)
    var_rms = np.var(rms)
    var_tilt = np.var(tilts)
    var_hnr = np.var(hnr_frames)
    scale_rms = (np.mean(rms) ** 2 + 1e-10)
    scale_tilt = 1.0
    scale_hnr = 100.0
    norm_var = (var_rms / scale_rms + var_tilt * scale_tilt + var_hnr / scale_hnr) / 3.0
    aci = 1.0 - min(1.0, norm_var)
    return float(max(0, aci))


# ---------------------------------------------------------------------------
# Combined breath support score (0–100)
# ---------------------------------------------------------------------------
def _normalize_hnr(hnr_db: float) -> float:
    """Map HNR (dB) to 0–1. ~20+ = 1, 10–20 = 0.5–1, <10 = 0–0.5."""
    if np.isnan(hnr_db):
        return 0.5
    if hnr_db >= 20:
        return 1.0
    if hnr_db <= 0:
        return 0.0
    return min(1.0, hnr_db / 20.0)


def _normalize_cv(cv: float) -> float:
    """Lower CV is better. CV 0 = 1, CV 1.5+ = 0."""
    if np.isnan(cv) or cv < 0:
        return 0.5
    return max(0, 1.0 - min(cv, 1.5) / 1.5)


def _normalize_pitch_var(pitch_var: float) -> float:
    """Lower variance = better. Log scale so wide range (e.g. 50–50000 Hz²) maps smoothly to 1–0."""
    if np.isnan(pitch_var) or pitch_var < 0:
        return 0.5
    # log(1 + x): 0 -> 0, 100 -> 4.6, 5000 -> 8.5, 50000 -> 10.8. Map 100->0.9, 50000->0.
    x = np.log1p(min(pitch_var, 100000.0))
    return max(0, 1.0 - x / 11.0)


def _normalize_tilt(tilt: float) -> float:
    """Steep negative = good. Tilt typically -2 to +1. Map -2 -> 1, +1 -> 0."""
    if np.isnan(tilt):
        return 0.5
    return max(0, min(1.0, (1.0 - tilt) / 3.0))


def _normalize_decay(k: float) -> float:
    """Lower k = better. k 0 = 1, k 10 = 0."""
    if np.isnan(k) or k < 0:
        return 0.5
    return max(0, 1.0 - min(k / 10.0, 1.0))


# Raw feature keys used for calibration (same as in compute_breath_support_score)
RAW_FEATURE_KEYS = [
    "hnr_db",
    "spectral_tilt",
    "rms_cv",
    "decay_k",
    "pitch_variance",
    "onset_hnr_db",
    "aci",
]
# For each key: True = higher is better (good support), False = lower is better
RAW_FEATURE_HIGHER_IS_BETTER = {
    "hnr_db": True,
    "spectral_tilt": False,  # steeper negative = better, so lower tilt value = better
    "rms_cv": False,
    "decay_k": False,
    "pitch_variance": False,
    "onset_hnr_db": True,
    "aci": True,
}


def compute_raw_features(y: np.ndarray, sr: int) -> dict:
    """Compute only raw breath-support features (no normalizations, no score). For calibration."""
    _, _, mean_hnr = compute_hnr(y, sr)
    _, mean_tilt = compute_spectral_tilt(y, sr)
    _, rms_cv = compute_rms_stability(y, sr)
    _, mean_decay_k = compute_decay_rate(y, sr)
    _, _, pitch_var, _ = compute_pitch_stability(y, sr)
    onset_hnr, _ = compute_onset_breathiness(y, sr)
    aci = compute_aci(y, sr)
    return {
        "hnr_db": mean_hnr,
        "spectral_tilt": mean_tilt,
        "rms_cv": rms_cv,
        "decay_k": mean_decay_k,
        "pitch_variance": pitch_var,
        "onset_hnr_db": onset_hnr,
        "aci": aci,
    }


def load_calibration(path: str | None = None) -> dict | None:
    """Load calibration dict from JSON. Default path: eda/output/breath_support_calibration.json."""
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "output", "breath_support_calibration.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def get_voiced_chunk_segments(y: np.ndarray, sr: int, min_duration_sec: float = MIN_CHUNK_DURATION_CALIBRATION):
    """Return list of (start_sec, end_sec) for voiced segments with duration >= min_duration_sec."""
    segments = get_phrase_segments(y, sr)
    return [(s0, s1) for s0, s1 in segments if (s1 - s0) >= min_duration_sec]


def apply_calibration(raw_features: dict, calibration: dict) -> float:
    """
    Map raw features to a score in [0, 1] using calibration.
    ref_high = data (good) reference, ref_low = test (poor) reference.
    n = (val - ref_low) / (ref_high - ref_low) so data ~ 1, test ~ 0.
    """
    scores = []
    for key in RAW_FEATURE_KEYS:
        if key not in calibration:
            continue
        val = raw_features.get(key)
        if val is None or (isinstance(val, float) and not np.isfinite(val)):
            continue
        high = calibration[key]["ref_high"]
        low = calibration[key]["ref_low"]
        denom = high - low
        if abs(denom) < 1e-10:
            continue
        n = (val - low) / denom
        n = max(0.0, min(1.0, n))
        scores.append(n)
    return float(np.mean(scores)) if scores else 0.5


def compute_breath_support_score(y: np.ndarray, sr: int, calibration: dict | None = None) -> dict:
    """
    Single function to compute all features and combined 0–100 breath support score.
    If calibration is provided, also adds calibrated_score in [0, 1] (data songs ~ 1, test ~ 0).
    """
    out = {}
    # (A) HNR
    _, hnr_frames, mean_hnr = compute_hnr(y, sr)
    out["hnr_db"] = mean_hnr
    out["hnr_normalized"] = _normalize_hnr(mean_hnr)

    # (B) Spectral tilt
    tilts, mean_tilt = compute_spectral_tilt(y, sr)
    out["spectral_tilt"] = mean_tilt
    out["spectral_tilt_normalized"] = _normalize_tilt(mean_tilt)

    # (C) RMS stability
    rms_frames, rms_cv = compute_rms_stability(y, sr)
    out["rms_cv"] = rms_cv
    out["rms_stability_normalized"] = _normalize_cv(rms_cv)

    # (D) Decay
    decay_list, mean_decay_k = compute_decay_rate(y, sr)
    out["decay_k"] = mean_decay_k
    out["decay_normalized"] = _normalize_decay(mean_decay_k)

    # (E) Pitch
    f0, voiced_flag, pitch_var, jitter = compute_pitch_stability(y, sr)
    out["pitch_variance"] = pitch_var
    out["jitter"] = jitter
    out["pitch_stability_normalized"] = _normalize_pitch_var(pitch_var)

    # (F) Onset
    onset_hnr, onset_tilt = compute_onset_breathiness(y, sr)
    out["onset_hnr_db"] = onset_hnr
    out["onset_tilt"] = onset_tilt
    out["onset_normalized"] = _normalize_hnr(onset_hnr) if np.isfinite(onset_hnr) else 0.5

    # ACI
    out["aci"] = compute_aci(y, sr)

    # Phrase-level
    segments = get_phrase_segments(y, sr)
    out["n_phrases"] = len(segments)
    energy_slope, pitch_sag = compute_phrase_level_scores(y, sr, segments)
    out["phrase_energy_slope"] = energy_slope
    out["phrase_pitch_sag"] = pitch_sag

    # Combined score: each feature weight * (normalized - 0.5); all normalized in [0,1], so raw in [-20, 20]
    raw = (
        W_HNR * (out["hnr_normalized"] - 0.5)
        + W_RMS_STABILITY * (out["rms_stability_normalized"] - 0.5)
        + W_PITCH_STABILITY * (out["pitch_stability_normalized"] - 0.5)
        + W_TILT * (out["spectral_tilt_normalized"] - 0.5)
        + W_DECAY * (out["decay_normalized"] - 0.5)
        + W_ONSET * (out["onset_normalized"] - 0.5)
        + W_ACI * (out["aci"] - 0.5)
    )
    # Map raw (approx -20 to +20) to 0–100
    score = 50.0 + raw * 2.5
    score = max(0.0, min(100.0, score))
    out["breath_support_score"] = round(score, 1)
    if calibration:
        raw_features = {k: out.get(k) for k in RAW_FEATURE_KEYS if k in out}
        out["calibrated_score"] = round(apply_calibration(raw_features, calibration), 3)
    return out


# ---------------------------------------------------------------------------
# Live / real-time mode
# ---------------------------------------------------------------------------
def _fmt(x, fmt_str: str = ".1f"):
    """Format value for display; use '—' when nan."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    if isinstance(x, float):
        return format(x, fmt_str)
    return str(x)


def _print_input_devices():
    """Print default and all input devices for mic selection."""
    if not SOUNDDEVICE_AVAILABLE:
        print("sounddevice not installed", file=sys.stderr)
        return
    default = sd.query_devices(sd.default.device[0])
    print("Default input device:", sd.default.device[0], "-", default["name"])
    print("\nAll input devices (use --device N to pick one):")
    print(sd.query_devices(kind="input"))


def run_live_score(
    device: int | None = None,
    sr: int = SR,
    chunk_duration_sec: float = 3.0,
    callback_interval_sec: float = 1.0,
    score_callback=None,
    min_energy_rms: float = MIN_ENERGY_RMS_FOR_LIVE,
    input_gain: float = 1.0,
):
    """
    Stream microphone audio, compute breath support on sliding windows, call score_callback(result_dict).
    Only reports a score when chunk energy is above min_energy_rms (avoids nan when quiet).
    input_gain: multiply captured audio by this (e.g. 5 for quiet mics).
    """
    if not SOUNDDEVICE_AVAILABLE:
        raise RuntimeError("Live mode requires sounddevice. Install: pip install sounddevice")
    dev_idx = device if device is not None else sd.default.device[0]
    dev = sd.query_devices(dev_idx)
    # Use device's native sample rate to avoid silence (e.g. MacBook mic is 44100, we were using 22050)
    stream_sr = int(dev.get("default_samplerate", sr))
    if stream_sr <= 0:
        stream_sr = sr
    chunk_samples = int(stream_sr * chunk_duration_sec)
    buffer = deque(maxlen=chunk_samples * 2)

    print(f"Using input device: {dev_idx} - {dev['name']} @ {stream_sr} Hz", flush=True)
    print("(Run with --list-devices to see all inputs; use --device N to switch.)", flush=True)

    def stream_callback(indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        buffer.extend(indata[:, 0].tolist())

    stream = sd.InputStream(
        device=device,
        channels=1,
        samplerate=stream_sr,
        blocksize=int(stream_sr * 0.05),
        callback=stream_callback,
        dtype="float32",
    )
    stream.start()
    no_signal_count = 0
    try:
        while True:
            time.sleep(callback_interval_sec)
            if len(buffer) < chunk_samples:
                print("Listening... (buffer filling)", flush=True)
                continue
            y = np.array(list(buffer)[-chunk_samples:], dtype=np.float32)
            if input_gain != 1.0:
                y = (y * input_gain).astype(np.float32)
            max_rms = float(np.sqrt(np.mean(y ** 2)))
            max_abs = float(np.max(np.abs(y)))
            # No signal at all: wrong device or no mic permission
            if max_rms == 0 and max_abs == 0:
                no_signal_count += 1
                if no_signal_count == 1 or no_signal_count % 5 == 0:
                    print(
                        "No audio detected. Check: 1) Mic permission (System Settings → Privacy) "
                        "2) Correct device: run with --list-devices then --device N",
                        flush=True,
                    )
                continue
            no_signal_count = 0
            if max_rms < min_energy_rms:
                print(
                    f"Listening... (sing to see score)  [level: {max_rms:.4f}]",
                    flush=True,
                )
                continue
            result = compute_breath_support_score(y, stream_sr)
            if score_callback:
                score_callback(result)
            else:
                print(
                    f"Breath support: {result['breath_support_score']:.1f}/100 "
                    f"(HNR={_fmt(result['hnr_db'])} dB, RMS CV={_fmt(result['rms_cv'], '.2f')})",
                    flush=True,
                )
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop()
        stream.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Breath support score from audio (file or live)")
    ap.add_argument("input", nargs="?", help="Audio or video file (wav, mp3, mp4, etc.). Omit for live mic.")
    ap.add_argument("--live", action="store_true", help="Use live microphone input")
    ap.add_argument("--device", type=int, default=None, help="Sounddevice input device index (live only)")
    ap.add_argument("--chunk", type=float, default=3.0, help="Live chunk duration in seconds")
    ap.add_argument("--interval", type=float, default=1.0, help="Live callback interval in seconds")
    ap.add_argument("--sr", type=int, default=SR, help="Sample rate")
    ap.add_argument("--gain", type=float, default=1.0, help="Live only: multiply input by this (e.g. 5 for quiet mics)")
    ap.add_argument("--min-level", type=float, default=MIN_ENERGY_RMS_FOR_LIVE, help="Live only: min RMS to show score (default %s)" % MIN_ENERGY_RMS_FOR_LIVE)
    ap.add_argument("--list-devices", action="store_true", help="List sounddevice input devices and exit (use with --live)")
    args = ap.parse_args()

    if args.list_devices:
        _print_input_devices()
        return

    if args.live:
        run_live_score(
            device=args.device,
            sr=args.sr,
            chunk_duration_sec=args.chunk,
            callback_interval_sec=args.interval,
            min_energy_rms=args.min_level,
            input_gain=args.gain,
        )
        return

    if not args.input:
        ap.error("Provide an input file or use --live for microphone")
        return

    calibration = load_calibration()
    y, sr = load_audio(args.input, sr=args.sr)
    result = compute_breath_support_score(y, sr, calibration=calibration)
    print("Breath support score:", result["breath_support_score"], "/ 100")
    if calibration and "calibrated_score" in result:
        print("Calibrated score (0–1, data~1 test~0):", result["calibrated_score"])
    print("HNR (dB):", result["hnr_db"])
    print("Spectral tilt:", result["spectral_tilt"])
    print("RMS CV:", result["rms_cv"])
    print("Decay k:", result["decay_k"])
    print("Pitch variance:", result["pitch_variance"])
    print("Onset HNR (dB):", result["onset_hnr_db"])
    print("ACI:", result["aci"])
    print("Phrases:", result["n_phrases"])
    print("Phrase energy slope:", result["phrase_energy_slope"])
    print("Phrase pitch sag:", result["phrase_pitch_sag"])


if __name__ == "__main__":
    main()

"""
Live singer demo backend. Serves the frontend and accepts audio via POST /api/analyze.
Computes mel spectrogram from audio, encodes with the trained VAE, and returns
artist probabilities and singing-style tags (vocal similarity) like predict_singer_vae.py.

Expects: body = raw Float32Array buffer, header X-Sample-Rate (e.g. 44100).
Returns: JSON with artists (top probs), attributes (tag confidence), and optional mel info.
"""

from __future__ import annotations

import json
import os
import pickle
import sys

import numpy as np

# Allow importing from parent eda folder (train_vae_singer, vae_singer_config, etc.)
_EDA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _EDA_DIR not in sys.path:
    sys.path.insert(0, _EDA_DIR)

from flask import Flask, request, jsonify, send_from_directory
from train_vae_singer import get_feature_chunks_from_buffer

# Use uniform priors in the demo so the top artist is "who does this sound like?"
# (likelihood-based), not "who has the most tracks in the dataset".
USE_UNIFORM_PRIORS = True

# Fixed checkpoint for this demo
DEFAULT_CKPT = os.path.join(
    _EDA_DIR,
    "output", "vae_singer", "checkpoints",
    "vae-epoch=019-val_loss=0.4187.ckpt",
)

app = Flask(__name__, static_folder="static", static_url_path="")

# Model and artifacts (lazy-loaded)
_model_artifacts = None
VAR_EPS = 1e-6


def _num(x):
    if x is None:
        return None
    if isinstance(x, (int, float)) and (x != x or abs(x) == float("inf")):
        return None
    return float(x) if hasattr(x, "__float__") else x


def load_singer_attributes(eda_dir: str):
    path = os.path.join(eda_dir, "singer_attributes.json")
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def log_gaussian_pdf(z: np.ndarray, mean: np.ndarray, var: np.ndarray) -> float:
    d = len(z)
    log_2pi = np.log(2 * np.pi)
    return -0.5 * (d * log_2pi + np.sum(np.log(var)) + np.sum((z - mean) ** 2 / var))


def fit_artist_gaussians(latents: np.ndarray, artists: list):
    unique_artists = sorted(set(artists))
    artist_to_idx = {a: i for i, a in enumerate(unique_artists)}
    n_artists = len(unique_artists)
    latent_dim = latents.shape[1]
    means = np.zeros((n_artists, latent_dim), dtype=np.float64)
    counts = np.zeros(n_artists, dtype=int)
    for i, a in enumerate(artists):
        j = artist_to_idx[a]
        means[j] += latents[i]
        counts[j] += 1
    for j in range(n_artists):
        if counts[j] > 0:
            means[j] /= counts[j]
    vars_ = np.zeros((n_artists, latent_dim), dtype=np.float64)
    for i, a in enumerate(artists):
        j = artist_to_idx[a]
        vars_[j] += (latents[i] - means[j]) ** 2
    for j in range(n_artists):
        if counts[j] > 1:
            vars_[j] /= counts[j] - 1
        vars_[j] = np.maximum(vars_[j], VAR_EPS)
    global_var = np.mean(vars_[counts > 1]) if np.any(counts > 1) else 1.0
    for j in range(n_artists):
        if counts[j] <= 1:
            vars_[j] = global_var
    return unique_artists, means, vars_, counts


def attribute_confidence_from_probs(unique_artists, probs, singer_attributes):
    attr_to_prob = {}
    for idx, artist in enumerate(unique_artists):
        if artist not in singer_attributes:
            continue
        p = probs[idx]
        for attr in singer_attributes[artist]:
            attr_to_prob[attr] = attr_to_prob.get(attr, 0.0) + p
    return sorted(attr_to_prob.items(), key=lambda x: -x[1])


def load_model_artifacts(ckpt_path: str):
    """Load VAE, scaler, config, latents, artists. Uses train_vae_singer.VAE and VAELightningModule."""
    global _model_artifacts
    if _model_artifacts is not None:
        return _model_artifacts

    import torch
    from train_vae_singer import VAE, VAELightningModule

    vae_dir = os.path.dirname(ckpt_path)
    # config.pkl, scaler.pkl, latents.npy, artists.npy live in parent of checkpoints
    vae_dir = os.path.dirname(vae_dir)

    with open(os.path.join(vae_dir, "config.pkl"), "rb") as f:
        config = pickle.load(f)
    with open(os.path.join(vae_dir, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    latents = np.load(os.path.join(vae_dir, "latents.npy"), allow_pickle=True)
    artists = list(np.load(os.path.join(vae_dir, "artists.npy"), allow_pickle=True))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lightning_model = VAELightningModule.load_from_checkpoint(ckpt_path, map_location=device)
    model = lightning_model.vae.to(device)
    model.eval()

    unique_artists, means, vars_, counts = fit_artist_gaussians(latents, artists)
    total_tracks = len(artists)
    if USE_UNIFORM_PRIORS:
        priors = np.ones(len(unique_artists)) / len(unique_artists)
    else:
        priors = np.array([counts[j] / total_tracks for j in range(len(unique_artists))])
    singer_attributes = load_singer_attributes(_EDA_DIR)

    _model_artifacts = {
        "model": model,
        "scaler": scaler,
        "config": config,
        "unique_artists": unique_artists,
        "means": means,
        "vars_": vars_,
        "priors": priors,
        "singer_attributes": singer_attributes,
        "device": device,
    }
    return _model_artifacts




@app.after_request
def cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """Accept raw float32 audio (binary), compute mel, run VAE, return artist probs and tags."""
    try:
        import torch
        import librosa
    except ImportError as e:
        return jsonify({"error": f"Missing dependency: {e}"}), 500

    ckpt_path = os.environ.get("VAE_CKPT", DEFAULT_CKPT)
    if not os.path.isfile(ckpt_path):
        return jsonify({"error": f"Checkpoint not found: {ckpt_path}"}), 500

    try:
        data = request.get_data()
        if not data or len(data) < 1000:
            return jsonify({"error": "Audio too short"}), 400
        sr_in = int(request.headers.get("X-Sample-Rate", 44100))
        if sr_in <= 0 or sr_in > 192000:
            sr_in = 44100
        y = np.frombuffer(data, dtype=np.float32)
        if len(y) < sr_in * 0.5:
            return jsonify({"error": "Need at least 0.5 s of audio"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    try:
        artifacts = load_model_artifacts(ckpt_path)
    except Exception as e:
        return jsonify({"error": f"Failed to load model: {e}"}), 500

    config = artifacts["config"]
    sr = config["sr"]

    # Resample if needed
    if sr_in != sr:
        y = librosa.resample(y.astype(np.float64), orig_sr=sr_in, target_sr=sr).astype(np.float32)

    # Use same feature extraction as training (mel / mfcc / mel_mfcc from config)
    chunk_list = get_feature_chunks_from_buffer(y, sr, config, min_one_chunk=True)
    if not chunk_list:
        chunk_sec = config.get("chunk_sec")
        if chunk_sec is None:
            n_frames = config["n_frames_per_chunk"]
            hop = config["hop_length"]
            chunk_sec = n_frames * hop / sr
        return jsonify({
            "error": f"Audio too short for at least one feature chunk (need ~{chunk_sec:.0f} s at {sr} Hz)"
        }), 400

    chunks = np.stack(chunk_list, axis=0).astype(np.float32)
    scaler = artifacts["scaler"]
    chunks_scaled = scaler.transform(chunks).astype(np.float32)
    model = artifacts["model"]
    device = artifacts["device"]
    unique_artists = artifacts["unique_artists"]
    means = artifacts["means"]
    vars_ = artifacts["vars_"]
    priors = artifacts["priors"]
    singer_attributes = artifacts["singer_attributes"]

    with torch.no_grad():
        x_t = torch.from_numpy(chunks_scaled).to(device)
        mu, _ = model.encode(x_t)
        chunk_latents = mu.cpu().numpy()
    z = np.mean(chunk_latents, axis=0)

    temperature = 5
    n_artists = len(unique_artists)
    log_lik = np.array([log_gaussian_pdf(z, means[j], vars_[j]) for j in range(n_artists)])
    log_post = log_lik + np.log(priors + 1e-10)
    log_post -= np.max(log_post)
    log_post_scaled = log_post / temperature
    probs = np.exp(log_post_scaled)
    probs /= probs.sum()

    # Top artists as list of {name, prob}
    order = np.argsort(probs)[::-1]
    artists_out = [{"name": unique_artists[idx], "prob": float(probs[idx])} for idx in order]

    attr_conf = attribute_confidence_from_probs(unique_artists, probs, singer_attributes)
    attributes_out = [{"tag": attr, "confidence": float(conf)} for attr, conf in attr_conf]

    return jsonify({
        "artists": artists_out,
        "attributes": attributes_out,
        "top_artist": unique_artists[order[0]],
        "top_prob": float(probs[order[0]]),
        "n_chunks": len(chunk_list),
        "sr_used": sr,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8766))
    app.run(host="0.0.0.0", port=port, debug=True, threaded=True)

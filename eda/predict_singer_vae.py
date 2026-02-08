"""
Given a new song (audio file), extract mel spectrogram chunks, encode with the
trained VAE (mean latent over chunks), and output P(artist | song) for each artist
using a Gaussian model in latent space. Run after train_vae_singer.py.
Also reports confidence in singing-style attributes (from singer_attributes.json)
and top 3 attributes given the predicted artist distribution.

Temperature: applied to log-posteriors before softmax; higher = softer distribution.

Usage:
  python predict_singer_vae.py <path_to_audio>
  python predict_singer_vae.py <path_to_audio> [path_to_checkpoint.ckpt]
  python predict_singer_vae.py <path_to_audio> --temperature 5
"""
import argparse
import json
import os
import sys

import numpy as np

_EDA_DIR = os.path.dirname(os.path.abspath(__file__))
if _EDA_DIR not in sys.path:
    sys.path.insert(0, _EDA_DIR)

from vae_singer_config import get_vae_output_dir, get_last_ckpt_path, get_preferred_checkpoint_path
from train_vae_singer import get_feature_chunks_for_path, get_feature_chunks_from_buffer

try:
    import torch
except ImportError:
    print("PyTorch is required. Install with: pip install torch")
    sys.exit(1)

try:
    import librosa
except ImportError:
    print("librosa is required. Install with: pip install librosa")
    sys.exit(1)

VAR_EPS = 1e-6  # minimum variance for numerical stability


def load_audio(path: str, sr: int):
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y


def load_audio_from_bytes(audio_bytes: bytes, sr: int):
    """Load mono audio from raw bytes (e.g. uploaded file)."""
    import io
    y, _ = librosa.load(io.BytesIO(audio_bytes), sr=sr, mono=True)
    return y


def load_vae_artifacts(vae_dir: str, ckpt_path: str = None):
    """Load model, scaler, config, latents, artists from vae_singer dir (mel-chunk VAE).
    If ckpt_path is given, load VAE from that checkpoint; else prefer last.ckpt, then model.pt.
    """
    import pickle
    from train_vae_singer import VAE, VAELightningModule

    with open(os.path.join(vae_dir, "config.pkl"), "rb") as f:
        config = pickle.load(f)
    with open(os.path.join(vae_dir, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    latents = np.load(os.path.join(vae_dir, "latents.npy"), allow_pickle=True)
    artists = list(np.load(os.path.join(vae_dir, "artists.npy"), allow_pickle=True))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_pt = os.path.join(vae_dir, "model.pt")
    preferred_ckpt = get_preferred_checkpoint_path()

    if ckpt_path and os.path.isfile(ckpt_path):
        lightning_model = VAELightningModule.load_from_checkpoint(ckpt_path, map_location=device)
        model = lightning_model.vae.to(device)
        print(f"Loaded VAE from {ckpt_path}")
    elif os.path.isfile(preferred_ckpt):
        lightning_model = VAELightningModule.load_from_checkpoint(preferred_ckpt, map_location=device)
        model = lightning_model.vae.to(device)
        print(f"Loaded VAE from {preferred_ckpt}")
    elif os.path.isfile(model_pt):
        model = VAE(
            config["input_dim"],
            config["latent_dim"],
            config["hidden_dims"],
        ).to(device)
        model.load_state_dict(torch.load(model_pt, map_location=device))
        print(f"Loaded VAE from {model_pt}")
    else:
        raise FileNotFoundError(
            f"No VAE weights found. Run train_vae_singer.py first. Looked for {last_ckpt} and {model_pt}"
        )
    model.eval()

    return model, scaler, config, latents, artists, device


def fit_artist_gaussians(latents: np.ndarray, artists: list):
    """
    For each artist, fit a Gaussian (mean, diagonal variance). Returns
    unique_artists, means (n_artists x latent_dim), vars (n_artists x latent_dim).
    """
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
            vars_[j] /= (counts[j] - 1)
        vars_[j] = np.maximum(vars_[j], VAR_EPS)
    global_var = np.mean(vars_[counts > 1]) if np.any(counts > 1) else 1.0
    for j in range(n_artists):
        if counts[j] <= 1:
            vars_[j] = global_var

    return unique_artists, means, vars_, counts


def log_gaussian_pdf(z: np.ndarray, mean: np.ndarray, var: np.ndarray) -> float:
    """Log of N(z; mean, diag(var))."""
    d = len(z)
    log_2pi = np.log(2 * np.pi)
    return -0.5 * (d * log_2pi + np.sum(np.log(var)) + np.sum((z - mean) ** 2 / var))


def load_singer_attributes(eda_dir: str) -> dict:
    """Load singer -> list of attributes from singer_attributes.json. Returns {} if missing."""
    path = os.path.join(eda_dir, "singer_attributes.json")
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def attribute_confidence_from_probs(
    unique_artists: list, probs: np.ndarray, singer_attributes: dict
) -> list[tuple[str, float]]:
    """
    Given P(artist | song), compute confidence for each attribute as the sum of
    P(artist | song) over artists that have that attribute. Returns list of
    (attribute, confidence) sorted by confidence descending.
    """
    attr_to_prob = {}
    for idx, artist in enumerate(unique_artists):
        if artist not in singer_attributes:
            continue
        p = probs[idx]
        for attr in singer_attributes[artist]:
            attr_to_prob[attr] = attr_to_prob.get(attr, 0.0) + p
    return sorted(attr_to_prob.items(), key=lambda x: -x[1])


def load_artifacts_for_service(vae_dir: str, ckpt_path: str = None) -> dict:
    """
    Load VAE, scaler, config, fitted artist Gaussians, and singer attributes.
    Returns a dict suitable for predict_from_buffer() so the backend can load once and reuse.
    """
    model, scaler, config, latents, artists, device = load_vae_artifacts(vae_dir, ckpt_path=ckpt_path)
    unique_artists, means, vars_, counts = fit_artist_gaussians(latents, artists)
    n_artists = len(unique_artists)
    total_tracks = len(artists)
    priors = np.array([counts[j] / total_tracks for j in range(n_artists)])
    singer_attributes = load_singer_attributes(_EDA_DIR)
    return {
        "model": model,
        "scaler": scaler,
        "config": config,
        "unique_artists": unique_artists,
        "means": means,
        "vars_": vars_,
        "priors": priors,
        "device": device,
        "singer_attributes": singer_attributes,
    }


def predict_from_buffer(
    artifacts: dict,
    audio_bytes: bytes,
    temperature: float = 5.0,
) -> dict:
    """
    Run VAE artist/attribute prediction on raw audio bytes (e.g. from an upload).
    artifacts: from load_artifacts_for_service().
    Returns dict with artist_probs, top_artist, top_3_artists, attributes, top_3_attributes, n_chunks.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    model = artifacts["model"]
    scaler = artifacts["scaler"]
    config = artifacts["config"]
    unique_artists = artifacts["unique_artists"]
    means = artifacts["means"]
    vars_ = artifacts["vars_"]
    priors = artifacts["priors"]
    device = artifacts["device"]
    singer_attributes = artifacts["singer_attributes"]
    n_artists = len(unique_artists)
    sr = config["sr"]

    y = load_audio_from_bytes(audio_bytes, sr)
    chunk_list = list(
        get_feature_chunks_from_buffer(y, sr, config, min_one_chunk=False)
    )
    if not chunk_list:
        return {
            "error": "No valid feature chunks (audio too short or load failed).",
            "artist_probs": {},
            "top_artist": None,
            "top_3_artists": [],
            "attributes": [],
            "top_3_attributes": [],
            "n_chunks": 0,
        }

    chunks = np.stack(chunk_list, axis=0).astype(np.float32)
    chunks_scaled = scaler.transform(chunks).astype(np.float32)
    with torch.no_grad():
        x_t = torch.from_numpy(chunks_scaled).to(device)
        mu, _ = model.encode(x_t)
        chunk_latents = mu.cpu().numpy()
    z = np.mean(chunk_latents, axis=0)

    log_lik = np.array([log_gaussian_pdf(z, means[j], vars_[j]) for j in range(n_artists)])
    log_post = log_lik + np.log(priors + 1e-10)
    log_post -= np.max(log_post)
    log_post_scaled = log_post / temperature
    probs = np.exp(log_post_scaled)
    probs /= probs.sum()

    order = np.argsort(probs)[::-1]
    artist_probs = {unique_artists[j]: float(probs[j]) for j in range(n_artists)}
    top_artist = unique_artists[order[0]]
    top_3_artists = [unique_artists[order[i]] for i in range(min(3, n_artists))]

    attr_conf = attribute_confidence_from_probs(unique_artists, probs, singer_attributes)
    attributes = [{"tag": a, "confidence": c} for a, c in attr_conf]
    top_3_attributes = [{"tag": a, "confidence": c} for a, c in attr_conf[:3]]

    return {
        "artist_probs": artist_probs,
        "top_artist": top_artist,
        "top_3_artists": top_3_artists,
        "attributes": attributes,
        "top_3_attributes": top_3_attributes,
        "n_chunks": len(chunk_list),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Predict artist and singing-style attributes from an audio file using the trained VAE."
    )
    parser.add_argument("audio_path", help="Path to the audio file")
    parser.add_argument(
        "checkpoint",
        nargs="?",
        default=None,
        help="Optional path to checkpoint .ckpt (default: use last.ckpt or model.pt)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=5.0,
        metavar="T",
        help="Temperature for softmax over artist log-posteriors (default: 5). Higher = softer distribution.",
    )
    args = parser.parse_args()

    audio_path = os.path.abspath(args.audio_path)
    if not os.path.isfile(audio_path):
        print(f"File not found: {audio_path}")
        sys.exit(1)
    ckpt_path = os.path.abspath(args.checkpoint) if args.checkpoint else None
    temperature = args.temperature
    if temperature <= 0:
        print("Temperature must be positive.")
        sys.exit(1)

    vae_dir = get_vae_output_dir()
    last_ckpt = get_last_ckpt_path()
    model_pt = os.path.join(vae_dir, "model.pt")
    if not os.path.isdir(vae_dir):
        print(f"VAE output dir not found: {vae_dir}. Run train_vae_singer.py first.")
        sys.exit(1)
    has_ckpt = (ckpt_path and os.path.isfile(ckpt_path)) or os.path.isfile(last_ckpt) or os.path.isfile(model_pt)
    if not has_ckpt:
        print(f"No VAE weights. Run train_vae_singer.py first or pass a checkpoint path.")
        sys.exit(1)

    print("Loading VAE and artist Gaussians...")
    model, scaler, config, latents, artists, device = load_vae_artifacts(vae_dir, ckpt_path=ckpt_path)
    unique_artists, means, vars_, counts = fit_artist_gaussians(latents, artists)
    n_artists = len(unique_artists)
    total_tracks = len(artists)
    priors = np.array([counts[j] / total_tracks for j in range(n_artists)])

    # Feature extraction must match training (mel / mfcc / mel_mfcc from config)

    print(f"Processing: {audio_path}")
    chunk_list = list(
        get_feature_chunks_for_path(audio_path, config)
    )
    if not chunk_list:
        print("No valid feature chunks (audio too short or load failed).")
        sys.exit(1)

    chunks = np.stack(chunk_list, axis=0).astype(np.float32)
    chunks_scaled = scaler.transform(chunks).astype(np.float32)
    with torch.no_grad():
        x_t = torch.from_numpy(chunks_scaled).to(device)
        mu, _ = model.encode(x_t)
        chunk_latents = mu.cpu().numpy()
    z = np.mean(chunk_latents, axis=0)

    # P(artist | z) ∝ P(z | artist) * P(artist); then apply temperature
    log_lik = np.array([log_gaussian_pdf(z, means[j], vars_[j]) for j in range(n_artists)])
    log_post = log_lik + np.log(priors + 1e-10)
    log_post -= np.max(log_post)
    log_post_scaled = log_post / temperature
    probs = np.exp(log_post_scaled)
    probs /= probs.sum()

    print(f"\nUsed {len(chunk_list)} chunks (mean latent), temperature={temperature}.")
    print("\nP(artist | song) for each artist:")
    order = np.argsort(probs)[::-1]
    for idx in order:
        p = probs[idx]
        name = unique_artists[idx]
        bar = "█" * int(round(p * 40)) + "░" * (40 - int(round(p * 40)))
        print(f"  {name:25s}  {p:6.2%}  {bar}")
    print(f"\nMost likely: {unique_artists[order[0]]} ({probs[order[0]]:.2%})")

    # Attribute confidence from singer_attributes.json (given similar-artist distribution)
    singer_attributes = load_singer_attributes(_EDA_DIR)
    if singer_attributes:
        attr_conf = attribute_confidence_from_probs(unique_artists, probs, singer_attributes)
        if attr_conf:
            print("\n--- Singing style attributes (confidence from predicted artists) ---")
            for attr, conf in attr_conf:
                print(f"  {attr:20s}  {conf:6.2%}")
            top3 = attr_conf[:3]
            print("\nTop 3 singing style attributes:")
            for i, (attr, conf) in enumerate(top3, 1):
                print(f"  {i}. {attr}: {conf:.1f}%")


if __name__ == "__main__":
    main()

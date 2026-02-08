"""
Load the trained VAE outputs (saved latents and artist labels from train_vae_singer.py),
aggregate latent vectors by artist (mean), and compute a pairwise similarity matrix
between artists. Saves CSV and heatmap. Also computes R^2 between VAE similarity
and tag IOU (intersection over union of artist tags from singer_attributes.json).

Usage:
  python artist_similarity_vae.py
      Use pre-saved latents.npy and artists.npy from the last training run.
  python artist_similarity_vae.py <path_to_checkpoint.ckpt>
      Load VAE from checkpoint, re-encode all tracks, then compute similarity.
"""
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

_EDA_DIR = os.path.dirname(os.path.abspath(__file__))
if _EDA_DIR not in sys.path:
    sys.path.insert(0, _EDA_DIR)

from vae_singer_config import get_vae_output_dir
from singer_attributes_utils import resolve_artist_for_attributes


def cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity between rows. Zero vectors get similarity 0."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = vectors / norms
    return np.dot(unit, unit.T)


def load_singer_attributes(eda_dir: str) -> dict:
    """Load artist -> list of tags from singer_attributes.json. Returns {} if missing."""
    path = os.path.join(eda_dir, "singer_attributes.json")
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def tag_iou_matrix(unique_artists: list, singer_attributes: dict) -> np.ndarray:
    """Pairwise IOU (intersection over union) of tag sets. Artists missing from attributes get empty set.
    Uses resolve_artist_for_attributes so names from data (e.g. Olivia Rodrigo) match attribute keys (e.g. Olivia).
    """
    n = len(unique_artists)
    iou = np.zeros((n, n), dtype=np.float64)
    sets = [set(singer_attributes.get(resolve_artist_for_attributes(a), [])) for a in unique_artists]
    for i in range(n):
        for j in range(n):
            if i == j:
                iou[i, j] = 1.0
            else:
                a, b = sets[i], sets[j]
                union = len(a | b)
                iou[i, j] = len(a & b) / union if union else 0.0
    return iou


def r2_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Coefficient of determination R^2 between x and y (1 - SS_res/SS_tot)."""
    if len(x) != len(y) or len(x) < 2:
        return float("nan")
    y_mean = np.mean(y)
    ss_tot = np.sum((y - y_mean) ** 2)
    if ss_tot == 0:
        return float("nan")
    # Simple linear regression: y ~ x
    cov = np.cov(x, y)
    slope = cov[0, 1] / cov[0, 0] if cov[0, 0] != 0 else 0.0
    intercept = y_mean - slope * np.mean(x)
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    return 1.0 - (ss_res / ss_tot)


def _latents_from_checkpoint(ckpt_path: str, vae_dir: str, eda_dir: str):
    """Load VAE from checkpoint, re-encode all tracks, return (latents_per_track, artists)."""
    import torch
    from train_vae_singer import VAELightningModule, get_mel_chunk_data

    DATASET_CACHE_DIR = "vae_singer_dataset_cache"
    cache_dir = os.path.join(eda_dir, "output", DATASET_CACHE_DIR)

    with open(os.path.join(vae_dir, "config.pkl"), "rb") as f:
        config = pickle.load(f)
    with open(os.path.join(vae_dir, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    data = get_mel_chunk_data(eda_dir, cache_dir=cache_dir)
    if data is None:
        raise RuntimeError("No mel chunk data. Check data/ and singers.txt.")
    chunks = data["chunks"]
    track_indices = data["track_indices"]
    artists = data["artists"]
    n_tracks = data["n_tracks"]
    latent_dim = config["latent_dim"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lightning_model = VAELightningModule.load_from_checkpoint(
        ckpt_path, map_location=device, strict=False
    )
    model = lightning_model.vae.to(device)
    model.eval()

    all_scaled = scaler.transform(chunks).astype(np.float32)
    with torch.no_grad():
        x_t = torch.from_numpy(all_scaled).to(torch.float32).to(device)
        mu, _ = model.encode(x_t)
        all_latents = mu.cpu().numpy()

    latents_per_track = np.zeros((n_tracks, latent_dim), dtype=np.float64)
    counts = np.zeros(n_tracks, dtype=int)
    for i in range(len(track_indices)):
        t = track_indices[i]
        latents_per_track[t] += all_latents[i]
        counts[t] += 1
    for t in range(n_tracks):
        if counts[t] > 0:
            latents_per_track[t] /= counts[t]

    return latents_per_track, artists


def main():
    eda_dir = _EDA_DIR
    vae_dir = get_vae_output_dir()
    ckpt_path = sys.argv[1].strip() if len(sys.argv) > 1 else None

    if ckpt_path:
        if not os.path.isfile(ckpt_path):
            print(f"Checkpoint not found: {ckpt_path}")
            return
        print(f"Re-encoding tracks with checkpoint: {ckpt_path}")
        latents, artists = _latents_from_checkpoint(ckpt_path, vae_dir, eda_dir)
        artists = list(artists)
    else:
        latents_path = os.path.join(vae_dir, "latents.npy")
        artists_path = os.path.join(vae_dir, "artists.npy")
        if not os.path.isfile(latents_path) or not os.path.isfile(artists_path):
            print(f"VAE outputs not found in {vae_dir}. Run train_vae_singer.py first (or pass a checkpoint path).")
            return
        latents = np.load(latents_path, allow_pickle=True)
        artists = list(np.load(artists_path, allow_pickle=True))

    # Aggregate by artist: mean latent vector per artist
    unique_artists = sorted(set(artists))
    artist_to_idx = {a: i for i, a in enumerate(unique_artists)}
    n_artists = len(unique_artists)
    dim = latents.shape[1]
    artist_vectors = np.zeros((n_artists, dim), dtype=np.float64)
    counts = np.zeros(n_artists, dtype=int)
    for i, a in enumerate(artists):
        j = artist_to_idx[a]
        artist_vectors[j] += latents[i]
        counts[j] += 1
    for j in range(n_artists):
        if counts[j] > 0:
            artist_vectors[j] /= counts[j]

    sim = cosine_similarity_matrix(artist_vectors)
    np.clip(sim, -1.0, 1.0, out=sim)

    # Upper triangle (excluding diagonal) for pairwise stats
    triu_i, triu_j = np.triu_indices(n_artists, k=1)
    sim_flat = sim[triu_i, triu_j]

    # R^2 between VAE similarity and tag IOU
    singer_attributes = load_singer_attributes(eda_dir)
    if singer_attributes:
        iou = tag_iou_matrix(unique_artists, singer_attributes)
        iou_flat = iou[triu_i, triu_j]
        r2 = r2_correlation(sim_flat, iou_flat)
        pearson_r = np.corrcoef(sim_flat, iou_flat)[0, 1] if len(sim_flat) > 1 else float("nan")
        print(f"VAE similarity vs tag IOU: R^2 = {r2:.4f}, Pearson r = {pearson_r:.4f}")
    else:
        print("singer_attributes.json not found; skipping R^2 (VAE similarity vs tag IOU).")

    # Save matrix
    out_dir = os.path.dirname(get_vae_output_dir())  # eda/output
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(sim, index=unique_artists, columns=unique_artists)
    csv_path = os.path.join(out_dir, "artist_similarity_vae.csv")
    df.to_csv(csv_path)
    print(f"Saved similarity matrix to {csv_path}")

    # Warn if similarities are mostly near ±1 (latent collapse, often from tag-IOU auxiliary loss)
    outside_mid = np.sum(np.abs(sim_flat) > 0.9)
    if outside_mid > 0.7 * len(sim_flat):
        print(
            f"Note: {100 * outside_mid / len(sim_flat):.0f}% of pairs have |cos_sim| > 0.9 (latent collapse). "
            "Retrain with TAG_IOU_LOSS_WEIGHT=0 in train_vae_singer.py for more spread-out similarities."
        )

    # Heatmap: cosine in [-1, 1]; use diverging scale so 0 is neutral
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("matplotlib/seaborn not installed; skipping heatmap.")
        return

    fig, ax = plt.subplots(figsize=(max(8, n_artists * 0.5), max(6, n_artists * 0.4)))
    sns.heatmap(
        sim,
        xticklabels=unique_artists,
        yticklabels=unique_artists,
        ax=ax,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        center=0,
    )
    ax.set_title("Artist similarity (VAE latent space, cosine)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    heatmap_path = os.path.join(out_dir, "artist_similarity_vae_heatmap.png")
    plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved heatmap to {heatmap_path}")


if __name__ == "__main__":
    main()

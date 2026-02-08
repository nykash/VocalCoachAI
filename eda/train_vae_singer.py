"""
Train a VAE on mel spectrogram chunks from the same singer tracks as knn_singer.
Each track is split into fixed-length chunks; each chunk's log-mel is flattened
and used as the VAE input. Train/validation split is by track (stratified by artist).
Dataset (mel chunks) is cached to disk and reused when paths/mtimes/config match.
Saves model, scaler, config, and per-track mean latents for artist_similarity_vae
and predict_singer_vae.
"""
import json
import os
import pickle
import sys
from typing import Optional

import numpy as np

if "LOKY_MAX_CPU_COUNT" not in os.environ:
    os.environ["LOKY_MAX_CPU_COUNT"] = "8"

_EDA_DIR = os.path.dirname(os.path.abspath(__file__))
if _EDA_DIR not in sys.path:
    sys.path.insert(0, _EDA_DIR)

from vae_singer_config import VAE_OUTPUT_DIR, CHECKPOINT_DIR, get_vae_output_dir
from singer_attributes_utils import resolve_artist_for_attributes
from knn_singer import find_songs_by_singers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import Counter

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, *args, **kwargs):
        return it

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    print("PyTorch is required. Install with: pip install torch")
    sys.exit(1)

try:
    import lightning.pytorch as pl
    from lightning.pytorch.loggers import TensorBoardLogger
    from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
except ImportError:
    try:
        import pytorch_lightning as pl
        from pytorch_lightning.loggers import TensorBoardLogger
        from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
    except ImportError:
        print("PyTorch Lightning is required. Install with: pip install lightning")
        sys.exit(1)

try:
    import librosa
except ImportError:
    print("librosa is required. Install with: pip install librosa")
    sys.exit(1)

# Audio / mel config (match analyze.py for consistency)
SR = 22050
HOP_LENGTH = 512
N_FFT = 2048
N_MELS = 128
N_MFCC = 20  # used when FEATURE_TYPE is "mfcc" or "mel_mfcc"
CHUNK_SEC = 6
# Frames per chunk
N_FRAMES_PER_CHUNK = int(round(CHUNK_SEC * SR / HOP_LENGTH))

# Feature type: "mel" (log-mel only), "mfcc", or "mel_mfcc" (concatenated).
# MFCCs are compact and often good for voice/speaker; mel_mfcc combines spectral shape + decorrelated coeffs.
FEATURE_TYPE = "mel"

# VAE config
LATENT_DIM = 64
# Deeper bottleneck (128) gives gentler compression and often better latent structure
HIDDEN_DIMS = [1024, 512, 256, 128]
EPOCHS = 25
BATCH_SIZE = 256
LR = 1e-4
VAL_FRAC = 0.2
VAE_KL_BETA = 1.0
LOG_SUBDIR = "logs"
VAL_LOSS_CHECKPOINT = "best_val_loss.ckpt"
DATASET_CACHE_DIR = "vae_singer_dataset_cache"
MANIFEST_FNAME = "manifest.json"
CHUNKS_FNAME = "chunks.npy"
TRACK_IDX_FNAME = "track_indices.npy"

# Optional: maximize agreement between latent similarity and tag similarity
SINGER_ATTRIBUTES_FNAME = "singer_attributes.json"
# Tag-IoU auxiliary loss: set to 0 to avoid latent collapse (similarities clustering near ±1).
# Non-zero (e.g. 0.05–0.1) pulls tag-similar artists together but often collapses space to two poles.
TAG_IOU_LOSS_WEIGHT = 0.0  # lambda_aux: weight of tag-IoU auxiliary loss (0 = disabled)


def _orthogonal_init(module: nn.Module) -> None:
    """Orthogonal init for linear layers; helps VAE training stability."""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=1.0)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class VAE(nn.Module):
    """
    MLP VAE with LayerNorm + GELU for more stable training and better latent structure.
    Encoder: Linear -> LayerNorm -> GELU per layer; then fc_mu / fc_logvar from last hidden.
    Decoder: mirror with Linear -> LayerNorm -> GELU; final layer linear only.
    """

    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: list):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder
        enc_layers = []
        d = input_dim
        for h in hidden_dims:
            enc_layers.append(nn.Linear(d, h))
            enc_layers.append(nn.LayerNorm(h))
            enc_layers.append(nn.GELU())
            d = h
        self.encoder = nn.Sequential(*enc_layers)
        self.fc_mu = nn.Linear(d, latent_dim)
        self.fc_logvar = nn.Linear(d, latent_dim)

        # Decoder
        dec_layers = []
        d = latent_dim
        for h in reversed(hidden_dims):
            dec_layers.append(nn.Linear(d, h))
            dec_layers.append(nn.LayerNorm(h))
            dec_layers.append(nn.GELU())
            d = h
        dec_layers.append(nn.Linear(d, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

        self.apply(_orthogonal_init)

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        logvar = torch.clamp(logvar, -20.0, 2.0)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss(x, x_recon, mu, logvar, beta=1.0):
    mse = nn.functional.mse_loss(x_recon, x, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return mse + beta * kl


def _tag_iou(set_a: set, set_b: set) -> float:
    """IoU between two tag sets; 0 if both empty."""
    if not set_a and not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union else 0.0


def load_tag_iou_for_artists(artists: list, attributes_path: str):
    """
    Load singer_attributes.json and build pairwise tag IoU matrix for the given
    artist list. artists[i] is the artist name for track i (so duplicates OK).
    Returns (unique_artists, artist_to_idx, tag_iou_matrix) or (None, None, None) if file missing.
    - unique_artists: list of unique names in order of first appearance.
    - artist_to_idx: name -> index in unique_artists.
    - tag_iou_matrix: (n_artists, n_artists) float array, IoU in [0,1]. Missing names get empty tags.
    """
    if not os.path.isfile(attributes_path):
        return None, None, None
    try:
        with open(attributes_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None, None, None
    # Resolve data artist names to attribute keys (aliases fix mismatches e.g. Olivia Rodrigo -> Olivia)
    attributes = {k.strip(): (set(v) if isinstance(v, (list, tuple)) else set()) for k, v in raw.items()}
    unique_artists = list(dict.fromkeys(artists))
    n = len(unique_artists)
    tag_sets = [attributes.get(resolve_artist_for_attributes(a), set()) for a in unique_artists]
    tag_iou_matrix = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            tag_iou_matrix[i, j] = _tag_iou(tag_sets[i], tag_sets[j])
    artist_to_idx = {a: i for i, a in enumerate(unique_artists)}
    return unique_artists, artist_to_idx, tag_iou_matrix


class VAELightningModule(pl.LightningModule):
    """Lightning module for VAE training with logging and checkpointing.
    Optional tag-IoU auxiliary loss: pulls latents of artists with similar tags closer.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: list,
        lr: float = 1e-3,
        beta: float = 1.0,
        tag_iou_matrix: Optional[np.ndarray] = None,
        artist_indices: Optional[np.ndarray] = None,
        tag_iou_weight: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["tag_iou_matrix", "artist_indices"])
        self.lr = lr
        self.beta = beta
        self.tag_iou_weight = tag_iou_weight
        self.vae = VAE(input_dim, latent_dim, hidden_dims)
        if tag_iou_matrix is not None and artist_indices is not None and tag_iou_weight > 0:
            self.register_buffer("_tag_iou_tensor", torch.from_numpy(tag_iou_matrix).float())
            self.register_buffer("_artist_indices_tensor", torch.from_numpy(artist_indices).long())
        else:
            self._tag_iou_tensor = None
            self._artist_indices_tensor = None

    def forward(self, x):
        return self.vae(x)

    def _tag_aux_loss(self, mu: torch.Tensor, track_idx: torch.Tensor) -> torch.Tensor:
        """Auxiliary loss: sum over pairs (i,j) of tag_iou(artist_i, artist_j) * ||mu_i - mu_j||^2.
        Minimizing this pulls latents of tag-similar artists together.
        """
        B = mu.shape[0]
        if B < 2:
            return mu.new_zeros(())
        device = mu.device
        ai = self._artist_indices_tensor[track_idx]  # (B,)
        iou = self._tag_iou_tensor[ai]  # (B, n_artists) -> we need (B, B) iou[i,j]
        iou_ij = iou[:, ai]  # (B, B): iou_ij[i,j] = tag_iou(artist_i, artist_j)
        # Pairwise squared distances: (B,B), dist[i,j] = ||mu_i - mu_j||^2
        mu_sq = (mu * mu).sum(dim=1)
        dist_sq = mu_sq.unsqueeze(1) + mu_sq.unsqueeze(0) - 2 * (mu @ mu.T).clamp(min=0)
        dist_sq = dist_sq.clamp(min=0)
        # Mask out diagonal and take upper triangle to avoid double count
        mask = torch.triu(torch.ones(B, B, device=device, dtype=torch.bool), diagonal=1)
        weights = iou_ij[mask]
        d = dist_sq[mask]
        if weights.numel() == 0:
            return mu.new_zeros(())
        return (weights * d).mean()

    def _step(self, batch, prefix: str):
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            x, track_idx = batch[0], batch[1]
        else:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            track_idx = None
        recon, mu, logvar = self.vae(x)
        loss = vae_loss(x, recon, mu, logvar, beta=self.beta)
        if (
            self.tag_iou_weight > 0
            and track_idx is not None
            and self._tag_iou_tensor is not None
            and self._artist_indices_tensor is not None
        ):
            aux = self._tag_aux_loss(mu, track_idx)
            loss = loss + self.tag_iou_weight * aux
            self.log(f"{prefix}_tag_aux", aux, on_epoch=True)
        mse = nn.functional.mse_loss(recon, x, reduction="mean")
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        self.log(f"{prefix}_loss", loss, on_step=(prefix == "train"), on_epoch=True, prog_bar=True)
        self.log(f"{prefix}_mse", mse, on_epoch=True)
        self.log(f"{prefix}_kl", kl, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }


def load_audio(path: str):
    y, sr = librosa.load(path, sr=SR, mono=True)
    return y, sr


def _dataset_config_hash():
    """Hash of mel/chunk config so cache invalidates when params change."""
    base = f"n_mels_{N_MELS}_nframes_{N_FRAMES_PER_CHUNK}_sr_{SR}_hop_{HOP_LENGTH}_nfft_{N_FFT}_feature_{FEATURE_TYPE}"
    if FEATURE_TYPE in ("mfcc", "mel_mfcc"):
        base += f"_n_mfcc_{N_MFCC}"
    return base


def _dataset_cache_valid(cache_dir: str, audio_paths: list) -> bool:
    """True if cache exists and matches current paths, mtimes, and config."""
    manifest_path = os.path.join(cache_dir, MANIFEST_FNAME)
    if not os.path.isfile(manifest_path):
        return False
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    if manifest.get("config_hash") != _dataset_config_hash():
        return False
    cached_paths = manifest.get("paths", [])
    cached_mtimes = manifest.get("mtimes", [])
    if len(cached_paths) != len(audio_paths):
        return False
    current = [(os.path.abspath(p), os.path.getmtime(p)) for p in audio_paths]
    if cached_paths != [p for p, _ in current] or cached_mtimes != [t for _, t in current]:
        return False
    return True


def _load_cached_dataset(cache_dir: str) -> Optional[dict]:
    """Load chunks, track_indices, artists, track_names, n_tracks from cache."""
    manifest_path = os.path.join(cache_dir, MANIFEST_FNAME)
    chunks_path = os.path.join(cache_dir, CHUNKS_FNAME)
    track_idx_path = os.path.join(cache_dir, TRACK_IDX_FNAME)
    if not all(os.path.isfile(p) for p in (manifest_path, chunks_path, track_idx_path)):
        return None
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        chunks = np.load(chunks_path)
        track_indices = np.load(track_idx_path)
        return {
            "chunks": chunks,
            "track_indices": track_indices,
            "artists": manifest["artists"],
            "track_names": manifest["track_names"],
            "path_artist": list(zip(manifest["paths"], manifest["artists"])),
            "n_tracks": manifest["n_tracks"],
        }
    except (KeyError, OSError):
        return None


def _save_cached_dataset(cache_dir: str, data: dict) -> None:
    """Save manifest and array files for dataset cache."""
    os.makedirs(cache_dir, exist_ok=True)
    paths = [os.path.abspath(p) for p, _ in data["path_artist"]]
    mtimes = [os.path.getmtime(p) for p in paths]
    manifest = {
        "config_hash": _dataset_config_hash(),
        "paths": paths,
        "mtimes": mtimes,
        "artists": data["artists"],
        "track_names": data["track_names"],
        "n_tracks": data["n_tracks"],
    }
    with open(os.path.join(cache_dir, MANIFEST_FNAME), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=0)
    np.save(os.path.join(cache_dir, CHUNKS_FNAME), data["chunks"])
    np.save(os.path.join(cache_dir, TRACK_IDX_FNAME), data["track_indices"])


def _features_from_audio(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Compute feature matrix (D, T) for the chosen FEATURE_TYPE.
    - mel: log-mel (N_MELS, T)
    - mfcc: MFCCs (N_MFCC, T)
    - mel_mfcc: vertical stack [log-mel; mfcc], shape (N_MELS + N_MFCC, T)
    """
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS, fmin=20, fmax=sr // 2
    )
    log_mel = np.log(mel + 1e-6).astype(np.float32)
    if FEATURE_TYPE == "mel":
        return log_mel
    mfcc = librosa.feature.mfcc(
        S=mel, n_mfcc=N_MFCC, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, fmin=20, fmax=sr // 2
    ).astype(np.float32)
    if FEATURE_TYPE == "mfcc":
        return mfcc
    if FEATURE_TYPE == "mel_mfcc":
        return np.vstack([log_mel, mfcc])
    return log_mel


def feature_chunks_for_path(path: str, track_idx: int):
    """
    Load audio, compute features (mel / mfcc / mel_mfcc), slice into non-overlapping chunks.
    Yields (chunk_flat, track_idx) for each chunk. chunk_flat shape (D * n_frames,).
    Uses module constants (FEATURE_TYPE, N_MELS, etc.).
    """
    try:
        y, sr = load_audio(path)
    except Exception:
        return
    feat = _features_from_audio(y, sr)
    D, T = feat.shape
    for start in range(0, T - N_FRAMES_PER_CHUNK + 1, N_FRAMES_PER_CHUNK):
        chunk = feat[:, start : start + N_FRAMES_PER_CHUNK]
        if chunk.shape[1] == N_FRAMES_PER_CHUNK:
            yield chunk.ravel(), track_idx


def _feature_matrix_from_audio(y: np.ndarray, sr: int, config: dict) -> np.ndarray:
    """Compute (D, T) feature matrix from audio using config (feature_type, n_mels, n_mfcc, etc.)."""
    hop_length = config["hop_length"]
    n_fft = config["n_fft"]
    n_mels = config["n_mels"]
    feature_type = config.get("feature_type", "mel")
    n_mfcc = config.get("n_mfcc", 20)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmin=20, fmax=sr // 2
    )
    log_mel = np.log(mel + 1e-6).astype(np.float32)
    if feature_type == "mel":
        return log_mel
    mfcc = librosa.feature.mfcc(
        S=mel, n_mfcc=n_mfcc, sr=sr, n_fft=n_fft, hop_length=hop_length, fmin=20, fmax=sr // 2
    ).astype(np.float32)
    if feature_type == "mfcc":
        return mfcc
    return np.vstack([log_mel, mfcc])


def get_feature_chunks_from_buffer(
    y: np.ndarray, sr: int, config: dict, min_one_chunk: bool = False
) -> list:
    """
    Compute feature chunks from in-memory audio (y, sr) using config. Matches get_feature_chunks_for_path.
    If min_one_chunk=True and audio is shorter than one chunk, pad with last frame so one chunk is returned.
    Returns list of 1D arrays.
    """
    n_frames_per_chunk = config["n_frames_per_chunk"]
    feat = _feature_matrix_from_audio(y, sr, config)
    D, T = feat.shape
    if T == 0:
        return []
    if min_one_chunk and T < n_frames_per_chunk:
        pad_width = n_frames_per_chunk - T
        last_frame = feat[:, -1:]
        feat = np.concatenate(
            [feat, np.broadcast_to(last_frame, (D, pad_width))], axis=1
        )
        T = feat.shape[1]
    out = []
    for start in range(0, T - n_frames_per_chunk + 1, n_frames_per_chunk):
        chunk = feat[:, start : start + n_frames_per_chunk]
        if chunk.shape[1] == n_frames_per_chunk:
            out.append(chunk.ravel())
    return out


def get_feature_chunks_for_path(path: str, config: dict):
    """
    Produce flattened feature chunks for a path using a saved config (from config.pkl).
    Use this in predict_singer_vae so extraction matches the trained model.
    Yields 1D numpy arrays; feature type is read from config (default "mel" if missing).
    """
    sr = config["sr"]
    try:
        y, _ = load_audio(path)
    except Exception:
        return
    for chunk in get_feature_chunks_from_buffer(y, sr, config, min_one_chunk=False):
        yield chunk


def get_mel_chunk_data(eda_dir: str, cache_dir: Optional[str] = None):
    """
    Get (path, artist) for each track, then build chunk array and track indices.
    If cache_dir is set and cache is valid (paths/mtimes/config match), load from cache.
    Otherwise compute and, if cache_dir is set, save to cache.
    Returns chunks (N, D), track_indices (N,), artists (n_tracks,), track_names (n_tracks,),
    and path_artist list for mapping track_idx -> (path, artist).
    """
    data_dir = os.path.join(eda_dir, "data")
    singers_path = os.path.join(eda_dir, "singers.txt")
    if not os.path.isdir(data_dir) or not os.path.isfile(singers_path):
        return None

    path_artist, _ = find_songs_by_singers(data_dir, singers_path)
    if not path_artist:
        return None
    counts = Counter(a for _, a in path_artist)
    path_artist = [(p, a) for p, a in path_artist if counts[a] >= 2]
    if not path_artist:
        return None

    audio_paths = [p for p, _ in path_artist]
    artists = [a for _, a in path_artist]
    track_names = [os.path.splitext(os.path.basename(p))[0] for p in audio_paths]
    n_tracks = len(audio_paths)

    if cache_dir and _dataset_cache_valid(cache_dir, audio_paths):
        data = _load_cached_dataset(cache_dir)
        if data is not None:
            print("Loading dataset from cache...")
            return data

    chunk_list = []
    track_indices = []
    for track_idx, path in enumerate(tqdm(audio_paths, desc="Feature chunks", unit="track")):
        for chunk_flat, ti in feature_chunks_for_path(path, track_idx):
            chunk_list.append(chunk_flat)
            track_indices.append(ti)

    if not chunk_list:
        return None
    chunks = np.stack(chunk_list, axis=0).astype(np.float32)
    track_indices_arr = np.array(track_indices, dtype=np.int64)
    data = {
        "chunks": chunks,
        "track_indices": track_indices_arr,
        "artists": artists,
        "track_names": track_names,
        "path_artist": path_artist,
        "n_tracks": n_tracks,
    }
    if cache_dir:
        _save_cached_dataset(cache_dir, data)
        print(f"Dataset cached to {cache_dir}")
    return data


def main():
    eda_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = get_vae_output_dir()
    cache_dir = os.path.join(eda_dir, "output", DATASET_CACHE_DIR)
    os.makedirs(out_dir, exist_ok=True)

    data = get_mel_chunk_data(eda_dir, cache_dir=cache_dir)
    if data is None:
        print("No data. Check data/ and singers.txt (need >= 2 songs per artist).")
        return

    chunks = data["chunks"]
    track_indices = data["track_indices"]
    artists = data["artists"]
    track_names = data["track_names"]
    n_tracks = data["n_tracks"]

    input_dim = chunks.shape[1]
    print(f"Total chunks: {chunks.shape[0]}, input_dim: {input_dim} (feature_type={FEATURE_TYPE}, n_frames={N_FRAMES_PER_CHUNK})")

    # Train/val split by track (stratified by artist)
    track_ids = np.arange(n_tracks)
    artist_per_track = np.array(artists)
    train_track_ids, val_track_ids = train_test_split(
        track_ids, test_size=VAL_FRAC, random_state=42, stratify=artist_per_track
    )
    train_track_set = set(train_track_ids)
    val_track_set = set(val_track_ids)
    train_mask = np.array([track_indices[i] in train_track_set for i in range(len(track_indices))])
    val_mask = np.array([track_indices[i] in val_track_set for i in range(len(track_indices))])

    X_train = chunks[train_mask]
    X_val = chunks[val_mask]
    train_track_idx = track_indices[train_mask]
    val_track_idx = track_indices[val_mask]
    print(f"Train chunks: {X_train.shape[0]}, Val chunks: {X_val.shape[0]}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_val_scaled = scaler.transform(X_val).astype(np.float32)

    # Optional tag-IoU auxiliary loss: need track_idx per sample
    attributes_path = os.path.join(eda_dir, SINGER_ATTRIBUTES_FNAME)
    unique_artists, artist_to_idx, tag_iou_matrix = load_tag_iou_for_artists(artists, attributes_path)
    artist_indices = None
    if tag_iou_matrix is not None:
        artist_indices = np.array([artist_to_idx[a] for a in artists], dtype=np.int64)
        print(f"Tag-IoU auxiliary loss enabled (weight={TAG_IOU_LOSS_WEIGHT}): {tag_iou_matrix.shape[0]} artists from {SINGER_ATTRIBUTES_FNAME}")

    use_tag_loss = TAG_IOU_LOSS_WEIGHT > 0 and tag_iou_matrix is not None and artist_indices is not None
    if use_tag_loss:
        train_dataset = TensorDataset(
            torch.from_numpy(X_train_scaled),
            torch.from_numpy(train_track_idx),
        )
        val_dataset = TensorDataset(
            torch.from_numpy(X_val_scaled),
            torch.from_numpy(val_track_idx),
        )
    else:
        train_dataset = TensorDataset(torch.from_numpy(X_train_scaled))
        val_dataset = TensorDataset(torch.from_numpy(X_val_scaled))

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    log_dir = os.path.join(out_dir, LOG_SUBDIR)
    ckpt_dir = os.path.join(out_dir, CHECKPOINT_DIR)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    lightning_model = VAELightningModule(
        input_dim=input_dim,
        latent_dim=LATENT_DIM,
        hidden_dims=HIDDEN_DIMS,
        lr=LR,
        beta=VAE_KL_BETA,
        tag_iou_matrix=tag_iou_matrix if use_tag_loss else None,
        artist_indices=artist_indices if use_tag_loss else None,
        tag_iou_weight=TAG_IOU_LOSS_WEIGHT if use_tag_loss else 0.0,
    )

    logger = TensorBoardLogger(save_dir=log_dir, name="vae_singer", version=None)
    model_ckpt = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="vae-{epoch:03d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        every_n_epochs=1,
    )
    callbacks = [
        model_ckpt,
        EarlyStopping(
            monitor="val_loss",
            patience=4,
            mode="min",
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=min(50, max(1, len(train_loader) // 2)),
        enable_progress_bar=True,
        gradient_clip_val=1.0,
        enable_model_summary=True,
    )
    print(f"Training VAE with Lightning: latent_dim={LATENT_DIM}, epochs={EPOCHS}")
    print(f"TensorBoard logs: {logger.log_dir}")
    trainer.fit(lightning_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Load best checkpoint for encoding and saving (else use final model)
    best_ckpt = getattr(model_ckpt, "best_model_path", None) if model_ckpt else None
    if best_ckpt and os.path.isfile(best_ckpt):
        lightning_model = VAELightningModule.load_from_checkpoint(
            best_ckpt, strict=False
        )  # strict=False: checkpoint may contain tag-IoU buffers we don't need for encoding
        print(f"Loaded best checkpoint: {best_ckpt}")
    else:
        lightning_model = trainer.model
    model = lightning_model.vae
    device = next(model.parameters()).device
    model.eval()

    # Encode all chunks, then mean latent per track
    all_scaled = scaler.transform(chunks).astype(np.float32)
    with torch.no_grad():
        x_t = torch.from_numpy(all_scaled).to(torch.float32).to(device)
        mu, _ = model.encode(x_t)
        all_latents = mu.cpu().numpy()

    latents_per_track = np.zeros((n_tracks, LATENT_DIM), dtype=np.float64)
    counts = np.zeros(n_tracks, dtype=int)
    for i in range(len(track_indices)):
        t = track_indices[i]
        latents_per_track[t] += all_latents[i]
        counts[t] += 1
    for t in range(n_tracks):
        if counts[t] > 0:
            latents_per_track[t] /= counts[t]

    # Save
    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
    config = {
        "input_dim": input_dim,
        "latent_dim": LATENT_DIM,
        "hidden_dims": HIDDEN_DIMS,
        "sr": SR,
        "hop_length": HOP_LENGTH,
        "n_fft": N_FFT,
        "n_mels": N_MELS,
        "n_mfcc": N_MFCC,
        "feature_type": FEATURE_TYPE,
        "chunk_sec": CHUNK_SEC,
        "n_frames_per_chunk": N_FRAMES_PER_CHUNK,
    }
    with open(os.path.join(out_dir, "config.pkl"), "wb") as f:
        pickle.dump(config, f)
    with open(os.path.join(out_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    np.save(os.path.join(out_dir, "latents.npy"), latents_per_track)
    np.save(os.path.join(out_dir, "track_names.npy"), np.array(track_names, dtype=object))
    np.save(os.path.join(out_dir, "artists.npy"), np.array(artists, dtype=object))

    print(f"Saved VAE and artifacts to {out_dir}")


if __name__ == "__main__":
    main()

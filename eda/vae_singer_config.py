"""
Shared config for VAE singer pipeline. Used by train_vae_singer.py,
artist_similarity_vae.py, and predict_singer_vae.py so they all use the same output paths.
"""
import os
import re

_EDA_DIR = os.path.dirname(os.path.abspath(__file__))
VAE_OUTPUT_DIR = "vae_singer"
CHECKPOINT_DIR = "checkpoints"
LAST_CKPT_FNAME = "last.ckpt"

# Preferred checkpoint to use for inference (backend / predict). Set env VAE_CHECKPOINT_PATH to override.
PREFERRED_CKPT_BASENAME = "vae-epoch=024-val_loss=0.4913.ckpt"


def get_vae_output_dir():
    """Absolute path to the VAE output directory (e.g. eda/output/vae_singer)."""
    return os.path.join(_EDA_DIR, "output", VAE_OUTPUT_DIR)


def get_checkpoint_dir():
    """Absolute path to the checkpoints subdir (e.g. eda/output/vae_singer/checkpoints)."""
    return os.path.join(get_vae_output_dir(), CHECKPOINT_DIR)


def get_last_ckpt_path():
    """Absolute path to last.ckpt."""
    return os.path.join(get_checkpoint_dir(), LAST_CKPT_FNAME)


def get_preferred_checkpoint_path():
    """
    Path to the checkpoint to use for inference (backend / predict).
    Order: 1) VAE_CHECKPOINT_PATH env, 2) PREFERRED_CKPT_BASENAME if present,
    3) best val_loss among vae-epoch=*-val_loss=*.ckpt, 4) last.ckpt.
    """
    ckpt_dir = get_checkpoint_dir()
    env_path = os.environ.get("VAE_CHECKPOINT_PATH", "").strip()
    if env_path and os.path.isfile(env_path):
        return env_path
    preferred = os.path.join(ckpt_dir, PREFERRED_CKPT_BASENAME)
    if os.path.isfile(preferred):
        return preferred
    # Best val_loss among vae-epoch=*-val_loss=*.ckpt
    pattern = re.compile(r"vae-epoch=\d+-val_loss=([\d.]+)\.ckpt$")
    best_path = None
    best_loss = float("inf")
    if os.path.isdir(ckpt_dir):
        for name in os.listdir(ckpt_dir):
            m = pattern.match(name)
            if m:
                try:
                    loss = float(m.group(1))
                    if not (loss != loss) and loss < best_loss:  # skip nan
                        best_loss = loss
                        best_path = os.path.join(ckpt_dir, name)
                except ValueError:
                    pass
    if best_path:
        return best_path
    return get_last_ckpt_path()

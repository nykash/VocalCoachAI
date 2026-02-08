"""
FastAPI REST service for audio analysis.
Exposes VAE singer/attribute prediction (from eda/predict_singer_vae) as a REST endpoint.
Serves song list and files from eda/data for karaoke.
Run from project root: uvicorn backend.main:app --reload
Or: python -m uvicorn backend.main:app --reload
"""
import os
import sys

# Ensure project root and eda are importable
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from contextlib import asynccontextmanager
from typing import Any, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Folder of songs for karaoke (eda/data)
_DATA_DIR = os.path.abspath(os.path.join(_ROOT, "eda", "data"))
_AUDIO_EXTENSIONS = {".mp3", ".m4a", ".wav", ".flac", ".ogg"}

# Lazy import after path is set so eda can be found
def _load_artifacts() -> Any:
    from eda.vae_singer_config import get_vae_output_dir, get_preferred_checkpoint_path
    from eda.predict_singer_vae import load_artifacts_for_service

    vae_dir = get_vae_output_dir()
    if not os.path.isdir(vae_dir):
        raise FileNotFoundError(
            f"VAE output dir not found: {vae_dir}. Run train_vae_singer.py in eda first."
        )
    ckpt_path = get_preferred_checkpoint_path()
    model_pt = os.path.join(vae_dir, "model.pt")
    if not os.path.isfile(ckpt_path) and not os.path.isfile(model_pt):
        raise FileNotFoundError(
            f"No VAE weights at {ckpt_path} or {model_pt}. Train the VAE first."
        )
    return load_artifacts_for_service(vae_dir, ckpt_path=None)


# Global artifacts (loaded at startup, None if load fails)
_artifacts: Any = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _artifacts
    try:
        _artifacts = _load_artifacts()
    except Exception as e:
        _artifacts = None
        print(f"VAE artifacts not loaded (service will return 503 for VAE analysis): {e}")
    yield
    _artifacts = None


app = FastAPI(
    title="VocalCoachAI Audio Analysis API",
    description="REST service for audio analysis (VAE singer/style tags).",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class VocalRegisterRealtimeRequest(BaseModel):
    """Base64-encoded float32 audio buffer from the client (e.g. Web Audio API)."""
    audio_base64: str
    sample_rate: int = 22050


@app.get("/health")
async def health():
    """Health check; reports whether VAE and vocal register models are loaded."""
    try:
        from backend.vocal_register_service import get_predictor
        vocal_loaded = get_predictor() is not None
    except Exception:
        vocal_loaded = False
    return {
        "status": "ok",
        "vae_loaded": _artifacts is not None,
        "vocal_register_loaded": vocal_loaded,
    }


def _safe_song_path(filename: str) -> Optional[str]:
    """Resolve filename to a path under _DATA_DIR; return None if invalid (no subdirs)."""
    if not filename or ".." in filename or os.path.isabs(filename) or os.sep in filename:
        return None
    path = os.path.normpath(os.path.join(_DATA_DIR, filename))
    if not path.startswith(_DATA_DIR):
        return None
    return path if os.path.isfile(path) else None


@app.get("/songs", response_model=List[str])
async def list_songs():
    """List audio filenames in eda/data for karaoke picker."""
    if not os.path.isdir(_DATA_DIR):
        return []
    names: List[str] = []
    for name in os.listdir(_DATA_DIR):
        if os.path.splitext(name)[1].lower() in _AUDIO_EXTENSIONS:
            names.append(name)
    return sorted(names)


@app.get("/songs/{filename:path}")
async def get_song(filename: str):
    """Stream a song file from eda/data by filename. CORS headers allow use in Web Audio API."""
    path = _safe_song_path(filename)
    if path is None:
        raise HTTPException(status_code=404, detail="Song not found")
    return FileResponse(
        path,
        media_type="audio/mpeg",
        filename=os.path.basename(path),
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Expose-Headers": "Content-Length, Content-Range",
        },
    )


@app.post("/analyze/vae-tags")
async def analyze_vae_tags(
    audio: UploadFile = File(..., description="Raw audio clip (e.g. WAV, MP3, FLAC)"),
    temperature: float = 10.0,
):
    """
    Analyze an audio clip with the trained VAE and return artist probabilities
    and singing-style attribute (tag) confidences.
    - **audio**: Raw audio file (any format supported by librosa).
    - **temperature**: Softmax temperature over artist posteriors (default 5.0).
    """
    if _artifacts is None:
        raise HTTPException(
            status_code=503,
            detail="VAE model not loaded. Train the VAE in eda (train_vae_singer.py) and ensure output exists.",
        )
    if temperature <= 0:
        raise HTTPException(status_code=400, detail="temperature must be positive")

    try:
        audio_bytes = await audio.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded file: {e}")

    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file")

    from eda.predict_singer_vae import predict_from_buffer

    try:
        result = predict_from_buffer(_artifacts, audio_bytes, temperature=temperature)
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"Analysis failed (e.g. unsupported format or corrupt audio): {e}",
        )

    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])

    return result


@app.post("/vocal-register/realtime")
async def vocal_register_realtime(body: VocalRegisterRealtimeRequest):
    """
    Real-time vocal register: Chest vs Head vs Falsetto.

    Send base64-encoded Float32Array (mono) and sample rate. Returns
    label (Chest Voice | Head Voice | Falsetto), confidence, and
    probabilities for all three classes.
    """
    from backend.vocal_register_service import predict_realtime_from_base64

    result = predict_realtime_from_base64(
        audio_base64=body.audio_base64,
        sample_rate=body.sample_rate,
    )
    if not result.get("success"):
        err = result.get("error", "Vocal register analysis failed")
        code = 400 if "short" in err.lower() or "no audio" in err.lower() else 503
        raise HTTPException(status_code=code, detail=err)
    return result

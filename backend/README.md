# Backend – Audio Analysis API

FastAPI REST service that accepts raw audio clips and returns VAE-based analysis (artist probabilities and singing-style tags).

## Prerequisites

1. **Train the VAE** (once) from the project root:
   ```bash
   cd eda && python train_vae_singer.py
   ```
   This produces `eda/output/vae_singer/` (and optionally `checkpoints/last.ckpt` or `model.pt`).

2. **Install dependencies** (from project root):
   ```bash
   pip install -r backend/requirements.txt
   ```
   The `eda` package is resolved from the project root when you run the server.

## Run the service

**Option A – from project root** (VocalCoachAI):

```bash
cd /path/to/VocalCoachAI
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

**Option B – from the backend directory** (use `main:app`, not `backend.main:app`):

```bash
cd /path/to/VocalCoachAI/backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
Make sure the project root is the parent of `backend` so the `eda` package is found.

**Option C – run script** (works from anywhere):

```bash
./backend/run.sh
```

- API docs: http://localhost:8000/docs  
- Health: http://localhost:8000/health  

## Endpoints

### `POST /analyze/vae-tags`

Upload a raw audio clip and get VAE analysis.

- **Body**: `multipart/form-data` with field `audio` (file). Supports formats supported by librosa (e.g. WAV, MP3, FLAC).
- **Query**: `temperature` (optional, default `5.0`) – softmax temperature over artist posteriors.

**Response** (JSON):

- `artist_probs`: `{ "Artist Name": 0.12, ... }`
- `top_artist`: most likely artist
- `top_3_artists`: list of top 3 artist names
- `attributes`: list of `{ "tag": "belt", "confidence": 0.45 }` (all tags with confidence)
- `top_3_attributes`: top 3 tags
- `n_chunks`: number of feature chunks used

**Example (curl):**

```bash
curl -X POST "http://localhost:8000/analyze/vae-tags?temperature=5" \
  -F "audio=@/path/to/your/clip.mp3"
```

If the VAE model is not loaded (e.g. not trained yet), the endpoint returns **503** and `/health` reports `vae_loaded: false`.

**Which checkpoint is used?** The service uses a preferred checkpoint in this order: 1) env `VAE_CHECKPOINT_PATH` if set and the file exists, 2) `vae-epoch=024-val_loss=0.4913.ckpt` in the checkpoints dir if present, 3) the checkpoint with the lowest `val_loss` among `vae-epoch=*-val_loss=*.ckpt`, 4) `last.ckpt`. So the “up to date” best val_loss checkpoint is used by default.

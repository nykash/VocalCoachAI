#!/usr/bin/env bash
# Run from repo root or from eda: use eda venv and start the live singer demo server.
# Usage: ./run.sh [path/to/checkpoint.ckpt]
#   If passed, checkpoint is used via VAE_CKPT; otherwise server uses its default.
cd "$(dirname "$0")"
ROOT="$(cd .. && pwd)"
VENV="${ROOT}/venv"
if [[ -d "${VENV}" ]]; then
  source "${VENV}/bin/activate"
fi
export PORT="${PORT:-8766}"
if [[ -n "$1" ]]; then
  export VAE_CKPT="$1"
  echo "Using checkpoint: ${VAE_CKPT}"
fi
echo "Starting live singer demo on http://0.0.0.0:${PORT}"
python server.py

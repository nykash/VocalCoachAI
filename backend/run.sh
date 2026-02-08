#!/usr/bin/env bash
# Run the FastAPI server from the project root so "backend" and "eda" resolve.
cd "$(dirname "$0")/.."
exec uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000 "$@"

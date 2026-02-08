#!/bin/bash
# Run the real-time breath support web app (backend + frontend)
# From project root or eda: ./run_breath_app.sh   or   cd eda && ./run_breath_app.sh
cd "$(dirname "$0")"
if [ -d "venv" ]; then
  source venv/bin/activate
fi
# Install flask if missing
pip install -q flask 2>/dev/null || true
export PORT="${PORT:-8765}"
echo "Breath support app: http://localhost:$PORT"
echo "Press Ctrl+C to stop."
exec python breath_app/server.py

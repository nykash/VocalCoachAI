"""
Real-time breath support backend. Serves the frontend and accepts audio chunks via POST /api/score.
Expects: body = raw Float32Array buffer (little-endian), header X-Sample-Rate (default 44100).
Returns: JSON with breath_support_score, calibrated_score, and feature breakdown.
"""

from __future__ import annotations

import io
import os
import sys

import numpy as np

# Allow importing breath_support from parent eda folder
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__, static_folder="static", static_url_path="")


@app.after_request
def cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp


# Load breath support and calibration once at startup
import breath_support as bs

CALIBRATION = None
_calibration_path = os.path.join(os.path.dirname(__file__), "..", "output", "breath_support_calibration.json")
if os.path.isfile(_calibration_path):
    CALIBRATION = bs.load_calibration(_calibration_path)


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/score", methods=["POST"])
def score():
    """Accept raw float32 audio (binary), return breath support score JSON."""
    try:
        data = request.get_data()
        if not data or len(data) < 1000:
            return jsonify({"error": "Audio too short"}), 400
        sr = int(request.headers.get("X-Sample-Rate", 44100))
        if sr <= 0 or sr > 192000:
            sr = 44100
        y = np.frombuffer(data, dtype=np.float32)
        if len(y) < sr * 0.5:
            return jsonify({"error": "Need at least 0.5 s of audio"}), 400
        result = bs.compute_breath_support_score(y, sr, calibration=CALIBRATION)
        # Return a JSON-serializable subset (no numpy types)
        out = {
            "breath_support_score": result["breath_support_score"],
            "hnr_db": _num(result.get("hnr_db")),
            "rms_cv": _num(result.get("rms_cv")),
            "calibrated_score": _num(result.get("calibrated_score")),
        }
        if out["calibrated_score"] is None and CALIBRATION:
            out["calibrated_score"] = 0.5
        return jsonify(out)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _num(x):
    if x is None:
        return None
    if isinstance(x, (int, float)) and (x != x or abs(x) == float("inf")):
        return None
    return float(x) if hasattr(x, "__float__") else x


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8765))
    app.run(host="0.0.0.0", port=port, debug=True, threaded=True)

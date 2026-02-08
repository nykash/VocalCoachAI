"""
Calibrate breath support so that:
- Singing chunks from MP3s in data/ score near 1.
- Chunks from all sound files in negative/ score near 0.

Usage:
  cd eda && source venv/bin/activate
  python calibrate_breath_support.py [--data DIR] [--negative DIR] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, *args, **kwargs):
        return it

# Run from eda/ so breath_support is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from breath_support import (
    RAW_FEATURE_KEYS,
    RAW_FEATURE_HIGHER_IS_BETTER,
    SR,
    apply_calibration,
    compute_raw_features,
    get_voiced_chunk_segments,
    load_audio,
)

DEFAULT_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DEFAULT_NEGATIVE_DIR = os.path.join(os.path.dirname(__file__), "negative")
DEFAULT_OUT_PATH = os.path.join(os.path.dirname(__file__), "output", "breath_support_calibration.json")


def _process_one_file(args: tuple) -> tuple[str, list[dict], str | None]:
    """Process one audio file: load, segment into voiced chunks, return (path, features, error)."""
    path, min_chunk_sec = args
    try:
        y, sr = load_audio(path, sr=SR)
    except Exception as e:
        return (path, [], str(e))
    segments = get_voiced_chunk_segments(y, sr, min_duration_sec=min_chunk_sec)
    features = []
    for s0, s1 in segments:
        start_sample = int(s0 * sr)
        end_sample = int(s1 * sr)
        chunk = y[start_sample:end_sample]
        if len(chunk) < sr * 0.5:
            continue
        try:
            raw = compute_raw_features(chunk, sr)
            features.append(raw)
        except Exception as e:
            print(f"  Chunk [{s0:.1f}-{s1:.1f}s] in {path}: {e}", file=sys.stderr)
    return (path, features, None)


def collect_data_chunk_features(
    data_dir: str,
    min_chunk_sec: float = 1.0,
    workers: int = 5,
) -> list[dict]:
    """Load each MP3 in data_dir, segment into voiced chunks, return list of raw feature dicts."""
    if not os.path.isdir(data_dir):
        return []
    exts = {".mp3", ".wav", ".m4a", ".flac"}
    paths = [
        os.path.join(data_dir, f)
        for f in sorted(os.listdir(data_dir))
        if os.path.splitext(f)[1].lower() in exts
    ]
    if not paths:
        return []
    workers = max(1, workers)
    all_features = []
    if workers == 1:
        for path in tqdm(paths, desc="Chunking", unit="file"):
            path, features, err = _process_one_file((path, min_chunk_sec))
            if err:
                print(f"Skip {path}: {err}", file=sys.stderr)
            else:
                all_features.extend(features)
    else:
        task_args = [(p, min_chunk_sec) for p in paths]
        with Pool(workers) as pool:
            results = list(tqdm(
                pool.imap(_process_one_file, task_args, chunksize=1),
                total=len(task_args),
                desc="Chunking",
                unit="file",
            ))
        for path, features, err in results:
            if err:
                print(f"Skip {path}: {err}", file=sys.stderr)
            else:
                all_features.extend(features)
    return all_features


def collect_chunk_features_from_dir(
    directory: str,
    min_chunk_sec: float = 1.0,
    workers: int = 5,
) -> list[dict]:
    """Load each audio file in directory, segment into voiced chunks, return list of raw feature dicts."""
    return collect_data_chunk_features(directory, min_chunk_sec=min_chunk_sec, workers=workers)


def fit_calibration(
    data_features: list[dict],
    negative_features: list[dict],
    data_percentile: float = 25.0,
    negative_percentile: float = 50.0,
) -> dict:
    """
    ref_high = "good" reference from data dir, ref_low = "poor" reference from negative dir.
    Uses percentiles: ref_high from data, ref_low = median (or negative_percentile) of negative chunks.
    """
    calibration = {}
    for key in RAW_FEATURE_KEYS:
        data_vals = []
        for d in data_features:
            v = d.get(key)
            if v is not None and isinstance(v, (int, float)) and np.isfinite(v):
                data_vals.append(float(v))
        neg_vals = []
        for d in negative_features:
            v = d.get(key)
            if v is not None and isinstance(v, (int, float)) and np.isfinite(v):
                neg_vals.append(float(v))
        if not data_vals:
            continue
        ref_high: float
        ref_low: float
        higher_better = RAW_FEATURE_HIGHER_IS_BETTER.get(key, True)
        pct = data_percentile if higher_better else (100.0 - data_percentile)
        ref_high = float(np.percentile(data_vals, pct))
        if neg_vals:
            ref_low = float(np.percentile(neg_vals, negative_percentile))
        else:
            ref_low = ref_high - 1e-6
        calibration[key] = {"ref_high": ref_high, "ref_low": ref_low}
    return calibration


def main():
    ap = argparse.ArgumentParser(
        description="Calibrate breath support: data/ chunks -> 1, negative/ chunks -> 0"
    )
    ap.add_argument("--data", default=DEFAULT_DATA_DIR, help="Directory of reference audio (good support)")
    ap.add_argument("--negative", default=DEFAULT_NEGATIVE_DIR, help="Directory of audio files (poor support)")
    ap.add_argument("--out", default=DEFAULT_OUT_PATH, help="Output calibration JSON path")
    ap.add_argument("--min-chunk", type=float, default=1.0, help="Min voiced chunk duration (sec)")
    ap.add_argument("--data-pct", type=float, default=25.0, help="Percentile for 'good' reference (data)")
    ap.add_argument("--negative-pct", type=float, default=50.0, help="Percentile for 'poor' reference (median of negative)")
    ap.add_argument("--workers", type=int, default=5, help="Number of parallel workers for chunking (default: 5)")
    args = ap.parse_args()

    print("Collecting features from data (voiced chunks only)...", flush=True)
    data_features = collect_data_chunk_features(
        args.data, min_chunk_sec=args.min_chunk, workers=args.workers
    )
    if not data_features:
        print("No chunks from data dir. Add audio to", args.data, file=sys.stderr)
        sys.exit(1)
    print(f"  Got {len(data_features)} chunks from {args.data}", flush=True)

    if not os.path.isdir(args.negative):
        print("Negative dir not found:", args.negative, file=sys.stderr)
        sys.exit(1)
    print("Collecting features from negative (voiced chunks only)...", flush=True)
    negative_features = collect_chunk_features_from_dir(
        args.negative, min_chunk_sec=args.min_chunk, workers=args.workers
    )
    if not negative_features:
        print("No chunks from negative dir. Add audio files to", args.negative, file=sys.stderr)
        sys.exit(1)
    print(f"  Got {len(negative_features)} chunks from {args.negative}", flush=True)

    calibration = fit_calibration(
        data_features,
        negative_features,
        data_percentile=args.data_pct,
        negative_percentile=args.negative_pct,
    )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(calibration, f, indent=2)
    print(f"Wrote calibration to {args.out}", flush=True)

    # Sanity check
    data_scores = [apply_calibration(d, calibration) for d in data_features]
    negative_scores = [apply_calibration(d, calibration) for d in negative_features]
    print(f"  Data chunks:    calibrated_score mean={np.mean(data_scores):.3f} median={np.median(data_scores):.3f} min={np.min(data_scores):.3f} max={np.max(data_scores):.3f}")
    print(f"  Negative chunks: calibrated_score mean={np.mean(negative_scores):.3f} median={np.median(negative_scores):.3f} min={np.min(negative_scores):.3f} max={np.max(negative_scores):.3f}")
    print("Done. Use calibration in breath_support via load_calibration().", flush=True)


if __name__ == "__main__":
    main()

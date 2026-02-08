"""
For each singer listed in singers.txt, find all songs in the data folder whose
filename contains the singer name (case insensitive). Then run KNN classification
where the class is the artist and the features are the FFT peak ratios (2nd/1st,
3rd/1st) per note for each song.
"""
import json
import os
import pickle
import sys

# Avoid joblib/loky warning when physical core count cannot be detected (e.g. no sysctl)
if 'LOKY_MAX_CPU_COUNT' not in os.environ:
    os.environ['LOKY_MAX_CPU_COUNT'] = '8'
from collections import Counter
from multiprocessing import Pool
from typing import Optional

import numpy as np

# Ensure we can import from analyze when run from project root or eda
_EDA_DIR = os.path.dirname(os.path.abspath(__file__))
if _EDA_DIR not in sys.path:
    sys.path.insert(0, _EDA_DIR)

from analyze import (
    load_audio,
    fft_over_time,
    pitch_and_notes,
    spectral_peaks_per_frame,
    frame_spectral_centroid,
    frame_harmonic_ratios,
    build_note_ratio_vectors,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, *args, **kwargs):
        return it


AUDIO_EXTENSIONS = {'.mp3', '.wav', '.flac', '.m4a', '.ogg'}
N_WORKERS = 5
CACHE_DIR_NAME = 'knn_ratio_cache'
MANIFEST_FNAME = 'manifest.json'
RESULTS_FNAME = 'results.pkl'


def _process_one_track(path: str):
    """
    Process a single audio file: FFT, pitch, spectral peaks, build note_timbre.
    Returns result dict (same shape as one element of peaks_per_note_per_track) or None on failure.
    Used by multiprocessing workers; do not use tqdm here.
    """
    name = os.path.splitext(os.path.basename(path))[0]
    try:
        y, sr = load_audio(path)
    except Exception:
        return None
    S, freqs, _ = fft_over_time(y, sr)
    f0, notes, voiced = pitch_and_notes(y, sr)
    frame_peaks = spectral_peaks_per_frame(S, freqs, show_progress=False)

    note_timbre = {}
    n_freqs = S.shape[0]
    for t, (peak_freqs, peak_mags) in enumerate([(pf, pm) for _, pf, pm in frame_peaks]):
        if t >= len(notes) or not voiced[t] or notes[t] is None:
            continue
        note = str(notes[t])
        if note not in note_timbre:
            note_timbre[note] = {
                'spectral_centroid': [],
                'ratio_2nd': [],
                'ratio_3rd': [],
                'spectrum_sum': np.zeros(n_freqs),
                'spectrum_count': 0,
            }
        cent = frame_spectral_centroid(S[:, t], freqs)
        if np.isfinite(cent):
            note_timbre[note]['spectral_centroid'].append(cent)
        ratios = frame_harmonic_ratios(peak_mags, n_ratios=3)
        if len(ratios) >= 1:
            note_timbre[note]['ratio_2nd'].append(ratios[0])
        if len(ratios) >= 2:
            note_timbre[note]['ratio_3rd'].append(ratios[1])
        note_timbre[note]['spectrum_sum'] += S[:, t]
        note_timbre[note]['spectrum_count'] += 1

    for k in list(note_timbre.keys()):
        nt = note_timbre[k]
        if nt['spectrum_count'] == 0:
            del note_timbre[k]
            continue
        note_timbre[k] = {
            'spectral_centroid': np.array(nt['spectral_centroid']),
            'ratio_2nd': np.array(nt['ratio_2nd']),
            'ratio_3rd': np.array(nt['ratio_3rd']),
            'spectrum_sum': nt['spectrum_sum'],
            'spectrum_count': nt['spectrum_count'],
        }
    return {'path': path, 'name': name, 'note_timbre': note_timbre, 'freqs': freqs}


def get_all_audio_paths(root: str):
    """Return all audio file paths under `root` (flat list)."""
    paths = []
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if not os.path.isfile(path):
            continue
        if os.path.splitext(name)[1].lower() in AUDIO_EXTENSIONS:
            paths.append(path)
    return paths


def load_singers(singers_path: str):
    """Load singer names from file, one per line, strip whitespace, skip empty."""
    with open(singers_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def assign_singer_to_path(path: str, singers: list) -> Optional[str]:
    """
    If the filename (case-insensitive) contains a singer name, return that singer.
    Prefer longest match so 'Ariana Grande' wins over 'Ariana'.
    """
    name = os.path.basename(path)
    name_lower = name.lower()
    matched = None
    for s in singers:
        if s.lower() in name_lower:
            if matched is None or len(s) > len(matched):
                matched = s
    return matched


def find_songs_by_singers(data_dir: str, singers_path: str):
    """
    Find all audio paths that match at least one singer. Returns list of
    (path, artist) and the list of singers (for reporting).
    """
    singers = load_singers(singers_path)
    if not singers:
        return [], singers
    all_paths = get_all_audio_paths(data_dir)
    path_artist = []
    for path in all_paths:
        artist = assign_singer_to_path(path, singers)
        if artist is not None:
            path_artist.append((path, artist))
    return path_artist, singers


def _cache_paths_and_mtimes(audio_paths: list) -> list:
    """Return list of (abs_path, mtime) for cache manifest."""
    return [
        (os.path.abspath(p), os.path.getmtime(p))
        for p in audio_paths
    ]


def _load_cached_results(audio_paths: list, cache_dir: str) -> Optional[list]:
    """
    If cache exists and manifest matches current paths and mtimes,
    load and return list of results (same length as audio_paths, entries may be None).
    Otherwise return None.
    """
    manifest_file = os.path.join(cache_dir, MANIFEST_FNAME)
    results_file = os.path.join(cache_dir, RESULTS_FNAME)
    if not os.path.isfile(manifest_file) or not os.path.isfile(results_file):
        return None
    current = _cache_paths_and_mtimes(audio_paths)
    try:
        with open(manifest_file, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
        cached_paths = manifest['paths']
        cached_mtimes = manifest['mtimes']
        if len(cached_paths) != len(audio_paths):
            return None
        if cached_paths != [p for p, _ in current] or cached_mtimes != [t for _, t in current]:
            return None
        with open(results_file, 'rb') as f:
            return pickle.load(f)
    except (json.JSONDecodeError, KeyError, pickle.PickleError):
        return None


def _save_cached_results(audio_paths: list, raw_results: list, cache_dir: str) -> None:
    """Save manifest (paths + mtimes) and pickled raw results."""
    os.makedirs(cache_dir, exist_ok=True)
    current = _cache_paths_and_mtimes(audio_paths)
    manifest = {
        'paths': [p for p, _ in current],
        'mtimes': [t for _, t in current],
    }
    with open(os.path.join(cache_dir, MANIFEST_FNAME), 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=0)
    with open(os.path.join(cache_dir, RESULTS_FNAME), 'wb') as f:
        pickle.dump(raw_results, f)


def main():
    eda_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(eda_dir, 'data')
    singers_path = os.path.join(eda_dir, 'singers.txt')

    if not os.path.isdir(data_dir):
        print(f"Data directory not found: {data_dir}")
        return
    if not os.path.isfile(singers_path):
        print(f"singers.txt not found: {singers_path}")
        return

    path_artist, singers = find_songs_by_singers(data_dir, singers_path)
    if not path_artist:
        print("No songs matched any singer in singers.txt")
        return

    # Count per artist; keep only artists with at least 2 songs
    counts = Counter(a for _, a in path_artist)
    path_artist = [(p, a) for p, a in path_artist if counts[a] >= 2]
    if not path_artist:
        print("Need at least 2 songs per artist for classification. Current counts:", dict(counts))
        return

    audio_paths = [p for p, _ in path_artist]
    path_to_artist = dict(path_artist)
    cache_dir = os.path.join(eda_dir, 'output', CACHE_DIR_NAME)

    print(f"Singers: {singers}")
    print(f"Total songs matched: {len(audio_paths)}")
    print(f"Per artist: {dict(Counter(a for _, a in path_artist))}")

    raw = _load_cached_results(audio_paths, cache_dir)
    if raw is not None:
        print("Loading FFT peak ratios from cache...")
    else:
        print(f"Extracting FFT peak ratios per note using {N_WORKERS} workers...")
        with Pool(N_WORKERS) as pool:
            raw = list(tqdm(
                pool.imap(_process_one_track, audio_paths),
                total=len(audio_paths),
                desc="Tracks",
                unit="track",
            ))
        _save_cached_results(audio_paths, raw, cache_dir)
        print(f"Cache saved to {cache_dir}")

    results = [r for r in raw if r is not None]
    for r in results:
        r['artist'] = path_to_artist.get(r['path'])

    results = [r for r in results if r.get('artist') is not None]
    if not results:
        print("No tracks could be processed.")
        return

    note_order, name_vectors = build_note_ratio_vectors(results)
    if not name_vectors:
        print("Could not build feature vectors (no notes).")
        return

    X = np.array([v for _, v in name_vectors], dtype=np.float64)
    y = [r['artist'] for r in results]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    n_classes = len(le.classes_)

    print(f"Feature dim: {X.shape[1]} (notes × 2 ratios)")
    print(f"Classes: {list(le.classes_)}")

    # Train/test split and fit KNN for k = 1 .. 20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.25, random_state=42, stratify=y_enc
    )
    k_range = range(1, 21)
    best_k, best_acc = 1, 0.0
    print("\nKNN test accuracy for k = 1 .. 20:")
    for k in k_range:
        clf = KNeighborsClassifier(n_neighbors=k, weights='distance', metric='euclidean')
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        if acc > best_acc:
            best_acc, best_k = acc, k
        print(f"  k={k:2d}  test acc = {acc:.3f}")

    # Full report for best k
    k = best_k
    clf = KNeighborsClassifier(n_neighbors=k, weights='distance', metric='euclidean')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    labels = np.arange(len(le.classes_))
    print(f"\nBest k = {k} (test accuracy {best_acc:.3f})")
    print("\nClassification report:")
    print(classification_report(
        y_test, y_pred, labels=labels, target_names=list(le.classes_),
        zero_division=0,
    ))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred, labels=labels))

    # Cross-validation on full dataset for best k
    cv_scores = cross_val_score(clf, X, y_enc, cv=min(5, len(y) // 2), scoring='accuracy')
    print(f"\nCV accuracy (mean ± std): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")


if __name__ == '__main__':
    main()

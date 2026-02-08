import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
from scipy.signal import find_peaks

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, *args, **kwargs):
        return it

# Root: project directory; change to your audio folder if needed (e.g. '/Users/nikash/Music/Downloaded by MediaHuman')
ROOT = "."#/Users/nikash/Music/Downloaded by MediaHuman"
AUDIO_EXTENSIONS = {'.mp3', '.wav', '.flac', '.m4a', '.ogg'}
MAX_TRACKS = 10
SR = 22050
HOP_LENGTH = 512
N_FFT = 2048
FMIN = librosa.note_to_hz('C2')
FMAX = librosa.note_to_hz('C7')
PEAK_PROMINENCE = 0.1  # relative prominence for spectral peaks
MAX_PEAKS_PER_FRAME = 8
# Use piptrack instead of pyin for ~5–10x faster pitch tracking (slightly less robust)
FAST_PITCH = True


def get_audio_paths(root: str, limit: int = MAX_TRACKS):
    """Return first `limit` audio file paths under `root` (flat list)."""
    paths = []
    for name in sorted(os.listdir(root)):
        if len(paths) >= limit:
            break
        path = os.path.join(root, name)
        if not os.path.isfile(path):
            continue
        if os.path.splitext(name)[1].lower() in AUDIO_EXTENSIONS:
            paths.append(path)
    return paths


def load_audio(path: str):
    """Load mono audio and sample rate (resampling is slow for long files)."""
    y, sr = librosa.load(path, sr=SR, mono=True)
    return y, sr


def fft_over_time(y: np.ndarray, sr: int):
    """STFT magnitude (FFT over time). Returns S (freq x time), freqs, times."""
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=HOP_LENGTH)
    return S, freqs, times


def pitch_and_notes(y: np.ndarray, sr: int):
    """Estimate F0 per frame and map to note names (voiced only)."""
    if FAST_PITCH:
        pitches, mags = librosa.piptrack(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, fmin=FMIN, fmax=FMAX)
        # take per-frame dominant pitch (max magnitude in valid range)
        f0 = np.nan * np.ones(pitches.shape[1])
        voiced_flag = np.zeros(pitches.shape[1], dtype=bool)
        for t in range(pitches.shape[1]):
            p, m = pitches[:, t], mags[:, t]
            valid = (p >= FMIN) & (p <= FMAX) & (m > 0)
            if np.any(valid):
                idx = np.nanargmax(np.where(valid, m, np.nan))
                f0[t] = p[idx]
                voiced_flag[t] = True
        notes = np.array([librosa.hz_to_note(h) if np.isfinite(h) and h > 0 else None for h in f0])
    else:
        f0, voiced_flag, _ = librosa.pyin(
            y, sr=sr, fmin=FMIN, fmax=FMAX,
            hop_length=HOP_LENGTH, frame_length=N_FFT
        )
        notes = np.array([librosa.hz_to_note(h) if np.isfinite(h) and h > 0 else None for h in f0])
    return f0, notes, voiced_flag


def spectral_peaks_per_frame(S: np.ndarray, freqs: np.ndarray, prominence: float = PEAK_PROMINENCE, show_progress: bool = True):
    """
    For each time frame, find peak frequencies in the magnitude spectrum.
    Returns list of (frame_idx, peak_freqs, peak_mags) per frame.
    """
    n_frames = S.shape[1]
    iterator = range(n_frames)
    if show_progress:
        iterator = tqdm(iterator, desc="Peaks", unit="frame", leave=False)
    frame_peaks = []
    for t in iterator:
        mag = S[:, t]
        if mag.max() <= 0:
            frame_peaks.append((t, np.array([]), np.array([])))
            continue
        p = prominence * mag.max()
        peaks, props = find_peaks(mag, prominence=p, height=0)
        if len(peaks) > MAX_PEAKS_PER_FRAME:
            order = np.argsort(mag[peaks])[::-1][:MAX_PEAKS_PER_FRAME]
            peaks = peaks[order]
        peak_freqs = freqs[peaks]
        peak_mags = mag[peaks]
        frame_peaks.append((t, peak_freqs, peak_mags))
    return frame_peaks


def frame_spectral_centroid(mag: np.ndarray, freqs: np.ndarray) -> float:
    """Spectral centroid (brightness) for one frame. Hz."""
    total = mag.sum()
    if total <= 0:
        return np.nan
    return np.sum(freqs * mag) / total


def frame_harmonic_ratios(peak_mags: np.ndarray, n_ratios: int = 3) -> np.ndarray:
    """
    Strength of higher peaks relative to strongest (timbre: harmonic balance).
    Returns ratios [2nd/1st, 3rd/1st, ...] or shorter if fewer peaks; empty if < 2 peaks.
    """
    if len(peak_mags) < 2:
        return np.array([])
    order = np.argsort(peak_mags)[::-1]
    sorted_mag = peak_mags[order]
    m0 = sorted_mag[0]
    if m0 <= 0:
        return np.array([])
    n = min(n_ratios, len(sorted_mag) - 1)
    return sorted_mag[1 : 1 + n] / m0


def peaks_per_note_per_track(audio_paths: list):
    """
    For each track: FFT over time, pitch/notes, spectral peaks; group by note.
    For timbre we store per (track, note): spectral centroids and harmonic ratios
    (2nd/1st, 3rd/1st peak magnitude) per frame. Returns list of dicts with
    'note_timbre': { note: { 'spectral_centroid': [], 'ratio_2nd': [], 'ratio_3rd': [] } }.
    """
    results = []
    for path in tqdm(audio_paths, desc="Tracks", unit="track"):
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            y, sr = load_audio(path)
        except Exception as e:
            tqdm.write(f"Skip {path}: {e}")
            continue
        S, freqs, _ = fft_over_time(y, sr)
        f0, notes, voiced = pitch_and_notes(y, sr)
        frame_peaks = spectral_peaks_per_frame(S, freqs)

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
        results.append({'path': path, 'name': name, 'note_timbre': note_timbre, 'freqs': freqs})
    return results


def _sorted_notes(results: list, key: str = 'note_timbre'):
    all_notes = set()
    for r in results:
        all_notes.update(r[key].keys())
    def _note_sort_key(n):
        try:
            m = librosa.note_to_midi(n)
            return (m,) if np.isfinite(m) else (-1,)
        except Exception:
            return (-1,)
    return sorted(all_notes, key=_note_sort_key)


def plot_timbre_by_note(
    results: list,
    metric: str,
    ylabel: str,
    out_path: str = None,
):
    """
    Compare timbre across tracks: one subplot per note, violin of metric by track.
    metric: 'spectral_centroid' (brightness) or 'ratio_2nd' / 'ratio_3rd' (harmonic balance).
    """
    all_notes = _sorted_notes(results)
    if not all_notes:
        print("No notes found.")
        return

    n_notes = len(all_notes)
    n_cols = min(4, n_notes)
    n_rows = (n_notes + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False)
    axes = axes.flatten()

    for idx, note in enumerate(all_notes):
        ax = axes[idx]
        values = []
        track_names = []
        for r in results:
            if note not in r['note_timbre'] or metric not in r['note_timbre'][note]:
                continue
            arr = r['note_timbre'][note][metric]
            arr = arr[np.isfinite(arr)]
            if len(arr) == 0:
                continue
            values.append(arr)
            track_names.extend([r['name'][:20]] * len(arr))
        if not values:
            ax.set_visible(False)
            continue
        df = pd.DataFrame({metric: np.concatenate(values), 'track': track_names})
        sns.violinplot(data=df, x='track', y=metric, ax=ax, cut=0)
        ax.set_title(f'Note: {note}')
        ax.set_ylabel(ylabel)
        ax.tick_params(axis='x', rotation=45)

    for j in range(idx + 1, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Saved {out_path}")
    plt.show()


def _sanitize_fname(s: str) -> str:
    """Replace characters unsafe for filenames."""
    return "".join(c if c.isalnum() or c in "._- " else "_" for c in s).strip() or "song"


def plot_per_song(result: dict, out_dir: str):
    """
    One figure per song: for each note, show average FFT spectrum and mean
    harmonic ratios (2nd/1st, 3rd/1st).
    """
    freqs = result["freqs"]
    note_timbre = result["note_timbre"]
    name = result["name"]
    if not note_timbre:
        return
    def _note_sort_key(n):
        try:
            m = librosa.note_to_midi(n)
            return (m,) if np.isfinite(m) else (-1,)
        except Exception:
            return (-1,)
    notes = sorted(note_timbre.keys(), key=_note_sort_key)

    n_notes = len(notes)
    n_cols = min(3, n_notes)
    n_rows = (n_notes + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
    axes = axes.flatten()

    for idx, note in enumerate(notes):
        ax = axes[idx]
        nt = note_timbre[note]
        count = nt["spectrum_count"]
        if count == 0:
            ax.set_visible(False)
            continue
        avg_spectrum = nt["spectrum_sum"] / count
        ax.plot(freqs, avg_spectrum, color="steelblue", linewidth=0.8)
        ax.set_title(f"Note: {note} (n={count} frames)")
        ax.set_ylabel("Avg magnitude")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_xlim(0, min(4000, freqs[-1]))
        ax.grid(True, alpha=0.3)

        mean_r2 = np.mean(nt["ratio_2nd"]) if len(nt["ratio_2nd"]) else np.nan
        mean_r3 = np.mean(nt["ratio_3rd"]) if len(nt["ratio_3rd"]) else np.nan
        txt = f"2nd/1st: {mean_r2:.2f}\n3rd/1st: {mean_r3:.2f}"
        ax.text(0.98, 0.97, txt, transform=ax.transAxes, fontsize=9, verticalalignment="top", horizontalalignment="right", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    for j in range(len(notes), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(f"Song: {name}", fontsize=12)
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"per_song_{_sanitize_fname(name)}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def build_note_ratio_vectors(results: list):
    """
    Build one vector per song: for each note (in canonical order), store
    [mean ratio_2nd, mean ratio_3rd]. Missing notes get 0. Returns note_order, list of (name, vector).
    """
    note_order = _sorted_notes(results)
    if not note_order:
        return [], []
    dim = 2 * len(note_order)  # ratio_2nd, ratio_3rd per note
    name_vectors = []
    for r in results:
        vec = np.zeros(dim, dtype=np.float64)
        for i, note in enumerate(note_order):
            if note not in r["note_timbre"]:
                continue
            nt = r["note_timbre"][note]
            r2 = np.mean(nt["ratio_2nd"]) if len(nt["ratio_2nd"]) else 0.0
            r3 = np.mean(nt["ratio_3rd"]) if len(nt["ratio_3rd"]) else 0.0
            vec[2 * i] = r2
            vec[2 * i + 1] = r3
        name_vectors.append((r["name"], vec))
    return note_order, name_vectors


def save_ratio_vectors(note_order: list, name_vectors: list, out_dir: str):
    """Save each song's ratio vector as .npy and a manifest with note_order and names."""
    vectors_dir = os.path.join(out_dir, "vectors")
    os.makedirs(vectors_dir, exist_ok=True)
    names = []
    for name, vec in name_vectors:
        fname = _sanitize_fname(name) + ".npy"
        np.save(os.path.join(vectors_dir, fname), vec)
        names.append(name)
    np.savez(
        os.path.join(vectors_dir, "manifest.npz"),
        note_order=np.array(note_order, dtype=object),
        names=np.array(names, dtype=object),
    )
    print(f"Saved {len(name_vectors)} vectors to {vectors_dir}")


def cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity between rows. Zero vectors get similarity 0."""
    n = len(vectors)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # avoid div by zero; those rows will get 0 sim
    unit = vectors / norms
    sim = np.dot(unit, unit.T)
    np.clip(sim, -1.0, 1.0, out=sim)
    return sim


def compute_and_save_similarity(
    note_order: list,
    name_vectors: list,
    out_dir: str,
):
    """Compute pairwise cosine similarity, save matrix and heatmap."""
    if not name_vectors:
        return
    names = [n for n, _ in name_vectors]
    vectors = np.array([v for _, v in name_vectors], dtype=np.float64)
    sim = cosine_similarity_matrix(vectors)
    np.save(os.path.join(out_dir, "similarity_matrix.npy"), sim)
    # CSV with labels
    df = pd.DataFrame(sim, index=names, columns=names)
    df.to_csv(os.path.join(out_dir, "similarity_matrix.csv"))
    print(f"Saved similarity matrix to {out_dir}")

    # Heatmap
    fig, ax = plt.subplots(figsize=(max(6, len(names) * 0.8), max(5, len(names) * 0.6)))
    short_names = [n[:25] for n in names]
    sns.heatmap(sim, xticklabels=short_names, yticklabels=short_names, ax=ax, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_title("Cosine similarity (peak ratios per note)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "similarity_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved similarity heatmap to {out_dir}/similarity_heatmap.png")


def plot_timbre_comparison(results: list, out_dir: str):
    """
    Generate timbre comparison plots: brightness (spectral centroid) and
    harmonic balance (2nd/1st, 3rd/1st peak ratio) per note across tracks.
    """
    plots = [
        ('spectral_centroid', 'Spectral centroid (Hz) — brightness', 'timbre_brightness_by_note.png'),
        ('ratio_2nd', '2nd peak / 1st peak — harmonic balance', 'timbre_harmonic_ratio_2nd_by_note.png'),
        ('ratio_3rd', '3rd peak / 1st peak — harmonic balance', 'timbre_harmonic_ratio_3rd_by_note.png'),
    ]
    for metric, ylabel, fname in tqdm(plots, desc="Plots", unit="plot"):
        plot_timbre_by_note(
            results,
            metric=metric,
            ylabel=ylabel,
            out_path=os.path.join(out_dir, fname),
        )


def main():
    audio_paths = get_audio_paths(ROOT)
    if not audio_paths:
        print(f"No audio files found under {ROOT}. Supported: {AUDIO_EXTENSIONS}")
        return
    print(f"Using root: {ROOT}")
    print(f"Processing first {len(audio_paths)} files (FAST_PITCH={FAST_PITCH})")
    print("Slow parts: loading/resampling audio, pitch tracking, and per-frame peak detection. Progress bars below.\n")

    results = peaks_per_note_per_track(audio_paths)
    if not results:
        print("No tracks could be processed.")
        return

    out_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(out_dir, exist_ok=True)
    plot_timbre_comparison(results, out_dir)

    for r in tqdm(results, desc="Per-song graphs", unit="song"):
        plot_per_song(r, out_dir)

    note_order, name_vectors = build_note_ratio_vectors(results)
    if name_vectors:
        save_ratio_vectors(note_order, name_vectors, out_dir)
        compute_and_save_similarity(note_order, name_vectors, out_dir)


if __name__ == '__main__':
    main()

"""
Detect pauses (silence) and breaths while singing from audio files.

- Pauses: contiguous low-energy (silence) regions lasting at least min_pause_duration.
- Breaths: short unvoiced segments with moderate energy (noise-like inhale/exhale).

Uses librosa for loading and frame-level RMS/STFT; voicing from piptrack to separate
singing from breath-like noise.
"""

import argparse
import os
import numpy as np
import librosa

# Match analyze.py for consistency
SR = 22050
HOP_LENGTH = 512
N_FFT = 2048
FMIN = librosa.note_to_hz("C2")
FMAX = librosa.note_to_hz("C7")

# Detection thresholds (tune for your material)
SILENCE_RMS_THRESHOLD = 0.01  # RMS below this = silence
BREATH_RMS_MIN = 0.005  # Breath has at least this much energy (above silence)
BREATH_RMS_MAX = 0.08   # Breath typically below singing level
MIN_PAUSE_DURATION_SEC = 0.15  # Merge silence gaps shorter than this into adjacent
MIN_BREATH_DURATION_SEC = 0.08
MAX_BREATH_DURATION_SEC = 1.2   # Long "breaths" may be pauses; cap as breath


def load_audio(path: str):
    """Load mono audio at fixed sample rate."""
    y, sr = librosa.load(path, sr=SR, mono=True)
    return y, sr


def frame_times(n_frames: int):
    """Time in seconds for each frame index."""
    return librosa.frames_to_time(np.arange(n_frames), sr=SR, hop_length=HOP_LENGTH)


def get_rms(y: np.ndarray) -> np.ndarray:
    """Per-frame RMS energy."""
    return librosa.feature.rms(y=y, hop_length=HOP_LENGTH, frame_length=N_FFT)[0]


def get_voiced_flag(y: np.ndarray) -> np.ndarray:
    """Per-frame: True if frame has a clear pitch in singing range (voiced)."""
    pitches, mags = librosa.piptrack(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, fmin=FMIN, fmax=FMAX
    )
    voiced = np.zeros(pitches.shape[1], dtype=bool)
    for t in range(pitches.shape[1]):
        p, m = pitches[:, t], mags[:, t]
        valid = (p >= FMIN) & (p <= FMAX) & (m > 0)
        if np.any(valid):
            idx = np.nanargmax(np.where(valid, m, np.nan))
            if np.isfinite(p[idx]) and p[idx] > 0:
                voiced[t] = True
    return voiced


def merge_adjacent_segments(segments: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Merge segments that are adjacent or overlapping; return sorted (start, end) list."""
    if not segments:
        return []
    sorted_seg = sorted(segments, key=lambda x: x[0])
    out = [list(sorted_seg[0])]
    for s, e in sorted_seg[1:]:
        if s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(a, b) for a, b in out]


def detect_pauses(rms: np.ndarray, times: np.ndarray) -> list[dict]:
    """Detect contiguous silence (pause) regions."""
    is_silent = rms < SILENCE_RMS_THRESHOLD
    n = len(is_silent)
    segments = []
    i = 0
    while i < n:
        if not is_silent[i]:
            i += 1
            continue
        start = times[i]
        while i < n and is_silent[i]:
            i += 1
        end = times[i - 1] if i > 0 else times[0]
        # Use frame duration for end of last frame
        frame_dur = (times[1] - times[0]) if n > 1 else 1.0 / SR
        end = end + frame_dur
        if end - start >= MIN_PAUSE_DURATION_SEC:
            segments.append({"start": start, "end": end, "type": "pause"})
    return segments


def detect_breaths(
    rms: np.ndarray,
    times: np.ndarray,
    voiced: np.ndarray,
) -> list[dict]:
    """Detect breath-like segments: unvoiced, short, moderate energy."""
    # Candidate: unvoiced and RMS in [BREATH_RMS_MIN, BREATH_RMS_MAX]
    is_breath_candidate = (
        ~voiced
        & (rms >= BREATH_RMS_MIN)
        & (rms <= BREATH_RMS_MAX)
    )
    n = len(is_breath_candidate)
    frame_dur = (times[1] - times[0]) if n > 1 else 1.0 / SR
    segments = []
    i = 0
    while i < n:
        if not is_breath_candidate[i]:
            i += 1
            continue
        start = times[i]
        while i < n and is_breath_candidate[i]:
            i += 1
        end_idx = i - 1
        end = times[end_idx] + frame_dur
        duration = end - start
        if MIN_BREATH_DURATION_SEC <= duration <= MAX_BREATH_DURATION_SEC:
            segments.append({"start": start, "end": end, "type": "breath"})
    return segments


def run_detection(
    audio_path: str,
    min_pause_duration_sec: float = MIN_PAUSE_DURATION_SEC,
    silence_threshold: float = SILENCE_RMS_THRESHOLD,
    min_breath_sec: float = MIN_BREATH_DURATION_SEC,
    max_breath_sec: float = MAX_BREATH_DURATION_SEC,
) -> tuple[list[dict], np.ndarray, np.ndarray, np.ndarray]:
    """
    Run pause and breath detection on one file.
    Returns (events, times, rms, voiced).
    """
    y, _ = load_audio(audio_path)
    rms = get_rms(y)
    voiced = get_voiced_flag(y)
    times = frame_times(len(rms))

    # Optionally override globals for this run
    global MIN_PAUSE_DURATION_SEC, SILENCE_RMS_THRESHOLD
    global MIN_BREATH_DURATION_SEC, MAX_BREATH_DURATION_SEC
    orig_min_pause, orig_silence = MIN_PAUSE_DURATION_SEC, SILENCE_RMS_THRESHOLD
    orig_min_breath, orig_max_breath = MIN_BREATH_DURATION_SEC, MAX_BREATH_DURATION_SEC
    MIN_PAUSE_DURATION_SEC = min_pause_duration_sec
    SILENCE_RMS_THRESHOLD = silence_threshold
    MIN_BREATH_DURATION_SEC = min_breath_sec
    MAX_BREATH_DURATION_SEC = max_breath_sec

    try:
        pauses = detect_pauses(rms, times)
        breaths = detect_breaths(rms, times, voiced)
    finally:
        MIN_PAUSE_DURATION_SEC, SILENCE_RMS_THRESHOLD = orig_min_pause, orig_silence
        MIN_BREATH_DURATION_SEC, MAX_BREATH_DURATION_SEC = orig_min_breath, orig_max_breath

    # Merge overlapping/adjacent and sort by start
    all_events = merge_adjacent_segments(
        [(e["start"], e["end"]) for e in pauses]
    )
    pause_set = set((e["start"], e["end"]) for e in pauses)
    events = []
    for start, end in all_events:
        events.append({"start": start, "end": end, "type": "pause"})
    for b in breaths:
        # Avoid double-counting if breath falls inside a pause
        overlap = any(
            b["start"] < pe and b["end"] > ps
            for ps, pe in all_events
        )
        if not overlap:
            events.append(b)
    events.sort(key=lambda e: e["start"])
    return events, times, rms, voiced


def main():
    parser = argparse.ArgumentParser(
        description="Detect pauses and breaths in singing audio."
    )
    parser.add_argument(
        "audio",
        nargs="?",
        default=os.path.join(os.path.dirname(__file__), "data"),
        help="Path to audio file or directory (default: eda/data)",
    )
    parser.add_argument(
        "--min-pause",
        type=float,
        default=MIN_PAUSE_DURATION_SEC,
        help="Minimum pause duration in seconds",
    )
    parser.add_argument(
        "--silence-threshold",
        type=float,
        default=SILENCE_RMS_THRESHOLD,
        help="RMS below this is considered silence",
    )
    parser.add_argument(
        "--min-breath",
        type=float,
        default=MIN_BREATH_DURATION_SEC,
        help="Minimum breath segment duration (sec)",
    )
    parser.add_argument(
        "--max-breath",
        type=float,
        default=MAX_BREATH_DURATION_SEC,
        help="Maximum breath segment duration (sec)",
    )
    parser.add_argument(
        "-o", "--output",
        help="Write events to CSV (columns: start, end, type)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot RMS and mark pauses/breaths (requires matplotlib)",
    )
    args = parser.parse_args()

    paths = []
    if os.path.isfile(args.audio):
        paths = [args.audio]
    elif os.path.isdir(args.audio):
        for name in sorted(os.listdir(args.audio)):
            p = os.path.join(args.audio, name)
            if os.path.isfile(p) and os.path.splitext(name)[1].lower() in {".mp3", ".wav", ".flac", ".m4a", ".ogg"}:
                paths.append(p)
    if not paths:
        print("No audio file(s) found.")
        return

    all_rows = []
    for path in paths:
        print(f"Processing: {path}")
        events, times, rms, voiced = run_detection(
            path,
            min_pause_duration_sec=args.min_pause,
            silence_threshold=args.silence_threshold,
            min_breath_sec=args.min_breath,
            max_breath_sec=args.max_breath,
        )
        print(f"  Pauses: {sum(1 for e in events if e['type'] == 'pause')}")
        print(f"  Breaths: {sum(1 for e in events if e['type'] == 'breath')}")
        for e in events:
            print(f"    {e['type']:6} {e['start']:.2f} – {e['end']:.2f} s")
            all_rows.append({"path": path, "start": e["start"], "end": e["end"], "type": e["type"]})

        if args.plot:
            try:
                import matplotlib.pyplot as plt
                from matplotlib.patches import Patch
            except ImportError:
                print("  (install matplotlib for --plot)")
            else:
                fig, ax = plt.subplots(1, 1, figsize=(12, 3))
                ax.fill_between(times, 0, rms, color="gray", alpha=0.6, label="RMS")
                for e in events:
                    color = "blue" if e["type"] == "pause" else "green"
                    ax.axvspan(e["start"], e["end"], alpha=0.3, color=color)
                legend_handles = [
                    Patch(facecolor="blue", alpha=0.3, label="Pause"),
                    Patch(facecolor="green", alpha=0.3, label="Breath"),
                ]
                ax.legend(handles=[Patch(facecolor="gray", alpha=0.6, label="RMS")] + legend_handles, loc="upper right")
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("RMS")
                ax.set_title(os.path.basename(path))
                ax.set_xlim(0, times[-1])
                plt.tight_layout()
                out_path = path + "_pauses_breaths.png"
                plt.savefig(out_path, dpi=150)
                plt.close()
                print(f"  Plot saved: {out_path}")

    if args.output and all_rows:
        import csv
        with open(args.output, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["path", "start", "end", "type"])
            w.writeheader()
            w.writerows(all_rows)
        print(f"Wrote CSV: {args.output}")


if __name__ == "__main__":
    main()

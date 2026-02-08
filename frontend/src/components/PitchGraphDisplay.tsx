import { useEffect, useRef } from "react";
import { PitchResult, frequencyToNote } from "@/lib/pitchDetection";
import { cn } from "@/lib/utils";

function formatFreqAsNote(freq: number): string {
  const { noteLabel, noteLabelFlat } = frequencyToNote(freq);
  return noteLabel === noteLabelFlat ? noteLabel : `${noteLabel}/${noteLabelFlat}`;
}

/** Octave-invariant pitch distance in cents (0 = same note in any octave, max 600). */
function pitchClassCents(freq1: number, freq2: number): number {
  const cents = 1200 * Math.log2(freq1 / freq2);
  const mod = ((cents % 1200) + 1200) % 1200;
  return Math.min(mod, 1200 - mod);
}

interface PitchGraphDisplayProps {
  targetPitch: PitchResult | null;
  userPitch: PitchResult | null;
  isListening: boolean;
  isPlaying: boolean;
}

/** Cents threshold: within this many cents (pitch-class) counts as "in tune" for the score */
const ALIGNMENT_CENTS_THRESHOLD = 100;
/** EMA alpha for denoising pitch before comparison (lower = smoother, 0.2 = heavy smoothing) */
const ALIGNMENT_SMOOTHING_ALPHA = 0.2;

const PitchGraphDisplay = ({
  targetPitch,
  userPitch,
  isListening,
  isPlaying,
}: PitchGraphDisplayProps) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const pitchHistoryRef = useRef<number[]>([]);
  const targetHistoryRef = useRef<number[]>([]);
  const lastTargetFreqRef = useRef<number | null>(null);
  const inTuneFramesRef = useRef(0);
  const totalFramesRef = useRef(0);
  const smoothedUserFreqRef = useRef<number | null>(null);
  const smoothedTargetFreqRef = useRef<number | null>(null);

  // Keep history of last 200 readings
  const maxHistory = 200;

  useEffect(() => {
    // Reset alignment score and smoothing when not singing along
    if (!isPlaying || !isListening) {
      inTuneFramesRef.current = 0;
      totalFramesRef.current = 0;
      smoothedUserFreqRef.current = null;
      smoothedTargetFreqRef.current = null;
    }

    // Add current pitch to history
    if (isListening && userPitch) {
      pitchHistoryRef.current.push(userPitch.frequency);
      if (pitchHistoryRef.current.length > maxHistory) {
        pitchHistoryRef.current.shift();
      }
    } else if (!isListening) {
      pitchHistoryRef.current = [];
    }

    if (isPlaying) {
      const freqToPush = targetPitch ? targetPitch.frequency : lastTargetFreqRef.current;
      if (freqToPush && isFinite(freqToPush)) {
        targetHistoryRef.current.push(freqToPush);
        lastTargetFreqRef.current = freqToPush;
      } else {
        // push a placeholder (NaN) so spacing remains consistent
        // but avoid pushing non-numeric values into history
        if (targetHistoryRef.current.length > 0) {
          targetHistoryRef.current.push(targetHistoryRef.current[targetHistoryRef.current.length - 1]);
        }
      }

      if (targetHistoryRef.current.length > maxHistory) {
        targetHistoryRef.current.shift();
      }

      // Accumulate pitch alignment for score: octave-invariant, denoised
      if (targetPitch && userPitch && isListening) {
        const alpha = ALIGNMENT_SMOOTHING_ALPHA;
        const u = userPitch.frequency;
        const t = targetPitch.frequency;

        if (smoothedUserFreqRef.current === null) {
          smoothedUserFreqRef.current = u;
        } else {
          smoothedUserFreqRef.current = alpha * smoothedUserFreqRef.current + (1 - alpha) * u;
        }
        if (smoothedTargetFreqRef.current === null) {
          smoothedTargetFreqRef.current = t;
        } else {
          smoothedTargetFreqRef.current = alpha * smoothedTargetFreqRef.current + (1 - alpha) * t;
        }

        const centsDist = pitchClassCents(
          smoothedUserFreqRef.current,
          smoothedTargetFreqRef.current
        );
        totalFramesRef.current += 1;
        if (centsDist <= ALIGNMENT_CENTS_THRESHOLD) {
          inTuneFramesRef.current += 1;
        }
      }
    } else {
      targetHistoryRef.current = [];
      lastTargetFreqRef.current = null;
    }
  }, [userPitch, targetPitch, isListening, isPlaying]);

  const totalFrames = totalFramesRef.current;
  const inTuneFrames = inTuneFramesRef.current;
  const alignmentScore =
    totalFrames >= 10 ? Math.round((inTuneFrames / totalFrames) * 100) : null;

  // Draw canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    const padding = 40;
    const graphWidth = width - padding * 2;
    const graphHeight = height - padding * 2;

    // Clear canvas (transparent) to reveal parent background
    ctx.clearRect(0, 0, width, height);

    // Draw grid and labels
    ctx.strokeStyle = "hsl(var(--muted-foreground) / 0.1)";
    ctx.fillStyle = "hsl(var(--muted-foreground))";
    ctx.font = "12px sans-serif";
    ctx.textAlign = "right";

    // Frequency range
    const minFreq = 70;
    const maxFreq = 500;
    const freqRange = maxFreq - minFreq;

    // Draw Y-axis labels and grid lines
    const labelFreqs = [100, 200, 300, 400, 500];
    for (const freq of labelFreqs) {
      if (freq >= minFreq && freq <= maxFreq) {
        const y = padding + graphHeight - ((freq - minFreq) / freqRange) * graphHeight;
        
        // Grid line
        ctx.setLineDash([2, 2]);
        ctx.strokeStyle = "hsl(var(--muted-foreground) / 0.1)";
        ctx.beginPath();
        ctx.moveTo(padding, y);
        ctx.lineTo(width - padding, y);
        ctx.stroke();
        ctx.setLineDash([]);

        // Label: frequency and note (sharp/flat)
        const noteStr = formatFreqAsNote(freq);
        ctx.fillText(`${freq}Hz ${noteStr}`, padding - 10, y + 4);
      }
    }

    // Draw axes
    ctx.strokeStyle = "hsl(var(--border))";
    ctx.setLineDash([]);
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(padding, padding);
    ctx.lineTo(padding, padding + graphHeight);
    ctx.lineTo(width - padding, padding + graphHeight);
    ctx.stroke();

    // Draw target pitch line (song) — use 7-point MA for stable "ground truth"
    if (targetHistoryRef.current.length > 0) {
      const raw = targetHistoryRef.current;
      const L = raw.length;
      const spacing = graphWidth / Math.max(1, maxHistory - 1);
      const halfWin = 3; // 7-point window

      ctx.strokeStyle = "rgb(59, 130, 246)"; // blue
      ctx.lineWidth = 3;
      ctx.beginPath();

      for (let i = 0; i < L; i++) {
        let sum = 0;
        let count = 0;
        for (let j = i - halfWin; j <= i + halfWin; j++) {
          if (j >= 0 && j < L) {
            const v = raw[j];
            if (isFinite(v)) {
              sum += v;
              count++;
            }
          }
        }
        let freq = count > 0 ? sum / count : raw[i];
        if (!isFinite(freq)) freq = raw[i];
        freq = Math.max(minFreq, Math.min(maxFreq, freq));

        const x = padding + spacing * (maxHistory - L + i);
        const y = padding + graphHeight - ((freq - minFreq) / freqRange) * graphHeight;

        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }

      ctx.stroke();

      // Draw current target point at right edge if available
      if (targetPitch) {
        const freq = Math.max(minFreq, Math.min(maxFreq, targetPitch.frequency));
        const x = padding + graphWidth;
        const y = padding + graphHeight - ((freq - minFreq) / freqRange) * graphHeight;
        ctx.fillStyle = "rgb(59, 130, 246)";
        ctx.beginPath();
        ctx.arc(x, y, 6, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // Draw user pitch line (right-aligned)
    if (pitchHistoryRef.current.length > 0) {
      const rawU = pitchHistoryRef.current;
      const L = rawU.length;
      const spacing = graphWidth / Math.max(1, maxHistory - 1);
      const isInTune = userPitch && targetPitch && Math.abs(userPitch.frequency - targetPitch.frequency) < 20;

      ctx.strokeStyle = isInTune ? "rgb(34, 197, 94)" : "rgb(249, 115, 22)"; // green or orange
      ctx.lineWidth = 3;
      ctx.beginPath();

      for (let i = 0; i < L; i++) {
        const prev = rawU[Math.max(0, i - 1)];
        const curr = rawU[i];
        const next = rawU[Math.min(L - 1, i + 1)];
        let freq = (prev + curr + next) / 3;
        if (!isFinite(freq)) freq = curr;
        freq = Math.max(minFreq, Math.min(maxFreq, freq));

        const x = padding + spacing * (maxHistory - L + i);
        const y = padding + graphHeight - ((freq - minFreq) / freqRange) * graphHeight;

        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }

      ctx.stroke();

      // Draw current user pitch point at right edge
      if (userPitch) {
        const freq = Math.max(minFreq, Math.min(maxFreq, userPitch.frequency));
        const x = padding + graphWidth;
        const y = padding + graphHeight - ((freq - minFreq) / freqRange) * graphHeight;
        ctx.fillStyle = isInTune ? "rgb(34, 197, 94)" : "rgb(249, 115, 22)";
        ctx.beginPath();
        ctx.arc(x, y, 6, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // legend removed — use constant UI boxes below canvas

    // Draw no data message
    if (pitchHistoryRef.current.length === 0 && targetHistoryRef.current.length === 0) {
      ctx.fillStyle = "hsl(var(--muted-foreground))";
      ctx.font = "14px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("Start singing to see the pitch graph", width / 2, height / 2);
    }
  }, [userPitch, targetPitch]);

  // Determine status color and message
  const getStatus = () => {
    if (!isPlaying) return { color: "text-muted-foreground", message: "Load a song and press play" };
    if (!isListening) return { color: "text-yellow-500", message: "Click the microphone to start singing" };
    if (!userPitch) return { color: "text-muted-foreground", message: "Listening... Start singing" };

    if (!targetPitch) return { color: "text-orange-500", message: "Waiting for song pitch..." };

    const diff = Math.abs(userPitch.frequency - targetPitch.frequency);
    if (diff < 15) return { color: "text-green-500", message: "Perfect! You're in tune! 🎵" };
    if (diff < 30) return { color: "text-green-600", message: "Great! Very close!" };
    if (diff < 50) return { color: "text-yellow-500", message: "Getting close! Keep adjusting..." };
    if (diff < 100) return { color: "text-orange-500", message: "Adjust your pitch..." };
    return { color: "text-red-500", message: "Keep adjusting - big difference" };
  };

  const status = getStatus();

  return (
    <div className="w-full space-y-4">
      {/* Status indicator */}
      <div className={cn("text-center text-sm font-semibold transition-colors", status.color)}>
        {status.message}
      </div>

      {/* Pitch alignment score */}
      {alignmentScore !== null && (
        <div className="rounded-lg border border-border bg-card p-4">
          <div className="text-center">
            <div className="text-xs text-muted-foreground mb-1">Pitch alignment</div>
            <div className="flex items-baseline justify-center gap-2">
              <span
                className={cn(
                  "text-4xl font-bold tabular-nums",
                  alignmentScore >= 80
                    ? "text-green-600"
                    : alignmentScore >= 60
                      ? "text-green-500"
                      : alignmentScore >= 40
                        ? "text-yellow-600"
                        : "text-orange-500"
                )}
              >
                {alignmentScore}
              </span>
              <span className="text-xl font-medium text-muted-foreground">%</span>
            </div>
            <p className="text-xs text-muted-foreground mt-1">
              Same note in any octave counts; within {ALIGNMENT_CENTS_THRESHOLD}¢ (smoothed)
            </p>
            <div className="mt-2 h-2 w-full overflow-hidden rounded-full bg-muted">
              <div
                className="h-full rounded-full bg-primary transition-all duration-300"
                style={{ width: `${alignmentScore}%` }}
              />
            </div>
          </div>
        </div>
      )}

      {/* Graph */}
      <div className="rounded-lg border border-border bg-card p-4">
        <canvas
          ref={canvasRef}
          width={600}
          height={300}
          className="w-full border border-border/50 rounded bg-transparent"
        />
      </div>

      {/* Current pitch display - notes as primary */}
      <div className="grid grid-cols-2 gap-4">
        {/* Song box - always visible */}
        <div className="rounded-lg border border-blue-500/20 bg-blue-500/5 p-3 text-center">
          <div className="text-xs text-muted-foreground">Song</div>
          <div className="text-2xl font-bold text-blue-500">
            {targetPitch
              ? targetPitch.noteLabel === targetPitch.noteLabelFlat
                ? targetPitch.noteLabel
                : `${targetPitch.noteLabel} (${targetPitch.noteLabelFlat})`
              : "—"}
          </div>
          <div className="text-xs text-muted-foreground">
            {targetPitch ? targetPitch.frequency.toFixed(1) + " Hz" : "—"}
          </div>
        </div>

        {/* User box - always visible */}
        <div className={cn(
          "rounded-lg border p-3 text-center",
          userPitch
            ? Math.abs(userPitch.frequency - (targetPitch?.frequency || 0)) < 30
              ? "border-green-500/20 bg-green-500/5"
              : "border-orange-500/20 bg-orange-500/5"
            : "border-border/20 bg-muted/5"
        )}>
          <div className="text-xs text-muted-foreground">You</div>
          <div className={cn(
            "text-2xl font-bold",
            userPitch
              ? Math.abs(userPitch.frequency - (targetPitch?.frequency || 0)) < 30
                ? "text-green-500"
                : "text-orange-500"
              : "text-muted-foreground"
          )}>
            {userPitch
              ? userPitch.noteLabel === userPitch.noteLabelFlat
                ? userPitch.noteLabel
                : `${userPitch.noteLabel} (${userPitch.noteLabelFlat})`
              : "—"}
          </div>
          <div className="text-xs text-muted-foreground">
            {userPitch ? userPitch.frequency.toFixed(1) + " Hz" : "—"}
          </div>
        </div>
      </div>

      {/* Difference indicator - in cents (note-based) */}
      <div className="rounded-lg border border-border bg-card p-3 text-center">
        <div className="text-xs text-muted-foreground mb-1">Pitch difference</div>
        <div className="text-lg font-semibold">
          {userPitch && targetPitch
            ? (() => {
                const cents = Math.round(1200 * Math.log2(userPitch.frequency / targetPitch.frequency));
                const absCents = Math.abs(cents);
                return absCents === 0 ? "In tune" : `${absCents}¢ ${cents > 0 ? "sharp" : "flat"}`;
              })()
            : "—"}
        </div>
      </div>
    </div>
  );
};

export default PitchGraphDisplay;

import { useCallback, useEffect, useRef, useState } from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import MicButton from "@/components/MicButton";
import { useAudioAnalyser } from "@/hooks/useAudioAnalyser";
import { usePitchDetection } from "@/hooks/usePitchDetection";
import { frequencyToNote } from "@/lib/pitchDetection";
import { fetchVocalRegisterRealtime } from "@/lib/analysisApi";

export interface VocalRangeSubRange {
  lowNote: string;
  highNote: string;
}

export interface VocalRangeResult {
  /** Overall lowest note (min of chest low and head low) */
  lowNote: string;
  /** Overall highest note (max of chest high and head high) */
  highNote: string;
  /** Same as lowNote — range start (low) */
  start: string;
  /** Same as highNote — range end (high) */
  end: string;
  /** Chest voice range when available */
  chestRange?: VocalRangeSubRange | null;
  /** Head voice (and falsetto) range when available */
  headRange?: VocalRangeSubRange | null;
}

/** Human voice frequency bounds for sanity (roughly C2–C6) */
const MIN_FREQ = 65;
const MAX_FREQ = 1047;
/** Pitch below this (Hz) we treat as chest when API doesn't respond */
const CHEST_HEAD_PITCH_BREAK_HZ = 280;

interface VocalRangeModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onClose?: (result: VocalRangeResult) => void;
}

type RegisterKind = "chest" | "head";

export default function VocalRangeModal({ open, onOpenChange, onClose }: VocalRangeModalProps) {
  const minFreqRef = useRef<number | null>(null);
  const maxFreqRef = useRef<number | null>(null);
  const chestMinRef = useRef<number | null>(null);
  const chestMaxRef = useRef<number | null>(null);
  const headMinRef = useRef<number | null>(null);
  const headMaxRef = useRef<number | null>(null);
  const currentRegisterRef = useRef<RegisterKind | null>(null);
  /** Shown in the bar; from API or pitch-based fallback so bar always updates */
  const [currentRegisterDisplay, setCurrentRegisterDisplay] = useState<RegisterKind | null>(null);
  const [rangeLow, setRangeLow] = useState<string | null>(null);
  const [rangeHigh, setRangeHigh] = useState<string | null>(null);
  const [chestRangeLow, setChestRangeLow] = useState<string | null>(null);
  const [chestRangeHigh, setChestRangeHigh] = useState<string | null>(null);
  const [headRangeLow, setHeadRangeLow] = useState<string | null>(null);
  const [headRangeHigh, setHeadRangeHigh] = useState<string | null>(null);

  const {
    isListening,
    analyserNode,
    error: micError,
    startListening,
    stopListeningAndGetRecordedBlob,
    getRecentFloat32Samples,
  } = useAudioAnalyser();

  const [hasStoppedOnce, setHasStoppedOnce] = useState(false);
  const prevListeningRef = useRef(false);

  useEffect(() => {
    if (prevListeningRef.current && !isListening) setHasStoppedOnce(true);
    prevListeningRef.current = isListening;
  }, [isListening]);

  const handleMicToggle = useCallback(() => {
    if (isListening) {
      stopListeningAndGetRecordedBlob();
      const min = minFreqRef.current;
      const max = maxFreqRef.current;
      if (min != null && max != null) {
        setRangeLow(frequencyToNote(min).noteLabel);
        setRangeHigh(frequencyToNote(max).noteLabel);
      }
      const cMin = chestMinRef.current;
      const cMax = chestMaxRef.current;
      if (cMin != null && cMax != null) {
        setChestRangeLow(frequencyToNote(cMin).noteLabel);
        setChestRangeHigh(frequencyToNote(cMax).noteLabel);
      }
      const hMin = headMinRef.current;
      const hMax = headMaxRef.current;
      if (hMin != null && hMax != null) {
        setHeadRangeLow(frequencyToNote(hMin).noteLabel);
        setHeadRangeHigh(frequencyToNote(hMax).noteLabel);
      }
    } else {
      minFreqRef.current = null;
      maxFreqRef.current = null;
      chestMinRef.current = null;
      chestMaxRef.current = null;
      headMinRef.current = null;
      headMaxRef.current = null;
      currentRegisterRef.current = null;
      setRangeLow(null);
      setRangeHigh(null);
      setChestRangeLow(null);
      setChestRangeHigh(null);
      setHeadRangeLow(null);
      setHeadRangeHigh(null);
      setCurrentRegisterDisplay(null);
      startListening();
    }
  }, [isListening, startListening, stopListeningAndGetRecordedBlob]);

  const { result: userPitch } = usePitchDetection(analyserNode, isListening, false);

  useEffect(() => {
    if (open) {
      minFreqRef.current = null;
      maxFreqRef.current = null;
      chestMinRef.current = null;
      chestMaxRef.current = null;
      headMinRef.current = null;
      headMaxRef.current = null;
      currentRegisterRef.current = null;
      setRangeLow(null);
      setRangeHigh(null);
      setChestRangeLow(null);
      setChestRangeHigh(null);
      setHeadRangeLow(null);
      setHeadRangeHigh(null);
      setCurrentRegisterDisplay(null);
      setHasStoppedOnce(false);
    }
  }, [open]);

  // Poll vocal register every 600ms while listening (for chest vs head attribution)
  useEffect(() => {
    if (!open || !isListening || !getRecentFloat32Samples) return;
    const interval = setInterval(async () => {
      const recent = getRecentFloat32Samples();
      if (!recent || recent.data.length < 4096) return;
      try {
        const out = await fetchVocalRegisterRealtime(recent.data, recent.sampleRate);
        if (out.success && out.prediction) {
          const label = out.prediction.label;
          const reg: RegisterKind = label === "Chest Voice" ? "chest" : "head";
          currentRegisterRef.current = reg;
          setCurrentRegisterDisplay(reg);
        }
      } catch {
        // keep previous; fallback will use pitch below
      }
    }, 600);
    return () => clearInterval(interval);
  }, [open, isListening, getRecentFloat32Samples]);

  useEffect(() => {
    if (!open || !isListening || userPitch == null) return;
    const f = Math.max(MIN_FREQ, Math.min(MAX_FREQ, userPitch.frequency));
    if (minFreqRef.current === null || f < minFreqRef.current) minFreqRef.current = f;
    if (maxFreqRef.current === null || f > maxFreqRef.current) maxFreqRef.current = f;
    // Use API register when available, else split by pitch so we always get two ranges
    const reg: RegisterKind =
      currentRegisterRef.current ?? (f < CHEST_HEAD_PITCH_BREAK_HZ ? "chest" : "head");
    setCurrentRegisterDisplay(reg);
    if (reg === "chest") {
      if (chestMinRef.current === null || f < chestMinRef.current) chestMinRef.current = f;
      if (chestMaxRef.current === null || f > chestMaxRef.current) chestMaxRef.current = f;
    } else {
      if (headMinRef.current === null || f < headMinRef.current) headMinRef.current = f;
      if (headMaxRef.current === null || f > headMaxRef.current) headMaxRef.current = f;
    }
  }, [open, isListening, userPitch]);

  const handleOpenChange = useCallback(
    (next: boolean) => {
      if (!next) {
        const low = rangeLow ?? (minFreqRef.current != null ? frequencyToNote(minFreqRef.current).noteLabel : "—");
        const high = rangeHigh ?? (maxFreqRef.current != null ? frequencyToNote(maxFreqRef.current).noteLabel : "—");
        const chestLow = chestRangeLow ?? (chestMinRef.current != null ? frequencyToNote(chestMinRef.current).noteLabel : null);
        const chestHigh = chestRangeHigh ?? (chestMaxRef.current != null ? frequencyToNote(chestMaxRef.current).noteLabel : null);
        const headLow = headRangeLow ?? (headMinRef.current != null ? frequencyToNote(headMinRef.current).noteLabel : null);
        const headHigh = headRangeHigh ?? (headMaxRef.current != null ? frequencyToNote(headMaxRef.current).noteLabel : null);
        onClose?.({
          lowNote: low,
          highNote: high,
          start: low,
          end: high,
          chestRange: chestLow != null && chestHigh != null ? { lowNote: chestLow, highNote: chestHigh } : null,
          headRange: headLow != null && headHigh != null ? { lowNote: headLow, highNote: headHigh } : null,
        });
      }
      onOpenChange(next);
    },
    [onOpenChange, onClose, rangeLow, rangeHigh, chestRangeLow, chestRangeHigh, headRangeLow, headRangeHigh]
  );

  const displayLow = rangeLow ?? (minFreqRef.current != null ? frequencyToNote(minFreqRef.current).noteLabel : "—");
  const displayHigh = rangeHigh ?? (maxFreqRef.current != null ? frequencyToNote(maxFreqRef.current).noteLabel : "—");
  const displayChest = chestRangeLow != null && chestRangeHigh != null ? `${chestRangeLow} – ${chestRangeHigh}` : (chestMinRef.current != null && chestMaxRef.current != null ? `${frequencyToNote(chestMinRef.current).noteLabel} – ${frequencyToNote(chestMaxRef.current).noteLabel}` : null);
  const displayHead = headRangeLow != null && headRangeHigh != null ? `${headRangeLow} – ${headRangeHigh}` : (headMinRef.current != null && headMaxRef.current != null ? `${frequencyToNote(headMinRef.current).noteLabel} – ${frequencyToNote(headMaxRef.current).noteLabel}` : null);

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>Vocal Range</DialogTitle>
          <DialogDescription>
            Sing in <strong>chest voice</strong> (low to high), then in <strong>head voice</strong> (or falsetto). We&apos;ll measure separate chest and head ranges when the backend is available; otherwise we show one overall range.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-5">
          <div className="flex items-center justify-between rounded-lg border border-border bg-muted/30 p-4">
            <div>
              <p className="text-sm font-medium">Microphone</p>
              <p className="text-xs text-muted-foreground">
                {isListening
                  ? "Sing high, then glide down to low"
                  : hasStoppedOnce
                    ? "Retake to measure again"
                    : "Click to start"}
              </p>
            </div>
            <MicButton
              isListening={isListening}
              onToggle={handleMicToggle}
              idleLabel={hasStoppedOnce ? "Retake" : undefined}
            />
          </div>

          {/* Live bar: Chest vs Head (from API or pitch-based) */}
          {isListening && (
            <div className="rounded-lg border border-border bg-muted/20 p-3">
              <p className="text-xs text-muted-foreground mb-2">Detecting</p>
              <div className="flex rounded-lg overflow-hidden border border-border bg-muted/40">
                <div
                  className={`flex-1 py-2 text-center text-sm font-medium transition-colors ${
                    currentRegisterDisplay === "chest"
                      ? "bg-blue-500/90 text-white"
                      : "bg-muted/60 text-muted-foreground"
                  }`}
                >
                  Chest
                </div>
                <div
                  className={`flex-1 py-2 text-center text-sm font-medium transition-colors ${
                    currentRegisterDisplay === "head"
                      ? "bg-violet-500/90 text-white"
                      : "bg-muted/60 text-muted-foreground"
                  }`}
                >
                  Head
                </div>
              </div>
              {currentRegisterDisplay && (
                <p className="text-xs text-muted-foreground mt-1.5 text-center">
                  Current: {currentRegisterDisplay === "chest" ? "Chest Voice" : "Head Voice"}
                </p>
              )}
            </div>
          )}

          <div className="rounded-lg border border-border bg-card p-5 space-y-3">
            <p className="text-xs text-muted-foreground">Overall Range (Low → High)</p>
            <div className="flex items-center justify-center gap-6 text-center">
              <div>
                <p className="text-xs text-muted-foreground">Low</p>
                <p className="text-2xl font-bold text-foreground">{displayLow}</p>
              </div>
              <span className="text-muted-foreground">→</span>
              <div>
                <p className="text-xs text-muted-foreground">High</p>
                <p className="text-2xl font-bold text-foreground">{displayHigh}</p>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-3 pt-2 border-t border-border">
              <div>
                <p className="text-xs text-muted-foreground">Chest Range</p>
                <p className="text-lg font-semibold text-foreground">{displayChest ?? "—"}</p>
              </div>
              <div>
                <p className="text-xs text-muted-foreground">Head Range</p>
                <p className="text-lg font-semibold text-foreground">{displayHead ?? "—"}</p>
              </div>
            </div>
            {isListening && userPitch && (
              <p className="text-xs text-muted-foreground text-center">
                Current: {userPitch.noteLabel} ({userPitch.frequency.toFixed(0)} Hz)
              </p>
            )}
          </div>

          {micError && <p className="text-sm text-destructive text-center">{micError}</p>}
        </div>
      </DialogContent>
    </Dialog>
  );
}

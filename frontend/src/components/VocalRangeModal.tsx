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

export interface VocalRangeResult {
  /** Lowest note detected (e.g. "A2") */
  lowNote: string;
  /** Highest note detected (e.g. "E4") */
  highNote: string;
  /** Same as lowNote — range start (low) */
  start: string;
  /** Same as highNote — range end (high) */
  end: string;
}

/** Human voice frequency bounds for sanity (roughly C2–C6) */
const MIN_FREQ = 65;
const MAX_FREQ = 1047;

interface VocalRangeModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onClose?: (result: VocalRangeResult) => void;
}

export default function VocalRangeModal({ open, onOpenChange, onClose }: VocalRangeModalProps) {
  const minFreqRef = useRef<number | null>(null);
  const maxFreqRef = useRef<number | null>(null);
  const [rangeLow, setRangeLow] = useState<string | null>(null);
  const [rangeHigh, setRangeHigh] = useState<string | null>(null);

  const {
    isListening,
    analyserNode,
    error: micError,
    startListening,
    stopListeningAndGetRecordedBlob,
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
      // Finalize range from refs
      const min = minFreqRef.current;
      const max = maxFreqRef.current;
      if (min != null && max != null) {
        const low = frequencyToNote(min);
        const high = frequencyToNote(max);
        setRangeLow(low.noteLabel);
        setRangeHigh(high.noteLabel);
      }
    } else {
      minFreqRef.current = null;
      maxFreqRef.current = null;
      setRangeLow(null);
      setRangeHigh(null);
      startListening();
    }
  }, [isListening, startListening, stopListeningAndGetRecordedBlob]);

  const { result: userPitch } = usePitchDetection(analyserNode, isListening, false);

  useEffect(() => {
    if (open) {
      minFreqRef.current = null;
      maxFreqRef.current = null;
      setRangeLow(null);
      setRangeHigh(null);
      setHasStoppedOnce(false);
    }
  }, [open]);

  useEffect(() => {
    if (!open || !isListening || userPitch == null) return;
    const f = Math.max(MIN_FREQ, Math.min(MAX_FREQ, userPitch.frequency));
    if (minFreqRef.current === null || f < minFreqRef.current) minFreqRef.current = f;
    if (maxFreqRef.current === null || f > maxFreqRef.current) maxFreqRef.current = f;
  }, [open, isListening, userPitch]);

  const handleOpenChange = useCallback(
    (next: boolean) => {
      if (!next) {
        const low = rangeLow ?? (minFreqRef.current != null ? frequencyToNote(minFreqRef.current).noteLabel : "—");
        const high = rangeHigh ?? (maxFreqRef.current != null ? frequencyToNote(maxFreqRef.current).noteLabel : "—");
        onClose?.({ lowNote: low, highNote: high, start: low, end: high });
      }
      onOpenChange(next);
    },
    [onOpenChange, onClose, rangeLow, rangeHigh]
  );

  const displayLow = rangeLow ?? (minFreqRef.current != null ? frequencyToNote(minFreqRef.current).noteLabel : "—");
  const displayHigh = rangeHigh ?? (maxFreqRef.current != null ? frequencyToNote(maxFreqRef.current).noteLabel : "—");

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>Vocal range</DialogTitle>
          <DialogDescription>
            Sing your <strong>highest</strong> comfortable note, then glide down to your <strong>lowest</strong>. We&apos;ll measure your range and return it as start (low) and end (high), e.g. start=A2, end=E4.
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

          <div className="rounded-lg border border-border bg-card p-5 space-y-3">
            <p className="text-xs text-muted-foreground">Range (low → high)</p>
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

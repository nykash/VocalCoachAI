import { useState, useRef, useEffect, useCallback } from "react";
import { detectPitch, type PitchResult } from "@/lib/pitchDetection";

export function usePitchDetection(
  analyserNode: AnalyserNode | null,
  isListening: boolean,
  isPaused: boolean,
) {
  const [result, setResult] = useState<PitchResult | null>(null);
  const animationRef = useRef<number>(0);
  const prevNoteRef = useRef<string | null>(null);
  const matchCountRef = useRef(0);

  const update = useCallback(() => {
    if (!analyserNode || !isListening || isPaused) {
      animationRef.current = requestAnimationFrame(update);
      return;
    }

    const buffer = new Float32Array(analyserNode.fftSize);
    analyserNode.getFloatTimeDomainData(buffer);

    const pitch = detectPitch(buffer, analyserNode.context.sampleRate);

    if (pitch === null) {
      // Reset smoothing state and clear display
      prevNoteRef.current = null;
      matchCountRef.current = 0;
      setResult(null);
    } else {
      const noteKey = pitch.noteLabel;
      if (noteKey === prevNoteRef.current) {
        matchCountRef.current++;
      } else {
        prevNoteRef.current = noteKey;
        matchCountRef.current = 1;
      }

      // Only update after 2 consecutive matching frames
      if (matchCountRef.current >= 2) {
        setResult(pitch);
      }
    }

    animationRef.current = requestAnimationFrame(update);
  }, [analyserNode, isListening, isPaused]);

  useEffect(() => {
    animationRef.current = requestAnimationFrame(update);
    return () => cancelAnimationFrame(animationRef.current);
  }, [update]);

  return { result };
}

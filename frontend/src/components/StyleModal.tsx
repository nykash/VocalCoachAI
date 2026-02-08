import { useEffect, useState, useRef, useCallback } from "react";
import { Loader2, Mic, Square } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { fetchVaeTags, isClipTooShortError, type VaeTagResult } from "@/lib/analysisApi";
import { float32ToWavBlob } from "@/lib/audioUtils";
import { detectPitch, type PitchResult } from "@/lib/pitchDetection";
import { buildPitchSummaryFromResults, type PitchHistorySummary } from "@/hooks/usePitchHistory";
import { EXERCISE_INSTRUCTIONS, DEFAULT_INSTRUCTION } from "@/lib/exerciseInstructions";

const PITCH_CHUNK_SIZE = 2048;

interface StyleModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  getRecordedBlob: () => Blob | null;
  /** When set, modal shows exercise-specific title and instructions (e.g. from an exercise card). */
  exerciseName?: string | null;
  /** When set (e.g. from chatbot), modal title is "Instructions" and this is the body content. */
  instructionsContent?: string | null;
  /** Called when the modal is closed; if the user got a result, it is passed with optional pitch summary from the same recording. */
  onCloseWithResult?: (result: VaeTagResult | null, pitchSummary?: PitchHistorySummary | null) => void;
}

export default function StyleModal({
  open,
  onOpenChange,
  getRecordedBlob,
  exerciseName = null,
  instructionsContent = null,
  onCloseWithResult,
}: StyleModalProps) {
  const [result, setResult] = useState<VaeTagResult | null>(null);
  const [pitchSummary, setPitchSummary] = useState<PitchHistorySummary | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [clipTooShort, setClipTooShort] = useState(false);
  const [noExternalBlob, setNoExternalBlob] = useState(false);

  const [isRecording, setIsRecording] = useState(false);
  const [recordError, setRecordError] = useState<string | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const chunksRef = useRef<Float32Array[]>([]);

  const stopRecording = useCallback(() => {
    if (!streamRef.current || !audioContextRef.current) return;
    try {
      if (sourceRef.current && processorRef.current) {
        sourceRef.current.disconnect();
        processorRef.current.disconnect();
      }
    } catch (_) {}
    streamRef.current.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    sourceRef.current = null;
    processorRef.current = null;
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
    setIsRecording(false);
  }, []);

  const startRecording = useCallback(async () => {
    setRecordError(null);
    chunksRef.current = [];
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
      });
      const ctx = new AudioContext();
      const source = ctx.createMediaStreamSource(stream);
      const bufferSize = 4096;
      const processor = ctx.createScriptProcessor(bufferSize, 1, 1);
      processor.onaudioprocess = (e: AudioProcessingEvent) => {
        const input = e.inputBuffer.getChannelData(0);
        chunksRef.current.push(new Float32Array(input));
      };
      source.connect(processor);
      processor.connect(ctx.destination);

      streamRef.current = stream;
      audioContextRef.current = ctx;
      sourceRef.current = source;
      processorRef.current = processor;
      setIsRecording(true);
    } catch (err) {
      const msg =
        err instanceof DOMException && err.name === "NotAllowedError"
          ? "Microphone access denied. Please allow the mic and try again."
          : "Could not access microphone.";
      setRecordError(msg);
    }
  }, []);

  const stopAndAnalyze = useCallback(async () => {
    const ctx = audioContextRef.current;
    const chunks = chunksRef.current;
    stopRecording();
    if (!ctx || !chunks.length) return;
    const totalLength = chunks.reduce((acc, c) => acc + c.length, 0);
    const merged = new Float32Array(totalLength);
    let offset = 0;
    for (const c of chunks) {
      merged.set(c, offset);
      offset += c.length;
    }
    // Compute pitch summary from recording for exercise grading
    const pitchResults: PitchResult[] = [];
    for (let i = 0; i + PITCH_CHUNK_SIZE <= merged.length; i += PITCH_CHUNK_SIZE) {
      const slice = merged.slice(i, i + PITCH_CHUNK_SIZE);
      const p = detectPitch(slice, ctx.sampleRate);
      if (p) pitchResults.push(p);
    }
    const summary =
      pitchResults.length > 0 ? buildPitchSummaryFromResults(pitchResults) : null;
    setPitchSummary(summary);
    const blob = float32ToWavBlob(merged, ctx.sampleRate);
    setLoading(true);
    setError(null);
    setNoExternalBlob(false);
    setResult(null);
    try {
      const r = await fetchVaeTags(blob);
      setResult(r);
      onCloseWithResult?.(r, summary);
      onOpenChange(false);
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Style analysis failed";
      if (isClipTooShortError(msg)) {
        setClipTooShort(true);
        setError(null);
      } else {
        setError(msg);
      }
    } finally {
      setLoading(false);
    }
  }, [stopRecording, onCloseWithResult, onOpenChange]);

  const handleOpenChange = useCallback(
    (nextOpen: boolean) => {
      if (!nextOpen) onCloseWithResult?.(result, pitchSummary);
      onOpenChange(nextOpen);
    },
    [onOpenChange, onCloseWithResult, result, pitchSummary]
  );

  useEffect(() => {
    if (!open) {
      setNoExternalBlob(false);
      setRecordError(null);
      setPitchSummary(null);
      setClipTooShort(false);
      stopRecording();
      return;
    }
    setResult(null);
    setPitchSummary(null);
    setError(null);
    setClipTooShort(false);
    setRecordError(null);
    const blob = getRecordedBlob();
    if (!blob || blob.size === 0) {
      setNoExternalBlob(true);
      setError(null);
      return;
    }
    setNoExternalBlob(false);
    setLoading(true);
    fetchVaeTags(blob)
      .then((r) => {
        setResult(r);
        setClipTooShort(false);
        onCloseWithResult?.(r, null);
        onOpenChange(false);
      })
      .catch((e) => {
        const msg = e instanceof Error ? e.message : "Style analysis failed";
        if (isClipTooShortError(msg)) {
          setClipTooShort(true);
          setError(null);
        } else {
          setError(msg);
        }
      })
      .finally(() => setLoading(false));
  }, [open, getRecordedBlob, stopRecording, onCloseWithResult, onOpenChange]);

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>
            {exerciseName
              ? `Record: ${exerciseName}`
              : instructionsContent
                ? "Instructions"
                : "Singing Style Analysis"}
          </DialogTitle>
          <DialogDescription>
            {exerciseName
              ? "Read the instructions below before recording."
              : instructionsContent
                ? "Read below before recording."
                : "Vocal style analysis from your recorded audio."}
          </DialogDescription>
        </DialogHeader>

        {exerciseName && (
          <div className="rounded-lg border border-border/60 bg-muted/30 p-3 max-h-[200px] overflow-y-auto">
            <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">Instructions</p>
            <p className="text-sm text-foreground whitespace-pre-wrap">
              {EXERCISE_INSTRUCTIONS[exerciseName] ?? DEFAULT_INSTRUCTION}
            </p>
          </div>
        )}

        {!exerciseName && instructionsContent && (
          <div className="rounded-lg border border-border/60 bg-muted/30 p-3 max-h-[200px] overflow-y-auto">
            <p className="text-sm text-foreground whitespace-pre-wrap">
              {instructionsContent}
            </p>
          </div>
        )}

        {loading && (
          <div className="flex items-center gap-3 py-6 justify-center text-muted-foreground">
            <Loader2 className="h-5 w-5 animate-spin" />
            <span className="text-sm">Analyzing your singing style...</span>
          </div>
        )}

        {noExternalBlob && !loading && !result && !clipTooShort && (
          <div className="space-y-3">
            {recordError && (
              <p className="text-sm text-destructive">{recordError}</p>
            )}
            <div className="flex gap-2">
              {!isRecording ? (
                <Button
                  type="button"
                  onClick={startRecording}
                  size="sm"
                  className="gap-2"
                >
                  <Mic className="h-4 w-4" />
                  Start
                </Button>
              ) : (
                <Button
                  type="button"
                  onClick={stopAndAnalyze}
                  size="sm"
                  variant="destructive"
                  className="gap-2"
                >
                  <Square className="h-4 w-4" />
                  Stop
                </Button>
              )}
            </div>
          </div>
        )}

        {clipTooShort && !loading && (
          <div className="flex flex-col items-center gap-3 py-2">
            <p className="text-sm text-muted-foreground">Clip too short</p>
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={() => {
                setClipTooShort(false);
                setNoExternalBlob(true);
                setResult(null);
              }}
            >
              Retake
            </Button>
          </div>
        )}

        {error && !loading && !noExternalBlob && !clipTooShort && (
          <p className="text-sm text-destructive">{error}</p>
        )}
      </DialogContent>
    </Dialog>
  );
}

import { useCallback, useEffect, useRef, useState } from "react";
import { Loader2 } from "lucide-react";
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
import { fetchVaeTags, type VaeTagResult } from "@/lib/analysisApi";
import { cn } from "@/lib/utils";

/** Cents threshold: within this many cents of nearest note counts as "in tune" */
const IN_TUNE_CENTS = 15;
const GAUGE_SMOOTHING_ALPHA = 0.2;
const GAUGE_CENTS_MIN = -50;
const GAUGE_CENTS_MAX = 50;
const GAUGE_CENTS_RANGE = GAUGE_CENTS_MAX - GAUGE_CENTS_MIN;

export interface SingStyleResult {
  minErrorCents: number;
  avgErrorCents: number;
  styleTags: VaeTagResult | null;
}

interface SingStyleModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onClose?: (result: SingStyleResult) => void;
}

export default function SingStyleModal({ open, onOpenChange, onClose }: SingStyleModalProps) {
  const sessionCentsRef = useRef<number[]>([]);
  const [styleTags, setStyleTags] = useState<VaeTagResult | null>(null);
  const [tagsLoading, setTagsLoading] = useState(false);
  const [tagsError, setTagsError] = useState<string | null>(null);

  const {
    isListening,
    isPaused,
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

  const handleMicToggle = useCallback(async () => {
    if (isListening) {
      const blob = await stopListeningAndGetRecordedBlob();
      if (blob && blob.size > 0) {
        setTagsError(null);
        setTagsLoading(true);
        try {
          const result = await fetchVaeTags(blob);
          setStyleTags(result);
        } catch (e) {
          setTagsError(e instanceof Error ? e.message : "Failed to get style tags");
          setStyleTags(null);
        } finally {
          setTagsLoading(false);
        }
      }
    } else {
      sessionCentsRef.current = [];
      setStyleTags(null);
      setTagsError(null);
      startListening();
    }
  }, [isListening, startListening, stopListeningAndGetRecordedBlob]);

  const { result: userPitch } = usePitchDetection(
    analyserNode,
    isListening,
    isPaused
  );

  const smoothedCentsRef = useRef<number | null>(null);
  const [smoothedCents, setSmoothedCents] = useState<number | null>(null);

  useEffect(() => {
    if (!isListening) {
      smoothedCentsRef.current = null;
      setSmoothedCents(null);
      return;
    }
    if (userPitch == null) return;
    const raw = userPitch.centsOff;
    const alpha = GAUGE_SMOOTHING_ALPHA;
    if (smoothedCentsRef.current === null) {
      smoothedCentsRef.current = raw;
    } else {
      smoothedCentsRef.current = alpha * smoothedCentsRef.current + (1 - alpha) * raw;
    }
    setSmoothedCents(smoothedCentsRef.current);
  }, [userPitch, isListening]);

  useEffect(() => {
    if (open) {
      sessionCentsRef.current = [];
      setHasStoppedOnce(false);
      setStyleTags(null);
      setTagsError(null);
    }
  }, [open]);
  useEffect(() => {
    if (open && userPitch != null) {
      sessionCentsRef.current.push(Math.abs(userPitch.centsOff));
    }
  }, [open, userPitch]);

  const gaugeCents = smoothedCents ?? 0;
  const indicatorPercent = Math.max(
    0,
    Math.min(100, ((gaugeCents - GAUGE_CENTS_MIN) / GAUGE_CENTS_RANGE) * 100)
  );
  const absCents = Math.abs(gaugeCents);
  const indicatorColor =
    absCents <= 10 ? "hsl(142, 71%, 42%)"
      : absCents <= 25 ? "hsl(45, 93%, 47%)"
        : "hsl(0, 84%, 55%)";

  const tuneReadout = (() => {
    if (!userPitch) return null;
    const centsOff = userPitch.centsOff;
    const abs = Math.abs(centsOff);
    if (abs <= IN_TUNE_CENTS) return { status: "in_tune" as const, label: "In tune" };
    if (centsOff > 0) return { status: "sharp" as const, label: `${Math.round(abs)}¢ sharp` };
    return { status: "flat" as const, label: `${Math.round(abs)}¢ flat` };
  })();

  const handleOpenChange = useCallback(
    (next: boolean) => {
      if (!next) {
        const arr = sessionCentsRef.current;
        const minErrorCents = arr.length ? Math.min(...arr) : 0;
        const avgErrorCents = arr.length ? arr.reduce((s, x) => s + x, 0) / arr.length : 0;
        onClose?.({ minErrorCents, avgErrorCents, styleTags });
      }
      onOpenChange(next);
    },
    [onOpenChange, onClose, styleTags]
  );

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>Sing style</DialogTitle>
          <DialogDescription>
            Free singing: we&apos;ll check your tuning and analyze your style (no song or lyrics).
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-5">
          <div className="flex items-center justify-between rounded-lg border border-border bg-muted/30 p-4">
            <div>
              <p className="text-sm font-medium">Microphone</p>
              <p className="text-xs text-muted-foreground">
                {isListening
                  ? "Listening… Sing to see feedback"
                  : hasStoppedOnce
                    ? "Retake to redo the clip and get new results"
                    : "Click to start"}
              </p>
            </div>
            <MicButton
              isListening={isListening}
              onToggle={handleMicToggle}
              idleLabel={hasStoppedOnce ? "Retake" : undefined}
            />
          </div>

          <div className="rounded-lg border border-border bg-card p-5 text-center">
            <p className="text-xs text-muted-foreground mb-3">Pitch feedback</p>
            {!isListening && (
              <p className="text-sm text-muted-foreground">Turn on the mic and sing to see if you&apos;re in tune.</p>
            )}
            {isListening && (
              <>
                <div className="mx-auto max-w-full mb-4">
                  <div className="flex justify-between mb-1 text-xs text-muted-foreground">
                    <span>-50¢</span>
                    <span className="font-medium text-foreground">0¢</span>
                    <span>+50¢</span>
                  </div>
                  <div
                    className="relative h-4 rounded-full overflow-hidden"
                    style={{
                      background: "linear-gradient(to right, hsl(0, 70%, 55%) 0%, hsl(45, 90%, 55%) 25%, hsl(142, 60%, 40%) 50%, hsl(45, 90%, 55%) 75%, hsl(25, 85%, 55%) 100%)",
                    }}
                  >
                    <div className="absolute top-0 bottom-0 w-0.5 bg-foreground/30" style={{ left: "50%", transform: "translateX(-50%)" }} />
                    <div
                      className="absolute top-1/2 h-6 w-6 rounded-full border-2 border-white shadow-md transition-all duration-75 ease-out"
                      style={{
                        left: `${indicatorPercent}%`,
                        transform: "translate(-50%, -50%)",
                        backgroundColor: userPitch != null ? indicatorColor : "hsl(0, 0%, 65%)",
                      }}
                    />
                  </div>
                </div>
                {tuneReadout ? (
                  <div className="space-y-1">
                    <p className="text-lg text-muted-foreground">
                      Nearest note: <span className="font-semibold text-foreground">{userPitch.noteLabel}</span>
                    </p>
                    <p
                      className={cn(
                        "text-2xl font-bold",
                        tuneReadout.status === "in_tune" && "text-green-600",
                        tuneReadout.status === "sharp" && "text-orange-500",
                        tuneReadout.status === "flat" && "text-blue-500"
                      )}
                    >
                      {tuneReadout.label}
                    </p>
                    {userPitch && (
                      <p className="text-xs text-muted-foreground">
                        {userPitch.frequency.toFixed(1)} Hz
                        {smoothedCents != null && (
                          <span className="ml-1">({smoothedCents >= 0 ? "+" : ""}{Math.round(smoothedCents)}¢)</span>
                        )}
                      </p>
                    )}
                  </div>
                ) : (
                  <p className="text-sm text-muted-foreground">Listening… sing a clear note to get feedback.</p>
                )}
              </>
            )}
          </div>

          {/* Style tags (after stopping) */}
          {(styleTags || tagsLoading || tagsError) && (
            <div className="rounded-lg border border-border bg-card p-4 space-y-3">
              <h3 className="text-sm font-semibold">Style tags</h3>
              {tagsLoading && (
                <div className="flex items-center gap-2 text-muted-foreground">
                  <Loader2 className="h-5 w-5 animate-spin" />
                  <span className="text-sm">Analyzing your recording…</span>
                </div>
              )}
              {tagsError && <p className="text-sm text-destructive">{tagsError}</p>}
              {styleTags && !tagsLoading && (
                <div className="space-y-3">
                  {styleTags.top_artist && (
                    <div>
                      <p className="text-xs text-muted-foreground">Closest match</p>
                      <p className="font-medium text-foreground">{styleTags.top_artist}</p>
                    </div>
                  )}
                  {styleTags.top_3_attributes.length > 0 && (
                    <div>
                      <p className="text-xs text-muted-foreground mb-1">Top style tags</p>
                      <div className="flex flex-wrap gap-2">
                        {styleTags.top_3_attributes.map(({ tag, confidence }) => (
                          <span
                            key={tag}
                            className="rounded-full bg-primary/15 px-3 py-1 text-sm font-medium text-primary"
                          >
                            {tag} ({(confidence * 100).toFixed(0)}%)
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {micError && <p className="text-sm text-destructive text-center">{micError}</p>}
        </div>
      </DialogContent>
    </Dialog>
  );
}

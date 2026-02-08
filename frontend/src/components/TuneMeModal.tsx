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
import { cn } from "@/lib/utils";

/** Cents threshold: within this many cents of nearest note counts as "in tune" */
const IN_TUNE_CENTS = 15;
/** EMA alpha for smoothing the gauge (lower = smoother, less noisy) */
const GAUGE_SMOOTHING_ALPHA = 0.2;
/** Gauge scale: -50¢ to +50¢, center = 0¢ */
const GAUGE_CENTS_MIN = -50;
const GAUGE_CENTS_MAX = 50;
const GAUGE_CENTS_RANGE = GAUGE_CENTS_MAX - GAUGE_CENTS_MIN;

export interface TuneMeResult {
  minErrorCents: number;
  avgErrorCents: number;
}

interface TuneMeModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** Called when the modal is closed with min and average absolute cents error for the session */
  onClose?: (result: TuneMeResult) => void;
}

export default function TuneMeModal({ open, onOpenChange, onClose }: TuneMeModalProps) {
  const sessionCentsRef = useRef<number[]>([]);
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

  const handleMicToggle = useCallback(() => {
    if (isListening) {
      stopListeningAndGetRecordedBlob();
    } else {
      // Retake: clear session so the new clip is what gets returned on close
      sessionCentsRef.current = [];
      startListening();
    }
  }, [isListening, startListening, stopListeningAndGetRecordedBlob]);

  const { result: userPitch } = usePitchDetection(
    analyserNode,
    isListening,
    isPaused
  );

  // EMA-smoothed cents for the gauge (less noisy)
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

  // Session tracking: clear when modal opens, push |centsOff| while open
  useEffect(() => {
    if (open) {
      sessionCentsRef.current = [];
      setHasStoppedOnce(false);
    }
  }, [open]);
  useEffect(() => {
    if (open && userPitch != null) {
      sessionCentsRef.current.push(Math.abs(userPitch.centsOff));
    }
  }, [open, userPitch]);

  // Gauge position: 0% = -50¢, 50% = 0¢, 100% = +50¢
  const gaugeCents = smoothedCents ?? 0;
  const indicatorPercent = Math.max(
    0,
    Math.min(100, ((gaugeCents - GAUGE_CENTS_MIN) / GAUGE_CENTS_RANGE) * 100)
  );

  // Indicator color: more green when closer to 0
  const absCents = Math.abs(gaugeCents);
  const indicatorColor =
    absCents <= 10
      ? "hsl(142, 71%, 42%)"
      : absCents <= 25
        ? "hsl(45, 93%, 47%)"
        : "hsl(0, 84%, 55%)";

  // In-tune to nearest note: use raw userPitch for label, smoothed for gauge
  const tuneReadout = (() => {
    if (!userPitch) return null;
    const centsOff = userPitch.centsOff;
    const absCents = Math.abs(centsOff);
    if (absCents <= IN_TUNE_CENTS) {
      return { status: "in_tune" as const, cents: 0, centsOff, label: "In tune" };
    }
    if (centsOff > 0) {
      return {
        status: "sharp" as const,
        cents: Math.round(absCents),
        centsOff,
        label: `${Math.round(absCents)}¢ sharp`,
      };
    }
    return {
      status: "flat" as const,
      cents: Math.round(absCents),
      centsOff,
      label: `${Math.round(absCents)}¢ flat`,
    };
  })();

  const handleOpenChange = useCallback(
    (next: boolean) => {
      if (!next) {
        const arr = sessionCentsRef.current;
        const minErrorCents = arr.length ? Math.min(...arr) : 0;
        const avgErrorCents = arr.length ? arr.reduce((s, x) => s + x, 0) / arr.length : 0;
        onClose?.({ minErrorCents, avgErrorCents });
      }
      onOpenChange(next);
    },
    [onOpenChange, onClose]
  );

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>Tune Me</DialogTitle>
          <DialogDescription>
            We&apos;ll tell you if you&apos;re in tune to the nearest note, or how far off you are (flat or sharp).
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-5">
          {/* Microphone */}
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

          {/* Tune readout — gauge + nearest note, in tune / flat / sharp */}
          <div className="rounded-lg border border-border bg-card p-5 text-center">
            <p className="text-xs text-muted-foreground mb-3">Pitch feedback</p>
            {!isListening && (
              <p className="text-sm text-muted-foreground">Turn on the mic and sing to see if you&apos;re in tune.</p>
            )}
            {isListening && (
              <>
                {/* Gauge: green at center (0¢), gradient to red at edges */}
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
                    {/* Center line */}
                    <div
                      className="absolute top-0 bottom-0 w-0.5 bg-foreground/30"
                      style={{ left: "50%", transform: "translateX(-50%)" }}
                    />
                    {/* Indicator dot — position from smoothed cents */}
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
                          <span className="ml-1">
                            ({smoothedCents >= 0 ? "+" : ""}{Math.round(smoothedCents)}¢)
                          </span>
                        )}
                      </p>
                    )}
                  </div>
                ) : (
                  <p className="text-sm text-muted-foreground">
                    Listening… sing a clear note to get feedback.
                  </p>
                )}
              </>
            )}
          </div>

          {micError && (
            <p className="text-sm text-destructive text-center">{micError}</p>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}

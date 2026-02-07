import { usePitchDetection } from "@/hooks/usePitchDetection";

interface NoteDetectorProps {
  analyserNode: AnalyserNode | null;
  isListening: boolean;
  isPaused: boolean;
  sampleRate: number;
}

const NoteDetector = ({ analyserNode, isListening, isPaused }: NoteDetectorProps) => {
  const { result } = usePitchDetection(analyserNode, isListening, isPaused);

  const centsOff = result?.centsOff ?? 0;
  const absCents = Math.abs(centsOff);

  // Color based on tuning accuracy
  const indicatorColor =
    absCents <= 10
      ? "hsl(142, 71%, 45%)"  // green
      : absCents <= 25
        ? "hsl(45, 93%, 47%)"  // yellow
        : "hsl(0, 84%, 60%)"; // red

  // Position: -50 cents = 0%, 0 cents = 50%, +50 cents = 100%
  const indicatorPercent = result ? ((centsOff + 50) / 100) * 100 : 50;

  return (
    <div className="px-6 py-5">
      {/* Title */}
      <h2
        className="mb-4 text-center font-bold text-sm"
        style={{ color: "hsl(0, 0%, 20%)" }}
      >
        Note Detector
      </h2>

      {/* Gauge */}
      <div className="mx-auto max-w-md">
        {/* Cent labels */}
        <div className="flex justify-between mb-1 text-xs" style={{ color: "hsl(0, 0%, 50%)" }}>
          <span>-50¢</span>
          <span>0¢</span>
          <span>+50¢</span>
        </div>

        {/* Track */}
        <div
          className="relative h-3 rounded-full"
          style={{ background: "hsl(226, 60%, 92%)" }}
        >
          {/* Center line */}
          <div
            className="absolute top-0 bottom-0 w-px"
            style={{ left: "50%", background: "hsl(0, 0%, 70%)" }}
          />

          {/* Indicator dot */}
          <div
            className="absolute top-1/2 h-5 w-5 rounded-full shadow-md"
            style={{
              left: `${indicatorPercent}%`,
              transform: "translate(-50%, -50%)",
              backgroundColor: result ? indicatorColor : "hsl(0, 0%, 70%)",
              transition: "left 0.1s ease-out, background-color 0.2s ease",
            }}
          />
        </div>
      </div>

      {/* Note info */}
      <div className="mt-4 flex items-baseline justify-center gap-3">
        {result ? (
          <>
            <span className="text-3xl font-bold" style={{ color: "hsl(0, 0%, 15%)" }}>
              {result.noteLabel}
            </span>
            <span className="text-sm" style={{ color: "hsl(0, 0%, 45%)" }}>
              {result.frequency.toFixed(1)} Hz
            </span>
            <span
              className="text-sm font-medium"
              style={{ color: indicatorColor }}
            >
              {centsOff >= 0 ? "+" : ""}{centsOff}¢
            </span>
          </>
        ) : (
          <>
            <span className="text-3xl font-bold" style={{ color: "hsl(0, 0%, 60%)" }}>
              ---
            </span>
            <span className="text-sm" style={{ color: "hsl(0, 0%, 60%)" }}>
              No pitch detected
            </span>
          </>
        )}
      </div>
    </div>
  );
};

export default NoteDetector;

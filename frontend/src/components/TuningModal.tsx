import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import type { PitchHistorySummary } from "@/hooks/usePitchHistory";

interface TuningModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  summary: PitchHistorySummary;
}

export default function TuningModal({
  open,
  onOpenChange,
  summary,
}: TuningModalProps) {
  const { totalDetections, notes, overallAvgDeviation, currentPitch } = summary;

  // Compute an accuracy score: 100 = perfect, 0 = 50¢ average deviation
  const accuracyScore =
    totalDetections > 0
      ? Math.max(0, Math.round(100 - overallAvgDeviation * 2))
      : null;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Tuning Analysis</DialogTitle>
          <DialogDescription>
            Pitch accuracy from the last 30 seconds
          </DialogDescription>
        </DialogHeader>

        {totalDetections === 0 ? (
          <p className="text-sm text-muted-foreground py-4">
            No pitch data recorded yet. Start singing with your mic on!
          </p>
        ) : (
          <div className="space-y-4">
            {/* Accuracy score */}
            {accuracyScore !== null && (
              <div className="flex items-center gap-4">
                <div
                  className={`text-4xl font-bold ${
                    accuracyScore >= 80
                      ? "text-green-500"
                      : accuracyScore >= 60
                        ? "text-yellow-500"
                        : "text-red-500"
                  }`}
                >
                  {accuracyScore}%
                </div>
                <div className="text-sm text-muted-foreground">
                  <p>Overall Accuracy</p>
                  <p>Avg deviation: {overallAvgDeviation}&#162;</p>
                </div>
              </div>
            )}

            {/* Current pitch */}
            {currentPitch && (
              <div className="rounded-md bg-muted px-3 py-2 text-sm">
                Current note:{" "}
                <span className="font-semibold">{currentPitch.noteLabel}</span>{" "}
                ({currentPitch.frequency.toFixed(1)} Hz,{" "}
                {currentPitch.centsOff >= 0 ? "+" : ""}
                {currentPitch.centsOff}&#162;)
              </div>
            )}

            {/* Note breakdown */}
            <div>
              <h4 className="text-sm font-semibold mb-2">
                Notes Detected ({totalDetections} Total)
              </h4>
              <div className="space-y-1.5">
                {notes.map(({ note, count, avgDeviation }) => {
                  const pct = Math.round((count / totalDetections) * 100);
                  return (
                    <div key={note} className="flex items-center gap-3 text-sm">
                      <span className="w-10 font-mono font-semibold">
                        {note}
                      </span>
                      <div className="flex-1 h-2 rounded-full bg-muted overflow-hidden">
                        <div
                          className="h-full bg-primary rounded-full"
                          style={{ width: `${pct}%` }}
                        />
                      </div>
                      <span className="w-20 text-right text-muted-foreground text-xs">
                        {count}x &middot;{" "}
                        {avgDeviation >= 0 ? "+" : ""}
                        {avgDeviation}&#162;
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}

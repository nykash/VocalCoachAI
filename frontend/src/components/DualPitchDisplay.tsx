import { PitchResult } from "@/lib/pitchDetection";
import { cn } from "@/lib/utils";

interface DualPitchDisplayProps {
  targetPitch: PitchResult | null;
  userPitch: PitchResult | null;
  isListening: boolean;
  isPlaying: boolean;
}

export default function DualPitchDisplay({
  targetPitch,
  userPitch,
  isListening,
  isPlaying,
}: DualPitchDisplayProps) {
  // Calculate if user is in tune (within 50 cents)
  const isInTune =
    userPitch &&
    targetPitch &&
    Math.abs(userPitch.centsOff - targetPitch.centsOff) < 50;

  // Determine visual feedback color
  const getStatusColor = () => {
    if (!isPlaying) return "text-muted-foreground";
    if (!isListening) return "text-yellow-500";
    if (!userPitch) return "text-muted-foreground";
    if (isInTune) return "text-green-500";
    if (Math.abs(userPitch.centsOff - (targetPitch?.centsOff || 0)) < 100) {
      return "text-orange-500";
    }
    return "text-red-500";
  };

  const getStatusText = () => {
    if (!isPlaying) return "Load a song and press play";
    if (!isListening) return "Click the microphone button to start singing";
    if (!userPitch) return "Listening... Start singing";
    if (isInTune) return "Perfect! You're in tune!";
    if (Math.abs(userPitch.centsOff - (targetPitch?.centsOff || 0)) < 100) {
      return "Getting close! Adjust slightly";
    }
    return "Keep adjusting to match the pitch";
  };

  return (
    <div className="w-full space-y-6">
      {/* Status Indicator */}
      <div className="text-center">
        <div className={cn("text-sm font-medium transition-colors", getStatusColor())}>
          {getStatusText()}
        </div>
      </div>

      {/* Pitch Comparison */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Target Pitch (Song) */}
        <div className="rounded-lg border border-border bg-card p-6">
          <div className="text-center">
            <h3 className="text-sm font-semibold text-muted-foreground mb-4">
              Target Pitch (Song)
            </h3>
            {targetPitch ? (
              <div className="space-y-3">
                <div className="text-4xl font-bold text-blue-500">
                  {targetPitch.noteLabel}
                </div>
                <div className="text-sm text-muted-foreground">
                  {targetPitch.frequency.toFixed(1)} Hz
                </div>
                <div className="text-xs text-muted-foreground">
                  Clarity: {(targetPitch.clarity * 100).toFixed(0)}%
                </div>
              </div>
            ) : (
              <div className="h-20 flex items-center justify-center text-muted-foreground text-sm">
                No audio detected
              </div>
            )}
          </div>
        </div>

        {/* User Pitch (Microphone) */}
        <div className="rounded-lg border border-border bg-card p-6">
          <div className="text-center">
            <h3 className="text-sm font-semibold text-muted-foreground mb-4">
              Your Pitch (Microphone)
            </h3>
            {userPitch ? (
              <div className="space-y-3">
                <div
                  className={cn(
                    "text-4xl font-bold transition-colors",
                    isInTune ? "text-green-500" : "text-orange-500"
                  )}
                >
                  {userPitch.noteLabel}
                </div>
                <div className="text-sm text-muted-foreground">
                  {userPitch.frequency.toFixed(1)} Hz
                </div>
                <div className="text-xs">
                  {userPitch.centsOff > 0 ? "+" : ""}
                  {userPitch.centsOff} cents
                </div>
              </div>
            ) : (
              <div className="h-20 flex items-center justify-center text-muted-foreground text-sm">
                Listening...
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Visual Match Indicator */}
      {targetPitch && userPitch && (
        <div className="rounded-lg border border-border bg-card p-4">
          <div className="text-sm font-semibold text-muted-foreground mb-3">
            Pitch Match
          </div>
          <div className="flex items-center gap-4">
            {/* Pitch difference visualization */}
            <div className="flex-1 h-12 relative rounded bg-muted overflow-hidden">
              {/* Reference line (center) */}
              <div className="absolute left-1/2 top-0 bottom-0 w-1 bg-blue-500 opacity-50 transform -translate-x-1/2" />

              {/* User pitch indicator */}
              <div
                className={cn(
                  "absolute top-0 bottom-0 w-1 transform -translate-x-1/2 transition-all",
                  isInTune ? "bg-green-500" : "bg-orange-500"
                )}
                style={{
                  left: `${50 + Math.max(-50, Math.min(50, (userPitch.centsOff - targetPitch.centsOff) / 200)) * 100}%`,
                }}
              />
            </div>

            {/* Difference text */}
            <div className="text-right min-w-fit">
              <div className="text-xs text-muted-foreground">Difference</div>
              <div
                className={cn(
                  "text-sm font-semibold",
                  isInTune ? "text-green-500" : "text-orange-500"
                )}
              >
                {Math.abs(userPitch.centsOff - targetPitch.centsOff).toFixed(0)} cents
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Instructions */}
      {!isPlaying && (
        <div className="rounded-lg border border-border/50 bg-muted/50 p-4 text-sm text-muted-foreground">
          <p className="font-semibold mb-2">How to Use:</p>
          <ol className="list-inside list-decimal space-y-1 text-xs">
            <li>Upload a song audio file</li>
            <li>Click the play button to start the song</li>
            <li>Press the microphone button to allow singing</li>
            <li>Sing along and try to match the target pitch</li>
          </ol>
        </div>
      )}
    </div>
  );
}

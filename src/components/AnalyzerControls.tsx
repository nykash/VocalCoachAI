import { Pause, Play, RotateCcw } from "lucide-react";
import { Button } from "@/components/ui/button";

import { Input } from "@/components/ui/input";

interface AnalyzerControlsProps {
  isListening: boolean;
  isPaused: boolean;
  ampScale: number;
  showMode: "audio" | "freq" | "both";
  onToggleListening: () => void;
  onTogglePause: () => void;
  onAmpScaleChange: (val: number) => void;
  onShowModeChange: (mode: "audio" | "freq" | "both") => void;
}

const AnalyzerControls = ({
  isListening,
  isPaused,
  ampScale,
  showMode,
  onToggleListening,
  onTogglePause,
  onAmpScaleChange,
  onShowModeChange,
}: AnalyzerControlsProps) => {
  return (
    <div className="flex flex-wrap items-center justify-center gap-4 rounded-b-xl border border-t-0 border-border bg-card px-4 py-3">
      {/* Play/Pause */}
      <div className="flex items-center gap-2">
        <Button
          variant="outline"
          size="sm"
          onClick={isListening ? onTogglePause : onToggleListening}
          className="h-8 w-8 p-0"
        >
          {!isListening || isPaused ? (
            <Play className="h-4 w-4" />
          ) : (
            <Pause className="h-4 w-4" />
          )}
        </Button>

        <Button
          variant="outline"
          size="sm"
          onClick={onToggleListening}
          className="h-8 w-8 p-0"
          title={isListening ? "Stop" : "Start"}
        >
          <RotateCcw className="h-4 w-4" />
        </Button>
      </div>

      {/* Show mode radio */}
      <div className="flex items-center gap-3 text-sm">
        <span className="font-semibold text-foreground">Show:</span>
        {(["audio", "freq", "both"] as const).map((mode) => (
          <label key={mode} className="flex items-center gap-1.5 cursor-pointer">
            <input
              type="radio"
              name="showMode"
              checked={showMode === mode}
              onChange={() => onShowModeChange(mode)}
              className="accent-primary"
            />
            <span className="text-muted-foreground">{mode === "freq" ? "freq." : mode}</span>
          </label>
        ))}
      </div>

      {/* Amp Scale */}
      <div className="flex items-center gap-2 text-sm">
        <span className="font-medium text-foreground">Amp Scale:</span>
        <Input
          type="number"
          step="0.05"
          min="0.05"
          max="2"
          value={ampScale}
          onChange={(e) => {
            const v = parseFloat(e.target.value);
            if (!isNaN(v) && v > 0) onAmpScaleChange(v);
          }}
          className="h-7 w-16 text-center text-sm"
        />
      </div>
    </div>
  );
};

export default AnalyzerControls;

import { useState } from "react";
import FrequencySpectrum from "@/components/FrequencySpectrum";
import WaveformDisplay from "@/components/WaveformDisplay";
import NoteDetector from "@/components/NoteDetector";
import AnalyzerControls from "@/components/AnalyzerControls";
import { useAudioAnalyser } from "@/hooks/useAudioAnalyser";

const Index = () => {
  const {
    isListening,
    isPaused,
    analyserNode,
    sampleRate,
    error,
    toggleListening,
    togglePause,
  } = useAudioAnalyser();

  const [ampScale, setAmpScale] = useState(0.02);
  const [showMode, setShowMode] = useState<"audio" | "freq" | "both">("both");

  return (
    <div className="flex min-h-screen flex-col items-center bg-background px-4 py-6">
      {/* Header */}
      <h1 className="mb-4 text-xl font-bold text-destructive tracking-wide">
        Microphone Sound Analyzer
      </h1>

      {/* Main container */}
      <div className="w-full max-w-4xl">
        {/* Panels */}
        <div className="rounded-t-xl border border-border bg-card overflow-hidden">
          {(showMode === "freq" || showMode === "both") && (
            <FrequencySpectrum
              analyserNode={analyserNode}
              isListening={isListening}
              isPaused={isPaused}
              sampleRate={sampleRate}
            />
          )}

          {showMode === "both" && (
            <div className="border-t border-border" />
          )}

          {(showMode === "audio" || showMode === "both") && (
            <WaveformDisplay
              analyserNode={analyserNode}
              isListening={isListening}
              isPaused={isPaused}
              ampScale={ampScale}
            />
          )}

          <div className="border-t border-border" />

          <NoteDetector
            analyserNode={analyserNode}
            isListening={isListening}
            isPaused={isPaused}
            sampleRate={sampleRate}
          />
        </div>

        {/* Controls bar */}
        <AnalyzerControls
          isListening={isListening}
          isPaused={isPaused}
          ampScale={ampScale}
          showMode={showMode}
          onToggleListening={toggleListening}
          onTogglePause={togglePause}
          onAmpScaleChange={setAmpScale}
          onShowModeChange={setShowMode}
        />

        {/* Error */}
        {error && (
          <div className="mt-4 rounded-lg bg-destructive/10 px-4 py-2 text-sm text-destructive text-center">
            {error}
          </div>
        )}

        {/* Status */}
        {!isListening && (
          <p className="mt-4 text-center text-sm text-muted-foreground">
            Press ▶ to start capturing audio from your microphone
          </p>
        )}
      </div>
    </div>
  );
};

export default Index;

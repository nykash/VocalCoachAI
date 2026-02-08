import { useState } from "react";
import { Link } from "react-router-dom";
import FrequencySpectrum from "@/components/FrequencySpectrum";
import WaveformDisplay from "@/components/WaveformDisplay";
import NoteDetector from "@/components/NoteDetector";
import AnalyzerControls from "@/components/AnalyzerControls";
import ChatPanel from "@/components/ChatPanel";
import { useTuneMeModal } from "@/contexts/TuneMeModalContext";
import { useSingStyleModal } from "@/contexts/SingStyleModalContext";
import { useVocalRangeModal } from "@/contexts/VocalRangeModalContext";
import { useAudioAnalyser } from "@/hooks/useAudioAnalyser";
import { usePitchDetection } from "@/hooks/usePitchDetection";
import { usePitchHistory } from "@/hooks/usePitchHistory";
import { Button } from "@/components/ui/button";

const Index = () => {
  const { showTuneMeModal } = useTuneMeModal();
  const { showSingStyleModal } = useSingStyleModal();
  const { showVocalRangeModal } = useVocalRangeModal();
  const {
    isListening,
    isPaused,
    analyserNode,
    sampleRate,
    error,
    toggleListening,
    togglePause,
    getRecordedBlob,
  } = useAudioAnalyser();

  const { result: pitchResult } = usePitchDetection(analyserNode, isListening, isPaused);
  const { formatContext, getHistorySummary } = usePitchHistory(isListening ? pitchResult : null);

  const [ampScale, setAmpScale] = useState(0.02);
  const [showMode, setShowMode] = useState<"audio" | "freq" | "both">("both");

  return (
    <div className="flex min-h-screen flex-col items-center bg-background px-4 py-6">
      {/* Header with Navigation */}
      <div className="w-full max-w-4xl mb-6 flex justify-between items-center">
        <h1 className="text-xl font-bold text-destructive tracking-wide">
          Microphone Sound Analyzer
        </h1>
        <div className="flex gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => {
              showVocalRangeModal().then((result) => {
                console.log("Vocal range result:", result);
                console.log("start=" + result.start + ", end=" + result.end);
              });
            }}
          >
            Vocal range (debug)
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => {
              showSingStyleModal().then((result) => {
                console.log("Sing style result:", result);
                console.log("minErrorCents:", result.minErrorCents);
                console.log("avgErrorCents:", result.avgErrorCents);
                console.log("styleTags:", result.styleTags);
              });
            }}
          >
            Sing style (debug)
          </Button>
          <Button
            variant="default"
            size="sm"
            onClick={() => showTuneMeModal()}
          >
            Tune me
          </Button>
          <Link to="/sing-along">
            <Button variant="outline" size="sm">
              Sing Along
            </Button>
          </Link>
        </div>
      </div>

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

      {/* Chat Panel — floating button + side drawer */}
      <ChatPanel
        pitchContext={formatContext()}
        isListening={isListening}
        getHistorySummary={getHistorySummary}
        getRecordedBlob={getRecordedBlob}
      />
    </div>
  );
};

export default Index;

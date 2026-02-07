import { Music2 } from "lucide-react";
import AudioVisualizer from "@/components/AudioVisualizer";
import MicButton from "@/components/MicButton";
import VolumeIndicator from "@/components/VolumeIndicator";
import { useAudioAnalyser } from "@/hooks/useAudioAnalyser";

const Index = () => {
  const { isListening, analyserNode, error, toggleListening } = useAudioAnalyser();

  return (
    <div className="flex min-h-screen flex-col items-center justify-center bg-background px-4 py-12">
      {/* Header */}
      <div className="mb-10 text-center animate-float">
        <div className="mx-auto mb-4 flex h-14 w-14 items-center justify-center rounded-2xl bg-primary/10">
          <Music2 className="h-7 w-7 text-primary" />
        </div>
        <h1 className="text-4xl font-extrabold tracking-tight text-foreground sm:text-5xl">
          Sing<span className="text-primary">Wave</span>
        </h1>
        <p className="mt-2 text-muted-foreground text-lg">
          Sing into your mic and watch the magic
        </p>
      </div>

      {/* Visualizer */}
      <div className="w-full max-w-2xl space-y-6">
        <AudioVisualizer analyserNode={analyserNode} isListening={isListening} />

        {/* Volume indicator */}
        <VolumeIndicator analyserNode={analyserNode} isListening={isListening} />

        {/* Controls */}
        <div className="flex flex-col items-center gap-4">
          <MicButton isListening={isListening} onToggle={toggleListening} />

          <p className="text-sm text-muted-foreground">
            {isListening
              ? "Listening… Sing your heart out! 🎶"
              : "Tap the mic to start"}
          </p>

          {error && (
            <div className="mt-2 rounded-lg bg-destructive/10 px-4 py-2 text-sm text-destructive">
              {error}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Index;

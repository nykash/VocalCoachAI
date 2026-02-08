import { useState } from "react";
import { Play, Pause, Square } from "lucide-react";
import AudioUploader from "@/components/AudioUploader";
import LyricsUploader from "@/components/LyricsUploader";
import PitchGraphDisplay from "@/components/PitchGraphDisplay";
import MicButton from "@/components/MicButton";
import { Button } from "@/components/ui/button";
import { useAudioAnalyser } from "@/hooks/useAudioAnalyser";
import { useSongAnalyser } from "@/hooks/useSongAnalyser";
import { usePitchDetection } from "@/hooks/usePitchDetection";
import { Slider } from "@/components/ui/slider";
import LyricsPanel from "@/components/LyricsPanel";

const SingAlong = () => {
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [lyricsFile, setLyricsFile] = useState<File | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);

  const {
    isListening,
    isPaused,
    analyserNode,
    error: micError,
    toggleListening,
  } = useAudioAnalyser();

  const { songPitch, currentTime, duration, play, pause, stop, seek } =
    useSongAnalyser(audioFile, isPlaying);

  const { result: userPitch } = usePitchDetection(
    analyserNode,
    isListening,
    isPaused
  );

  const handleFileSelect = (file: File) => {
    setAudioFile(file);
    setIsPlaying(false);
    pause();
  };

  const handleLyricsSelect = (file: File) => {
    setLyricsFile(file);
  };

  const handleClearLyrics = () => {
    setLyricsFile(null);
  };

  const handlePlayToggle = () => {
    if (isPlaying) {
      pause();
      setIsPlaying(false);
    } else {
      play();
      setIsPlaying(true);
    }
  };

  const handleStop = () => {
    stop();
    setIsPlaying(false);
  };

  const formatTime = (seconds: number) => {
    if (!isFinite(seconds)) return "0:00";
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  return (
    <div className="flex min-h-screen flex-col items-center bg-background px-4 py-8">
      {/* Header */}
      <div className="mb-8 text-center">
        <h1 className="mb-2 text-3xl font-bold text-foreground">Sing Along</h1>
        <p className="text-muted-foreground">
          Upload a song and match the pitch in real-time
        </p>
      </div>

      {/* Main container (horizontal layout: controls + lyrics) */}
      <div className="w-full max-w-6xl grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-6">
        {/* File Upload Section */}
        <div className="rounded-lg border border-border bg-card p-6 space-y-4">
          <div>
            <h3 className="text-sm font-semibold mb-3">Audio File</h3>
            <AudioUploader
              onFileSelect={handleFileSelect}
              disabled={isPlaying}
            />
          </div>

          {audioFile && (
            <div>
              <h3 className="text-sm font-semibold mb-3">Lyrics File (.lrc)</h3>
              <LyricsUploader
                onFileSelect={handleLyricsSelect}
                onClear={handleClearLyrics}
                disabled={isPlaying}
                fileName={lyricsFile?.name}
              />
              <p className="text-xs text-muted-foreground mt-2">
                Upload a .lrc file to display synced lyrics while you sing.
              </p>
            </div>
          )}
        </div>

        {/* Song Controls */}
        {audioFile && (
          <div className="rounded-lg border border-border bg-card p-6 space-y-4">
            <div className="space-y-2">
              <div className="flex gap-3">
                <Button
                  onClick={handlePlayToggle}
                  variant="default"
                  size="sm"
                  className="gap-2"
                >
                  {isPlaying ? (
                    <>
                      <Pause className="h-4 w-4" />
                      Pause
                    </>
                  ) : (
                    <>
                      <Play className="h-4 w-4" />
                      Play
                    </>
                  )}
                </Button>
                <Button
                  onClick={handleStop}
                  variant="outline"
                  size="sm"
                  className="gap-2"
                >
                  <Square className="h-4 w-4" />
                  Stop
                </Button>
              </div>

              {/* Progress Bar */}
              <div className="space-y-2">
                <Slider
                  value={[currentTime]}
                  max={duration || 0}
                  step={0.1}
                  onValueChange={(value) => seek(value[0])}
                  disabled={!isPlaying}
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-muted-foreground">
                  <span>{formatTime(currentTime)}</span>
                  <span>{formatTime(duration)}</span>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Microphone Control */}
        {audioFile && (
          <div className="rounded-lg border border-border bg-card p-6 flex items-center justify-between">
            <div>
              <h3 className="font-semibold text-foreground">Microphone</h3>
              <p className="text-sm text-muted-foreground">
                {isListening ? "Recording your voice" : "Ready to capture your voice"}
              </p>
            </div>
            <MicButton
              isListening={isListening}
              onToggle={toggleListening}
            />
          </div>
        )}

        {/* Pitch Graph Display */}
        {audioFile && (
          <PitchGraphDisplay
            targetPitch={songPitch}
            userPitch={isListening ? userPitch : null}
            isListening={isListening}
            isPlaying={isPlaying}
          />
        )}

        </div>

        {/* Right column: Lyrics */}
        <div className="lg:col-span-1">
          {audioFile && (
            <LyricsPanel
              lyricsFile={lyricsFile}
              currentTime={currentTime}
              isPlaying={isPlaying}
              onSeek={(t) => seek(t)}
            />
          )}
        </div>

        {/* Error Message */}
        {micError && (
          <div className="rounded-lg bg-destructive/10 px-4 py-3 text-sm text-destructive border border-destructive/20">
            {micError}
          </div>
        )}

        {/* Initial State */}
        {!audioFile && (
          <div className="rounded-lg border border-dashed border-border bg-muted/50 p-12 text-center">
            <p className="text-muted-foreground">
              Start by uploading a song audio file
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default SingAlong;

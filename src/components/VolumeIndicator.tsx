import { useEffect, useState } from "react";

interface VolumeIndicatorProps {
  analyserNode: AnalyserNode | null;
  isListening: boolean;
}

const VolumeIndicator = ({ analyserNode, isListening }: VolumeIndicatorProps) => {
  const [volume, setVolume] = useState(0);

  useEffect(() => {
    if (!analyserNode || !isListening) {
      setVolume(0);
      return;
    }

    const dataArray = new Uint8Array(analyserNode.frequencyBinCount);
    let animId: number;

    const update = () => {
      analyserNode.getByteFrequencyData(dataArray);
      const avg = dataArray.reduce((sum, val) => sum + val, 0) / dataArray.length;
      setVolume(avg / 255);
      animId = requestAnimationFrame(update);
    };

    update();
    return () => cancelAnimationFrame(animId);
  }, [analyserNode, isListening]);

  const label = volume < 0.1 ? "Quiet" : volume < 0.3 ? "Soft" : volume < 0.6 ? "Moderate" : "Loud";

  return (
    <div className="flex items-center gap-4 glass-card rounded-xl px-5 py-3">
      <span className="text-sm font-medium text-muted-foreground">Volume</span>
      <div className="flex-1 h-2 bg-secondary rounded-full overflow-hidden">
        <div
          className="h-full bg-primary rounded-full transition-all duration-75"
          style={{ width: `${Math.min(volume * 100, 100)}%` }}
        />
      </div>
      <span className="text-sm font-semibold text-foreground min-w-[70px] text-right">
        {isListening ? label : "—"}
      </span>
    </div>
  );
};

export default VolumeIndicator;

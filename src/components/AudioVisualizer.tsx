import { useRef, useEffect, useCallback } from "react";

interface AudioVisualizerProps {
  analyserNode: AnalyserNode | null;
  isListening: boolean;
}

const AudioVisualizer = ({ analyserNode, isListening }: AudioVisualizerProps) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>(0);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !analyserNode) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const width = rect.width;
    const height = rect.height;

    const bufferLength = analyserNode.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    analyserNode.getByteFrequencyData(dataArray);

    ctx.clearRect(0, 0, width, height);

    const barCount = 64;
    const gap = 3;
    const totalGaps = (barCount - 1) * gap;
    const barWidth = (width - totalGaps) / barCount;
    const step = Math.floor(bufferLength / barCount);

    for (let i = 0; i < barCount; i++) {
      const value = dataArray[i * step];
      const normalizedValue = value / 255;
      const barHeight = Math.max(4, normalizedValue * height * 0.85);

      const x = i * (barWidth + gap);
      const y = (height - barHeight) / 2;

      // Gradient from primary blue to lighter blue based on intensity
      const alpha = 0.4 + normalizedValue * 0.6;
      const lightness = 60 - normalizedValue * 15;

      ctx.fillStyle = `hsla(217, 91%, ${lightness}%, ${alpha})`;
      ctx.beginPath();
      ctx.roundRect(x, y, barWidth, barHeight, barWidth / 2);
      ctx.fill();

      // Glow effect for active bars
      if (normalizedValue > 0.5) {
        ctx.shadowColor = `hsla(217, 91%, 60%, ${normalizedValue * 0.5})`;
        ctx.shadowBlur = 15;
        ctx.fill();
        ctx.shadowBlur = 0;
      }
    }

    animationRef.current = requestAnimationFrame(draw);
  }, [analyserNode]);

  const drawIdle = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const width = rect.width;
    const height = rect.height;
    const barCount = 64;
    const gap = 3;
    const totalGaps = (barCount - 1) * gap;
    const barWidth = (width - totalGaps) / barCount;
    const time = Date.now() / 1000;

    ctx.clearRect(0, 0, width, height);

    for (let i = 0; i < barCount; i++) {
      const wave = Math.sin(time * 1.5 + i * 0.15) * 0.5 + 0.5;
      const barHeight = 4 + wave * 20;
      const x = i * (barWidth + gap);
      const y = (height - barHeight) / 2;

      ctx.fillStyle = `hsla(217, 91%, 60%, ${0.15 + wave * 0.15})`;
      ctx.beginPath();
      ctx.roundRect(x, y, barWidth, barHeight, barWidth / 2);
      ctx.fill();
    }

    animationRef.current = requestAnimationFrame(drawIdle);
  }, []);

  useEffect(() => {
    if (isListening && analyserNode) {
      draw();
    } else {
      drawIdle();
    }

    return () => {
      cancelAnimationFrame(animationRef.current);
    };
  }, [isListening, analyserNode, draw, drawIdle]);

  return (
    <div className="w-full glass-card rounded-2xl p-6 glow-primary">
      <canvas
        ref={canvasRef}
        className="w-full h-48 sm:h-64 md:h-72"
        style={{ display: "block" }}
      />
    </div>
  );
};

export default AudioVisualizer;

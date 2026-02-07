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
    const freqData = new Uint8Array(bufferLength);
    analyserNode.getByteFrequencyData(freqData);

    ctx.clearRect(0, 0, width, height);

    // === BAR VISUALIZATION ===
    const barCount = 64;
    const gap = 3;
    const totalGaps = (barCount - 1) * gap;
    const barWidth = (width - totalGaps) / barCount;
    const step = Math.floor(bufferLength / barCount);

    for (let i = 0; i < barCount; i++) {
      const value = freqData[i * step];
      const normalizedValue = value / 255;
      const barHeight = Math.max(4, normalizedValue * height * 0.85);

      const x = i * (barWidth + gap);
      const y = (height - barHeight) / 2;

      const alpha = 0.4 + normalizedValue * 0.6;
      const lightness = 60 - normalizedValue * 15;

      ctx.fillStyle = `hsla(217, 91%, ${lightness}%, ${alpha})`;
      ctx.beginPath();
      ctx.roundRect(x, y, barWidth, barHeight, barWidth / 2);
      ctx.fill();

      if (normalizedValue > 0.5) {
        ctx.shadowColor = `hsla(217, 91%, 60%, ${normalizedValue * 0.5})`;
        ctx.shadowBlur = 15;
        ctx.fill();
        ctx.shadowBlur = 0;
      }
    }

    // === FFT FREQUENCY CURVE (drawn on top) ===
    // Use logarithmic frequency scaling for a more musical representation
    const fftPoints = 128;
    const points: { x: number; y: number }[] = [];

    for (let i = 0; i < fftPoints; i++) {
      // Log scale mapping: more resolution at low frequencies
      const logIndex = Math.pow(i / fftPoints, 2) * (bufferLength - 1);
      const lowerIndex = Math.floor(logIndex);
      const upperIndex = Math.min(lowerIndex + 1, bufferLength - 1);
      const frac = logIndex - lowerIndex;

      // Interpolate between bins
      const value = freqData[lowerIndex] * (1 - frac) + freqData[upperIndex] * frac;
      const normalized = value / 255;

      const x = (i / (fftPoints - 1)) * width;
      const y = height - (normalized * height * 0.85) - (height * 0.075);

      points.push({ x, y });
    }

    // Draw filled area under curve
    ctx.beginPath();
    ctx.moveTo(points[0].x, height);
    ctx.lineTo(points[0].x, points[0].y);

    // Smooth curve using bezier through points
    for (let i = 1; i < points.length - 1; i++) {
      const cpX = (points[i].x + points[i + 1].x) / 2;
      const cpY = (points[i].y + points[i + 1].y) / 2;
      ctx.quadraticCurveTo(points[i].x, points[i].y, cpX, cpY);
    }
    const last = points[points.length - 1];
    ctx.lineTo(last.x, last.y);
    ctx.lineTo(last.x, height);
    ctx.closePath();

    const fillGrad = ctx.createLinearGradient(0, 0, 0, height);
    fillGrad.addColorStop(0, "hsla(217, 91%, 60%, 0.15)");
    fillGrad.addColorStop(1, "hsla(217, 91%, 60%, 0.02)");
    ctx.fillStyle = fillGrad;
    ctx.fill();

    // Draw the FFT line
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);

    for (let i = 1; i < points.length - 1; i++) {
      const cpX = (points[i].x + points[i + 1].x) / 2;
      const cpY = (points[i].y + points[i + 1].y) / 2;
      ctx.quadraticCurveTo(points[i].x, points[i].y, cpX, cpY);
    }
    ctx.lineTo(last.x, last.y);

    ctx.strokeStyle = "hsla(217, 91%, 70%, 0.9)";
    ctx.lineWidth = 2;
    ctx.shadowColor = "hsla(217, 91%, 60%, 0.6)";
    ctx.shadowBlur = 10;
    ctx.stroke();
    ctx.shadowBlur = 0;

    // Draw peak dots on the FFT curve
    for (let i = 2; i < points.length - 2; i++) {
      if (
        points[i].y < points[i - 1].y &&
        points[i].y < points[i + 1].y &&
        points[i].y < height * 0.6
      ) {
        ctx.beginPath();
        ctx.arc(points[i].x, points[i].y, 3, 0, Math.PI * 2);
        ctx.fillStyle = "hsla(217, 91%, 80%, 0.9)";
        ctx.fill();
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

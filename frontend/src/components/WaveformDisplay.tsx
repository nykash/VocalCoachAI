import { useRef, useEffect, useCallback } from "react";

interface WaveformDisplayProps {
  analyserNode: AnalyserNode | null;
  isListening: boolean;
  isPaused: boolean;
  ampScale: number;
}

const PADDING = { top: 30, right: 20, bottom: 45, left: 65 };

const WaveformDisplay = ({ analyserNode, isListening, isPaused, ampScale }: WaveformDisplayProps) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>(0);
  const frozenDataRef = useRef<Float32Array<ArrayBuffer> | null>(null);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const w = rect.width;
    const h = rect.height;
    const plotW = w - PADDING.left - PADDING.right;
    const plotH = h - PADDING.top - PADDING.bottom;

    // Background
    ctx.fillStyle = "hsl(226, 60%, 92%)";
    ctx.fillRect(PADDING.left, PADDING.top, plotW, plotH);

    // Get data
    let timeData: Float32Array<ArrayBuffer>;
    if (analyserNode && isListening && !isPaused) {
      const buf = new Float32Array(analyserNode.fftSize);
      analyserNode.getFloatTimeDomainData(buf);
      timeData = buf as Float32Array<ArrayBuffer>;
      frozenDataRef.current = timeData;
    } else if (frozenDataRef.current) {
      timeData = frozenDataRef.current;
    } else {
      timeData = new Float32Array(2048).fill(0);
    }

    const sampleCount = timeData.length;
    const duration = (sampleCount / (analyserNode?.context.sampleRate || 44100)) * 1000; // ms

    // Helpers
    const timeToX = (t: number) => PADDING.left + (t / duration) * plotW;
    const ampToY = (a: number) => {
      const scaled = a / ampScale;
      return PADDING.top + plotH / 2 - scaled * (plotH / 2);
    };

    // Grid lines & Y-axis labels (amplitude)
    ctx.strokeStyle = "hsl(0, 0%, 80%)";
    ctx.lineWidth = 0.5;
    ctx.fillStyle = "hsl(0, 0%, 30%)";
    ctx.font = "11px 'Plus Jakarta Sans', sans-serif";
    ctx.textAlign = "right";

    const ampSteps = [-ampScale, -ampScale * 0.5, 0, ampScale * 0.5, ampScale];
    for (const amp of ampSteps) {
      const y = ampToY(amp);
      ctx.beginPath();
      ctx.moveTo(PADDING.left, y);
      ctx.lineTo(PADDING.left + plotW, y);
      ctx.stroke();

      const label = amp === 0 ? "0.0" : amp.toFixed(1);
      ctx.fillText(label, PADDING.left - 8, y + 4);
    }

    // X-axis labels (ms)
    ctx.textAlign = "center";
    const timeStep = duration > 30 ? 5 : duration > 15 ? 2 : 1;
    for (let t = 0; t <= duration; t += timeStep) {
      const x = timeToX(t);
      ctx.beginPath();
      ctx.moveTo(x, PADDING.top);
      ctx.lineTo(x, PADDING.top + plotH);
      ctx.stroke();
      ctx.fillText(`${t.toFixed(1)}`, x, PADDING.top + plotH + 16);
    }

    // Axis labels
    ctx.fillStyle = "hsl(0, 0%, 20%)";
    ctx.font = "bold 12px 'Plus Jakarta Sans', sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("t (ms)", PADDING.left + plotW / 2, h - 4);

    ctx.save();
    ctx.translate(14, PADDING.top + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("amplitude", 0, 0);
    ctx.restore();

    // Title
    ctx.fillStyle = "hsl(0, 0%, 20%)";
    ctx.font = "bold 13px 'Plus Jakarta Sans', sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("Microphone Signal", PADDING.left + plotW / 2, 18);

    // Draw waveform
    ctx.beginPath();
    ctx.strokeStyle = "hsl(0, 75%, 50%)";
    ctx.lineWidth = 1.5;

    for (let i = 0; i < sampleCount; i++) {
      const t = (i / sampleCount) * duration;
      const x = timeToX(t);
      const y = ampToY(timeData[i]);

      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.stroke();

    // Plot border
    ctx.strokeStyle = "hsl(0, 0%, 60%)";
    ctx.lineWidth = 1;
    ctx.strokeRect(PADDING.left, PADDING.top, plotW, plotH);

    animationRef.current = requestAnimationFrame(draw);
  }, [analyserNode, isListening, isPaused, ampScale]);

  useEffect(() => {
    draw();
    return () => cancelAnimationFrame(animationRef.current);
  }, [draw]);

  return (
    <canvas
      ref={canvasRef}
      className="w-full"
      style={{ display: "block", height: "260px" }}
    />
  );
};

export default WaveformDisplay;

import { useRef, useEffect, useCallback } from "react";

interface FrequencySpectrumProps {
  analyserNode: AnalyserNode | null;
  isListening: boolean;
  isPaused: boolean;
  sampleRate: number;
}

const PADDING = { top: 30, right: 20, bottom: 45, left: 65 };

const FrequencySpectrum = ({ analyserNode, isListening, isPaused, sampleRate }: FrequencySpectrumProps) => {
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
    let freqData: Float32Array<ArrayBuffer>;
    if (analyserNode && isListening && !isPaused) {
      const buf = new Float32Array(analyserNode.frequencyBinCount);
      analyserNode.getFloatFrequencyData(buf);
      freqData = buf as Float32Array<ArrayBuffer>;
      frozenDataRef.current = freqData;
    } else if (frozenDataRef.current) {
      freqData = frozenDataRef.current;
    } else {
      freqData = new Float32Array(1024).fill(-80);
    }

    const maxFreq = 10000; // 10 kHz
    const binCount = freqData.length;
    const binFreqWidth = sampleRate / (binCount * 2);
    const maxBin = Math.min(Math.floor(maxFreq / binFreqWidth), binCount);

    const dbMin = -90;
    const dbMax = -10;

    // Helper: freq -> x
    const freqToX = (f: number) => PADDING.left + (f / maxFreq) * plotW;
    // Helper: dB -> y
    const dbToY = (db: number) => {
      const clamped = Math.max(dbMin, Math.min(dbMax, db));
      return PADDING.top + plotH - ((clamped - dbMin) / (dbMax - dbMin)) * plotH;
    };

    // Grid lines & Y-axis labels (dB)
    ctx.strokeStyle = "hsl(0, 0%, 80%)";
    ctx.lineWidth = 0.5;
    ctx.fillStyle = "hsl(0, 0%, 30%)";
    ctx.font = "11px 'Plus Jakarta Sans', sans-serif";
    ctx.textAlign = "right";

    for (let db = dbMin; db <= dbMax; db += 10) {
      const y = dbToY(db);
      ctx.beginPath();
      ctx.moveTo(PADDING.left, y);
      ctx.lineTo(PADDING.left + plotW, y);
      ctx.stroke();
      ctx.fillText(`${db}`, PADDING.left - 8, y + 4);
    }

    // X-axis labels (kHz)
    ctx.textAlign = "center";
    for (let f = 0; f <= maxFreq; f += 2000) {
      const x = freqToX(f);
      ctx.beginPath();
      ctx.moveTo(x, PADDING.top);
      ctx.lineTo(x, PADDING.top + plotH);
      ctx.stroke();
      ctx.fillText(`${(f / 1000).toFixed(1)}`, x, PADDING.top + plotH + 16);
    }

    // Axis labels
    ctx.fillStyle = "hsl(0, 0%, 20%)";
    ctx.font = "bold 12px 'Plus Jakarta Sans', sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("f (kHz)", PADDING.left + plotW / 2, h - 4);

    ctx.save();
    ctx.translate(14, PADDING.top + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("audio level (dB)", 0, 0);
    ctx.restore();

    // Title
    ctx.fillStyle = "hsl(0, 0%, 20%)";
    ctx.font = "bold 13px 'Plus Jakarta Sans', sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("Frequency Spectrum", PADDING.left + plotW / 2, 18);

    // Draw spectrum line
    ctx.beginPath();
    ctx.strokeStyle = "hsl(217, 91%, 45%)";
    ctx.lineWidth = 1.5;

    let started = false;
    const peaks: { x: number; y: number; freq: number; db: number }[] = [];

    for (let i = 0; i < maxBin; i++) {
      const freq = i * binFreqWidth;
      const x = freqToX(freq);
      const y = dbToY(freqData[i]);

      if (!started) {
        ctx.moveTo(x, y);
        started = true;
      } else {
        ctx.lineTo(x, y);
      }

      // Detect peaks (local maxima above threshold)
      if (
        i > 2 &&
        i < maxBin - 2 &&
        freqData[i] > freqData[i - 1] &&
        freqData[i] > freqData[i - 2] &&
        freqData[i] > freqData[i + 1] &&
        freqData[i] > freqData[i + 2] &&
        freqData[i] > dbMin + 15
      ) {
        peaks.push({ x, y, freq, db: freqData[i] });
      }
    }
    ctx.stroke();

    // Draw peak labels (top N)
    const topPeaks = peaks
      .sort((a, b) => b.db - a.db)
      .slice(0, 8);

    ctx.fillStyle = "hsl(217, 91%, 30%)";
    ctx.font = "bold 10px 'Plus Jakarta Sans', sans-serif";
    ctx.textAlign = "center";

    for (const peak of topPeaks) {
      ctx.beginPath();
      ctx.arc(peak.x, peak.y, 2.5, 0, Math.PI * 2);
      ctx.fillStyle = "hsl(217, 91%, 45%)";
      ctx.fill();

      ctx.fillStyle = "hsl(217, 91%, 25%)";
      ctx.fillText(`${(peak.freq / 1000).toFixed(2)}`, peak.x, peak.y - 8);
    }

    // Plot border
    ctx.strokeStyle = "hsl(0, 0%, 60%)";
    ctx.lineWidth = 1;
    ctx.strokeRect(PADDING.left, PADDING.top, plotW, plotH);

    animationRef.current = requestAnimationFrame(draw);
  }, [analyserNode, isListening, isPaused, sampleRate]);

  useEffect(() => {
    draw();
    return () => cancelAnimationFrame(animationRef.current);
  }, [draw]);

  return (
    <canvas
      ref={canvasRef}
      className="w-full"
      style={{ display: "block", height: "280px" }}
    />
  );
};

export default FrequencySpectrum;

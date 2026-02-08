import { useState, useRef, useCallback, useEffect } from "react";

/** Build a WAV Blob from mono Float32 samples (range -1..1). */
function float32ToWavBlob(samples: Float32Array, sampleRate: number): Blob {
  const numChannels = 1;
  const bitsPerSample = 16;
  const bytesPerSample = bitsPerSample / 8;
  const blockAlign = numChannels * bytesPerSample;
  const dataSize = samples.length * blockAlign;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);
  let offset = 0;
  const write = (value: number, bytes: number, littleEndian = true) => {
    if (bytes === 2) view.setInt16(offset, value, littleEndian);
    else if (bytes === 4) view.setInt32(offset, value, littleEndian);
    offset += bytes;
  };
  const writeStr = (str: string) => {
    for (let i = 0; i < str.length; i++) view.setUint8(offset++, str.charCodeAt(i));
  };
  writeStr("RIFF");
  write(36 + dataSize, 4);
  writeStr("WAVE");
  writeStr("fmt ");
  write(16, 4); // chunk size
  write(1, 2); // PCM
  write(numChannels, 2);
  write(sampleRate, 4);
  write(sampleRate * blockAlign, 4); // byte rate
  write(blockAlign, 2);
  write(bitsPerSample, 2);
  writeStr("data");
  write(dataSize, 4);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    offset += 2;
  }
  return new Blob([buffer], { type: "audio/wav" });
}

export function useAudioAnalyser() {
  const [isListening, setIsListening] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [analyserNode, setAnalyserNode] = useState<AnalyserNode | null>(null);
  const [sampleRate, setSampleRate] = useState(44100);

  const audioContextRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const recorderChunksRef = useRef<Float32Array[]>([]);
  const scriptProcessorRef = useRef<ScriptProcessorNode | null>(null);

  const startListening = useCallback(async () => {
    try {
      setError(null);
      recorderChunksRef.current = [];

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
        },
      });

      const audioContext = new AudioContext();
      const source = audioContext.createMediaStreamSource(stream);
      const analyser = audioContext.createAnalyser();
      const bufferSize = 4096;
      const scriptProcessor = audioContext.createScriptProcessor(bufferSize, 1, 1);

      analyser.fftSize = 2048;
      analyser.smoothingTimeConstant = 0.7;
      analyser.minDecibels = -90;
      analyser.maxDecibels = -10;

      scriptProcessor.onaudioprocess = (e) => {
        const input = e.inputBuffer.getChannelData(0);
        const output = e.outputBuffer.getChannelData(0);
        output.set(input); // pass through to analyser (otherwise analyser gets silence)
        recorderChunksRef.current.push(new Float32Array(input));
      };
      source.connect(scriptProcessor);
      scriptProcessor.connect(analyser);

      audioContextRef.current = audioContext;
      streamRef.current = stream;
      sourceRef.current = source;
      scriptProcessorRef.current = scriptProcessor;

      setSampleRate(audioContext.sampleRate);
      setAnalyserNode(analyser);
      setIsListening(true);
      setIsPaused(false);
    } catch (err) {
      if (err instanceof DOMException && err.name === "NotAllowedError") {
        setError("Microphone access denied. Please allow microphone access and try again.");
      } else {
        setError("Could not access microphone. Please check your device settings.");
      }
    }
  }, []);

  const stopListening = useCallback(() => {
    if (scriptProcessorRef.current) {
      try {
        if (sourceRef.current) sourceRef.current.disconnect();
        scriptProcessorRef.current.disconnect();
      } catch (_) {}
      scriptProcessorRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (sourceRef.current) {
      sourceRef.current = null;
    }
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }

    setAnalyserNode(null);
    setIsListening(false);
    setIsPaused(false);
  }, []);

  /** Stop the mic and return the recorded audio as a WAV Blob (for sending to analysis API). Resolves with null if nothing was recorded. */
  const stopListeningAndGetRecordedBlob = useCallback((): Promise<Blob | null> => {
    return new Promise((resolve) => {
      const chunks = recorderChunksRef.current;
      if (!chunks.length || !audioContextRef.current) {
        stopListening();
        resolve(null);
        return;
      }
      const sr = audioContextRef.current.sampleRate;
      if (scriptProcessorRef.current && sourceRef.current) {
        try {
          sourceRef.current.disconnect();
          scriptProcessorRef.current.disconnect();
        } catch (_) {}
        scriptProcessorRef.current = null;
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
      }
      sourceRef.current = null;
      if (audioContextRef.current) {
        audioContextRef.current.close();
        audioContextRef.current = null;
      }
      setAnalyserNode(null);
      setIsListening(false);
      setIsPaused(false);

      const totalLength = chunks.reduce((acc, c) => acc + c.length, 0);
      const merged = new Float32Array(totalLength);
      let offset = 0;
      for (const c of chunks) {
        merged.set(c, offset);
        offset += c.length;
      }
      recorderChunksRef.current = [];
      const blob = totalLength > 0 ? float32ToWavBlob(merged, sr) : null;
      resolve(blob);
    });
  }, [stopListening]);

  /** Return a WAV Blob of everything recorded so far WITHOUT stopping the mic. */
  const getRecordedBlob = useCallback((): Blob | null => {
    const chunks = recorderChunksRef.current;
    const ctx = audioContextRef.current;
    if (!chunks.length || !ctx) return null;
    const sr = ctx.sampleRate;
    const totalLength = chunks.reduce((acc, c) => acc + c.length, 0);
    if (totalLength === 0) return null;
    const merged = new Float32Array(totalLength);
    let off = 0;
    for (const c of chunks) {
      merged.set(c, off);
      off += c.length;
    }
    return float32ToWavBlob(merged, sr);
  }, []);

  const toggleListening = useCallback(() => {
    if (isListening) {
      stopListening();
    } else {
      startListening();
    }
  }, [isListening, startListening, stopListening]);

  const togglePause = useCallback(() => {
    setIsPaused((p) => !p);
  }, []);

  useEffect(() => {
    return () => {
      stopListening();
    };
  }, [stopListening]);

  return {
    isListening,
    isPaused,
    analyserNode,
    sampleRate,
    error,
    toggleListening,
    togglePause,
    startListening,
    stopListening,
    stopListeningAndGetRecordedBlob,
    getRecordedBlob,
  };
}

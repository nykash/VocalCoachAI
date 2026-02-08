import { useState, useRef, useEffect, useCallback } from "react";
import { detectPitch, frequencyToNote, type PitchResult } from "@/lib/pitchDetection";

/** EMA alpha for smoothing song pitch (lower = more stable, less responsive). */
const SONG_PITCH_SMOOTHING_ALPHA = 0.2;

/** Audio source: File (upload) or URL string (e.g. backend /songs/filename). */
export type SongAudioSource = File | string | null;

export function useSongAnalyser(audioSource: SongAudioSource, isPlaying: boolean) {
  const [songPitch, setSongPitch] = useState<PitchResult | null>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  
  const audioContextRef = useRef<AudioContext | null>(null);
  const sourceRef = useRef<AudioBufferSourceNode | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const audioElementRef = useRef<HTMLAudioElement | null>(null);
  const blobUrlRef = useRef<string | null>(null);
  const animationRef = useRef<number>(0);
  const prevNoteRef = useRef<string | null>(null);
  const matchCountRef = useRef(0);
  const missCountRef = useRef(0);
  const lastSuccessfulPitchRef = useRef<PitchResult | null>(null);
  const pitchHistoryRef = useRef<PitchResult[]>([]);
  const smoothedFreqRef = useRef<number | null>(null);

  const initializeAudio = useCallback(async () => {
    if (!audioSource) return;

    try {
      // Create or resume audio context
      if (!audioContextRef.current) {
        audioContextRef.current = new AudioContext();
      }
      const audioContext = audioContextRef.current;

      // Create audio element for playback (File → object URL, string → URL as-is)
      const audioElement = new Audio();
      if (typeof audioSource === "string") {
        // Required for Web Audio API when loading from another origin (e.g. backend)
        audioElement.crossOrigin = "anonymous";
        audioElement.src = audioSource;
      } else {
        const url = URL.createObjectURL(audioSource);
        audioElement.src = url;
        blobUrlRef.current = url;
      }
      audioElementRef.current = audioElement;

      // Create media element audio source
      const source = audioContext.createMediaElementSource(audioElement);
      const analyser = audioContext.createAnalyser();

      analyser.fftSize = 2048;
      analyser.smoothingTimeConstant = 0.7;
      analyser.minDecibels = -90;
      analyser.maxDecibels = -10;

      source.connect(analyser);
      analyser.connect(audioContext.destination);
      // Play song through speakers so user can sing along (karaoke)

      analyserRef.current = analyser;
      sourceRef.current = source as any;

      // Set duration
      audioElement.addEventListener("loadedmetadata", () => {
        setDuration(audioElement.duration);
      });

      // Update current time
      audioElement.addEventListener("timeupdate", () => {
        setCurrentTime(audioElement.currentTime);
      });

      // Reset when finished
      audioElement.addEventListener("ended", () => {
        setCurrentTime(0);
        setSongPitch(null);
      });

      return audioElement;
    } catch (err) {
      console.error("Error initializing audio:", err);
    }
  }, [audioSource]);

  const play = useCallback(async () => {
    if (!audioElementRef.current) {
      const audioElement = await initializeAudio();
      if (audioElement) {
        try {
          await audioContextRef.current?.resume();
        } catch (e) {
          /* ignore */
        }
        audioElement.play();
      }
    } else {
      try {
        await audioContextRef.current?.resume();
      } catch (e) {}
      audioElementRef.current.play();
    }
  }, [initializeAudio]);

  const pause = useCallback(() => {
    if (audioElementRef.current) {
      audioElementRef.current.pause();
    }
  }, []);

  const stop = useCallback(() => {
    if (audioElementRef.current) {
      audioElementRef.current.pause();
      audioElementRef.current.currentTime = 0;
      setCurrentTime(0);
      setSongPitch(null);
    }
  }, []);

  const seek = useCallback((time: number) => {
    if (audioElementRef.current) {
      audioElementRef.current.currentTime = time;
      setCurrentTime(time);
    }
  }, []);

  // Pitch detection loop
  useEffect(() => {
    const update = () => {
      // Always run the loop - if playing, detect pitch, otherwise just continue looping
      if (analyserRef.current && isPlaying) {
        const buffer = new Float32Array(analyserRef.current.fftSize);
        analyserRef.current.getFloatTimeDomainData(buffer);

        const pitch = detectPitch(buffer, audioContextRef.current?.sampleRate || 44100);
        if (pitch === null) {
          missCountRef.current++;
          // Hold the last successful pitch indefinitely instead of clearing
          // This keeps the line moving horizontally
          if (lastSuccessfulPitchRef.current) {
            setSongPitch(lastSuccessfulPitchRef.current);
          }
        } else {
          missCountRef.current = 0;
          const noteKey = pitch.noteLabel;
          if (noteKey === prevNoteRef.current) {
            matchCountRef.current++;
          } else {
            prevNoteRef.current = noteKey;
            matchCountRef.current = 1;
            // When note changes, snap smoothed freq toward new note so we don't lag
            if (smoothedFreqRef.current !== null) {
              const semis = 12 * Math.log2(pitch.frequency / smoothedFreqRef.current);
              if (Math.abs(semis) >= 0.5) smoothedFreqRef.current = pitch.frequency;
            }
          }

          // require two consistent frames before accepting
          if (matchCountRef.current >= 2) {
            const alpha = SONG_PITCH_SMOOTHING_ALPHA;
            const rawFreq = pitch.frequency;
            if (smoothedFreqRef.current === null) {
              smoothedFreqRef.current = rawFreq;
            } else {
              smoothedFreqRef.current = alpha * smoothedFreqRef.current + (1 - alpha) * rawFreq;
            }
            const freq = smoothedFreqRef.current;
            const note = frequencyToNote(freq);
            const smoothed: PitchResult = {
              frequency: freq,
              clarity: pitch.clarity,
              ...note,
            };
            setSongPitch(smoothed);
            lastSuccessfulPitchRef.current = smoothed;
            pitchHistoryRef.current.push(smoothed);
            if (pitchHistoryRef.current.length > 300) {
              pitchHistoryRef.current.shift();
            }
          }
        }
      } else if (!isPlaying) {
        // Clear pitch when not playing
        setSongPitch(null);
        lastSuccessfulPitchRef.current = null;
        prevNoteRef.current = null;
        matchCountRef.current = 0;
        missCountRef.current = 0;
        pitchHistoryRef.current = [];
        smoothedFreqRef.current = null;
      }

      animationRef.current = requestAnimationFrame(update);
    };

    animationRef.current = requestAnimationFrame(update);
    return () => cancelAnimationFrame(animationRef.current);
  }, [isPlaying]);

  // On source change: clear refs so next play() re-initializes; cleanup blob URL
  useEffect(() => {
    const prevBlob = blobUrlRef.current;
    const prevEl = audioElementRef.current;
    audioElementRef.current = null;
    blobUrlRef.current = null;
    return () => {
      if (prevEl) prevEl.pause();
      if (prevBlob) URL.revokeObjectURL(prevBlob);
    };
  }, [audioSource]);

  return {
    songPitch,
    currentTime,
    duration,
    play,
    pause,
    stop,
    seek,
  };
}

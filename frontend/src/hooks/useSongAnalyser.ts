import { useState, useRef, useEffect, useCallback } from "react";
import { detectPitch, type PitchResult } from "@/lib/pitchDetection";

export function useSongAnalyser(audioFile: File | null, isPlaying: boolean) {
  const [songPitch, setSongPitch] = useState<PitchResult | null>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  
  const audioContextRef = useRef<AudioContext | null>(null);
  const sourceRef = useRef<AudioBufferSourceNode | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const audioElementRef = useRef<HTMLAudioElement | null>(null);
  const animationRef = useRef<number>(0);
  const prevNoteRef = useRef<string | null>(null);
  const matchCountRef = useRef(0);
  const missCountRef = useRef(0);
  const lastSuccessfulPitchRef = useRef<PitchResult | null>(null);
  const pitchHistoryRef = useRef<PitchResult[]>([]);

  const initializeAudio = useCallback(async () => {
    if (!audioFile) return;

    try {
      // Create or resume audio context
      if (!audioContextRef.current) {
        audioContextRef.current = new AudioContext();
      }
      const audioContext = audioContextRef.current;

      // Create audio element for playback
      const audioElement = new Audio();
      audioElement.src = URL.createObjectURL(audioFile);
      audioElementRef.current = audioElement;

      // Create media element audio source
      const source = audioContext.createMediaElementSource(audioElement);
      const analyser = audioContext.createAnalyser();

      analyser.fftSize = 2048;
      analyser.smoothingTimeConstant = 0.7;
      analyser.minDecibels = -90;
      analyser.maxDecibels = -10;

      source.connect(analyser);
      // DO NOT connect to destination - no audio playback to prevent feedback
      // User will follow the visual guide (lyrics + pitch graph) instead

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
  }, [audioFile]);

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
          }

          // require two consistent frames before accepting
          if (matchCountRef.current >= 2) {
            setSongPitch(pitch);
            lastSuccessfulPitchRef.current = pitch;
            pitchHistoryRef.current.push(pitch);
            // Keep last 300 pitches for smoothing
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
      }

      animationRef.current = requestAnimationFrame(update);
    };

    animationRef.current = requestAnimationFrame(update);
    return () => cancelAnimationFrame(animationRef.current);
  }, [isPlaying]);

  // Cleanup on unmount or file change
  useEffect(() => {
    return () => {
      if (audioElementRef.current) {
        audioElementRef.current.pause();
        URL.revokeObjectURL(audioElementRef.current.src);
      }
    };
  }, [audioFile]);

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

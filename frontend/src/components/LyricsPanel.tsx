import React, { useState, useEffect, useRef } from "react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { parseLrcFile, type LyricLine } from "@/lib/lrcParser";

interface LyricsPanelProps {
  lyricsFile: File | null;
  currentTime: number;
  isPlaying: boolean;
  onSeek: (seconds: number) => void;
}

export default function LyricsPanel({ lyricsFile, currentTime, isPlaying, onSeek }: LyricsPanelProps) {
  const [lyrics, setLyrics] = useState<LyricLine[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [currentLyricId, setCurrentLyricId] = useState<number | null>(null);
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const currentLyricRef = useRef<HTMLDivElement>(null);
  const startsRef = useRef<number[]>([]);

  // Parse lyrics file when it changes
  useEffect(() => {
    if (!lyricsFile) {
      setLyrics([]);
      setError(null);
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const content = e.target?.result as string;
        const parsed = parseLrcFile(content);
        if (parsed.length === 0) {
          setError("No valid lyrics found in file");
          setLyrics([]);
        } else {
            setLyrics(parsed);
            // Precompute starts for fast lookup
            startsRef.current = parsed.map((p) => p.start);
            // default highlight to first line so the UI doesn't wait for a click
            if (parsed.length > 0) setCurrentLyricId(parsed[0].id);
            setError(null);
          }
      } catch (err) {
        setError(`Failed to parse lyrics: ${err instanceof Error ? err.message : "Unknown error"}`);
        setLyrics([]);
      }
    };
    reader.onerror = () => {
      setError("Failed to read file");
      setLyrics([]);
    };
    reader.readAsText(lyricsFile);
  }, [lyricsFile]);

  // Update current lyric when time changes using precomputed starts (binary search)
  useEffect(() => {
    if (lyrics.length === 0) return;

    const starts = startsRef.current;
    // find highest index where start <= currentTime
    let lo = 0;
    let hi = starts.length - 1;
    let found = -1;
    while (lo <= hi) {
      const mid = Math.floor((lo + hi) / 2);
      if (starts[mid] <= currentTime) {
        found = mid;
        lo = mid + 1;
      } else {
        hi = mid - 1;
      }
    }

    if (found >= 0) {
      const id = lyrics[found].id;
      if (id !== currentLyricId) {
        setCurrentLyricId(id);
        setTimeout(() => {
          currentLyricRef.current?.scrollIntoView({ behavior: "smooth", block: "center" });
        }, 0);
      }
    }
  }, [currentTime, lyrics, currentLyricId]);

  // If playback starts, ensure highlight begins from the first lyric
  useEffect(() => {
    if (isPlaying && lyrics.length > 0) {
      setCurrentLyricId(lyrics[0].id);
      setTimeout(() => {
        currentLyricRef.current?.scrollIntoView({ behavior: "smooth", block: "center" });
      }, 0);
    }
  }, [isPlaying, lyrics]);

  if (!lyricsFile) {
    return (
      <div className="rounded-lg border border-border bg-card p-4">
        <h3 className="text-sm font-semibold">Lyrics</h3>
        <p className="text-xs text-muted-foreground mt-1">
          Upload a .lrc lyrics file to display synced lyrics while you sing.
        </p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="rounded-lg border border-border bg-card p-4">
        <h3 className="text-sm font-semibold">Lyrics</h3>
        <p className="text-sm text-destructive mt-2">{error}</p>
      </div>
    );
  }

  if (lyrics.length === 0) {
    return (
      <div className="rounded-lg border border-border bg-card p-4">
        <h3 className="text-sm font-semibold">Lyrics</h3>
        <p className="text-xs text-muted-foreground mt-1">Loading lyrics...</p>
      </div>
    );
  }

  const contextLyrics = lyrics;

  return (
    <div className="rounded-lg border border-border bg-card p-4 space-y-3">
      <div>
        <h3 className="text-sm font-semibold">Lyrics (Singing Guide)</h3>
        <p className="text-xs text-muted-foreground">Blue highlight shows when to sing • Click to seek</p>
      </div>

      <ScrollArea className="h-72 w-full border rounded-lg p-4 bg-muted/30">
        <div className="space-y-2">
          {contextLyrics.map((lyric) => (
            <div
              key={lyric.id}
              ref={currentLyricId === lyric.id ? currentLyricRef : null}
              onClick={() => onSeek(lyric.start)}
              className={`p-3 rounded-lg cursor-pointer transition-all duration-200 ${
                currentLyricId === lyric.id
                  ? "bg-blue-500 text-white font-bold scale-105 shadow-lg"
                  : "hover:bg-accent/50 text-foreground"
              }`}
            >
              <div className="text-xs opacity-70 mb-1">{formatTime(lyric.start)}</div>
              <div className="text-sm font-medium">{lyric.text}</div>
            </div>
          ))}
        </div>
      </ScrollArea>
    </div>
  );
}

function formatTime(s: number) {
  const mins = Math.floor(s / 60);
  const secs = Math.floor(s % 60)
    .toString()
    .padStart(2, "0");
  return `${mins}:${secs}`;
}

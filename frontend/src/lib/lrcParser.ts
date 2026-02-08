export interface LyricLine {
  id: number;
  start: number; // seconds
  end?: number; // seconds (optional, calculated from next line)
  text: string;
}

/**
 * Parse .lrc (lyric) file format into timestamped segments
 * Format: [MM:SS.MS]text content
 * Example: [00:28.84]I'm finding ways to articulate
 */
export function parseLrcFile(content: string): LyricLine[] {
  const lines = content.split("\n");
  const lyrics: LyricLine[] = [];
  let id = 0;

  for (const line of lines) {
    const trimmed = line.trim();
    
    // Skip empty lines and metadata lines (those starting with [id:, [ar:, [al:, [ti:, [au:, [length:)
    if (!trimmed || /^\[(?:id|ar|al|ti|au|length):/i.test(trimmed)) {
      continue;
    }

    // Match timestamp pattern: [MM:SS.MS] or [MM:SS]
    const timestampMatch = trimmed.match(/^\[(\d{1,2}):(\d{2})(?:\.(\d{1,2}))?\](.*)/);
    
    if (timestampMatch) {
      const minutes = parseInt(timestampMatch[1], 10);
      const seconds = parseInt(timestampMatch[2], 10);
      const milliseconds = timestampMatch[3] 
        ? parseInt(timestampMatch[3], 10) * 10 // Convert centiseconds to milliseconds
        : 0;
      
      const startTime = minutes * 60 + seconds + milliseconds / 1000;
      const text = timestampMatch[4].trim();

      // Only add if there's actual text content
      if (text) {
        lyrics.push({
          id: id++,
          start: startTime,
          text,
        });
      }
    }
  }

  // Calculate end times based on next line's start time
  for (let i = 0; i < lyrics.length - 1; i++) {
    lyrics[i].end = lyrics[i + 1].start;
  }

  return lyrics;
}

/**
 * Find which lyric line should be displayed at a given time
 */
export function getLyricAtTime(lyrics: LyricLine[], currentTime: number): LyricLine | null {
  for (const lyric of lyrics) {
    if (currentTime >= lyric.start && (!lyric.end || currentTime < lyric.end)) {
      return lyric;
    }
  }
  return null;
}

/**
 * Get context around current time (current + next few lines)
 */
export function getLyricsContext(lyrics: LyricLine[], currentTime: number, contextLines: number = 5): LyricLine[] {
  const currentIndex = lyrics.findIndex(
    (l) => currentTime >= l.start && (!l.end || currentTime < l.end)
  );

  if (currentIndex === -1) {
    // If not found, show lines starting from first one after currentTime
    const nextIndex = lyrics.findIndex((l) => l.start > currentTime);
    if (nextIndex === -1) return [];
    return lyrics.slice(Math.max(0, nextIndex - 2), nextIndex + contextLines);
  }

  return lyrics.slice(Math.max(0, currentIndex - 1), currentIndex + contextLines);
}

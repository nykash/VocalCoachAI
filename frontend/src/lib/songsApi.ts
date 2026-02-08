const API_BASE = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export async function fetchSongList(): Promise<string[]> {
  const res = await fetch(`${API_BASE}/songs`);
  if (!res.ok) throw new Error("Failed to load song list");
  return res.json() as Promise<string[]>;
}

/** URL to stream a song by filename (for audio element or useSongAnalyser). */
export function songUrl(filename: string): string {
  return `${API_BASE}/songs/${encodeURIComponent(filename)}`;
}

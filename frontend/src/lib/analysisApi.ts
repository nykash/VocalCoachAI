const API_BASE = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export interface VaeTagResult {
  artist_probs: Record<string, number>;
  top_artist: string | null;
  top_3_artists: string[];
  attributes: { tag: string; confidence: number }[];
  top_3_attributes: { tag: string; confidence: number }[];
  n_chunks: number;
}

export async function fetchVaeTags(audioBlob: Blob, temperature = 5): Promise<VaeTagResult> {
  const form = new FormData();
  form.append("audio", audioBlob, "recording.wav");

  const res = await fetch(`${API_BASE}/analyze/vae-tags?temperature=${temperature}`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(res.status === 503 ? "Analysis service not available. Is the backend running and VAE trained?" : text || `Request failed: ${res.status}`);
  }

  return res.json() as Promise<VaeTagResult>;
}

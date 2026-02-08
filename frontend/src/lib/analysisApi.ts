const API_BASE = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

/** True when the API error indicates the recording was too short to analyze. */
export function isClipTooShortError(message: string): boolean {
  return /No valid feature chunks|audio too short|too short/i.test(message);
}

export interface VaeTagResult {
  artist_probs: Record<string, number>;
  top_artist: string | null;
  top_3_artists: string[];
  attributes: { tag: string; confidence: number }[];
  top_3_attributes: { tag: string; confidence: number }[];
  n_chunks: number;
  /** Breathiness 0–100 from spectral flatness (higher = more breathy). */
  breathiness?: number;
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

// Vocal register: defaults to VITE_API_URL. Set VITE_VOCAL_REGISTER_URL=http://localhost:8089 to use Head_Chest_Voice Flask server for chest/head range.
const VOCAL_REGISTER_API_BASE =
  import.meta.env.VITE_VOCAL_REGISTER_URL ?? import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export interface VocalRegisterPrediction {
  label: "Chest Voice" | "Head Voice" | "Falsetto";
  confidence: number;
  chest_probability: number;
  head_probability: number;
  falsetto_probability?: number;
  prediction_code?: number;
}

function float32ArrayToBase64(arr: Float32Array): string {
  const bytes = new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength);
  let binary = "";
  const chunk = 8192;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
  }
  return btoa(binary);
}

/** Normalize backend response to our shape. Flask returns "Head Voice/Falsetto". */
function normalizePrediction(p: {
  label: string;
  confidence?: number;
  chest_probability?: number;
  head_probability?: number;
  falsetto_probability?: number;
  prediction_code?: number;
}): VocalRegisterPrediction {
  const label =
    p.label === "Chest Voice"
      ? "Chest Voice"
      : p.label === "Falsetto"
        ? "Falsetto"
        : "Head Voice"; // "Head Voice" or "Head Voice/Falsetto" -> Head Voice
  return {
    label,
    confidence: p.confidence ?? 0,
    chest_probability: p.chest_probability ?? 0,
    head_probability: p.head_probability ?? 0,
    falsetto_probability: p.falsetto_probability ?? 0,
    prediction_code: p.prediction_code ?? (label === "Chest Voice" ? 0 : 1),
  };
}

/** Send a float32 audio buffer for chest/head. Tries FastAPI then Flask endpoint. */
export async function fetchVocalRegisterRealtime(
  audioFloat32: Float32Array,
  sampleRate: number
): Promise<{ success: true; prediction: VocalRegisterPrediction } | { success: false; error: string }> {
  const base64 = float32ArrayToBase64(audioFloat32);
  const body = JSON.stringify({ audio_base64: base64, sample_rate: sampleRate });

  // Try FastAPI endpoint first (backend/main.py)
  let res = await fetch(`${VOCAL_REGISTER_API_BASE}/vocal-register/realtime`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body,
  });

  // If 404, try Flask endpoint (Head_Chest_Voice backend_server.py on e.g. 8089)
  if (res.status === 404) {
    res = await fetch(`${VOCAL_REGISTER_API_BASE}/api/predict_realtime`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body,
    });
  }

  const data = (await res.json()) as
    | { success: true; prediction: Record<string, unknown> }
    | { success: false; error: string };
  if (!res.ok) {
    return {
      success: false,
      error:
        data && typeof (data as { error?: string }).error === "string"
          ? (data as { error: string }).error
          : `Request failed: ${res.status}`,
    };
  }
  if (!data.success || !(data as { prediction?: unknown }).prediction) {
    return { success: false, error: "Invalid response from server" };
  }
  const pred = (data as { prediction: Record<string, unknown> }).prediction;
  return {
    success: true,
    prediction: normalizePrediction(pred as Parameters<typeof normalizePrediction>[0]),
  };
}

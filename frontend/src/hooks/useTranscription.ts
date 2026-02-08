import { useState } from "react";

export type TranscriptSegment = {
  id: number;
  start: number; // seconds
  end: number; // seconds
  text: string;
};

export function useTranscription() {
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [segments, setSegments] = useState<TranscriptSegment[] | null>(null);
  const [rawText, setRawText] = useState<string | null>(null);

  async function transcribeWithOpenAI(file: File, apiKey: string) {
    setIsTranscribing(true);
    setError(null);
    setSegments(null);
    setRawText(null);

    try {
      const form = new FormData();
      form.append("file", file);
      // request verbose json to get timestamps (may differ between providers)
      form.append("response_format", "verbose_json");
      form.append("model", "whisper-1");

      const res = await fetch("https://api.openai.com/v1/audio/transcriptions", {
        method: "POST",
        headers: {
          Authorization: `Bearer ${apiKey}`,
        },
        body: form,
      });

      if (!res.ok) {
        const text = await res.text();
        throw new Error(`Transcription API error: ${res.status} ${text}`);
      }

      const data = await res.json();
      // For verbose_json, segments are usually at data.segments
      if (data.segments && Array.isArray(data.segments)) {
        const segs: TranscriptSegment[] = data.segments.map((s: any, i: number) => ({
          id: i,
          start: s.start ?? 0,
          end: s.end ?? (s.start ?? 0) + (s.duration ?? 0),
          text: s.text ?? s.transcript ?? "",
        }));
        setSegments(segs);
        setRawText(data.text ?? null);
      } else if (data.text) {
        // fallback: no timestamps, just text
        setRawText(data.text);
        setSegments(null);
      } else {
        setError("Unexpected transcription response format");
      }
    } catch (err: any) {
      setError(err?.message ?? String(err));
    } finally {
      setIsTranscribing(false);
    }
  }

  async function transcribe(file: File, opts?: { provider?: "openai"; apiKey?: string }) {
    setError(null);
    if (!file) return;
    if (!opts || opts.provider === "openai") {
      if (!opts?.apiKey) {
        setError("OpenAI API key required for transcription via OpenAI");
        return;
      }
      await transcribeWithOpenAI(file, opts.apiKey);
    } else {
      setError("No transcription provider configured");
    }
  }

  return {
    isTranscribing,
    error,
    segments,
    rawText,
    transcribe,
  };
}

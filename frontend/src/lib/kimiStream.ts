/**
 * Stream Kimi (Moonshot) chat completions and interpret reasoning_content vs content,
 * adapted from the Python Moonshot streaming example.
 */

export interface KimiMessage {
  role: "user" | "assistant" | "system";
  content: string;
}

export interface KimiStreamOptions {
  apiKey: string;
  model?: string;
  maxTokens?: number;
  temperature?: number;
}

/** Delta from a single SSE chunk (OpenAI-compatible streaming). */
interface StreamDelta {
  content?: string;
  reasoning_content?: string;
}

interface StreamChunk {
  choices?: Array<{ delta?: StreamDelta }>;
}

/**
 * Callbacks for streaming: reasoning is the model's "thinking", content is the final reply.
 */
export interface KimiStreamCallbacks {
  onReasoning?: (text: string) => void;
  onContent?: (text: string) => void;
}

const MOONSHOT_BASE = "https://api.moonshot.ai/v1";
const DEFAULT_MODEL = "kimi-k2-thinking";

/**
 * Stream a chat completion from Kimi (Moonshot). Calls onReasoning for reasoning_content
 * and onContent for content, matching the Python behavior (reasoning first, then content).
 */
export async function streamKimiChat(
  messages: KimiMessage[],
  options: KimiStreamOptions,
  callbacks: KimiStreamCallbacks
): Promise<{ content: string; reasoning: string }> {
  const {
    apiKey,
    model = DEFAULT_MODEL,
    maxTokens = 1024 * 32,
    temperature = 1.0,
  } = options;

  const response = await fetch(`${MOONSHOT_BASE}/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model,
      messages,
      max_tokens: maxTokens,
      stream: true,
      temperature,
    }),
  });

  if (!response.ok) {
    const text = await response.text().catch(() => "");
    throw new Error(`Moonshot API error (${response.status}): ${text}`);
  }

  const reader = response.body?.getReader();
  if (!reader) throw new Error("No response body");

  const decoder = new TextDecoder();
  let buffer = "";
  let content = "";
  let reasoning = "";
  let thinking = false;

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";

      for (const line of lines) {
        if (line.startsWith("data: ")) {
          const data = line.slice(6);
          if (data === "[DONE]") continue;

          try {
            const chunk = JSON.parse(data) as StreamChunk;
            const delta = chunk.choices?.[0]?.delta;
            if (!delta) continue;

            if (delta.reasoning_content) {
              if (!thinking) {
                thinking = true;
                // Optional: callbacks could log "Start Reasoning" here
              }
              reasoning += delta.reasoning_content;
              callbacks.onReasoning?.(delta.reasoning_content);
            }

            if (delta.content) {
              if (thinking) {
                thinking = false;
                // Optional: callbacks could log "End Reasoning" here
              }
              content += delta.content;
              callbacks.onContent?.(delta.content);
            }
          } catch {
            // Skip malformed JSON (e.g. incomplete chunk)
          }
        }
      }
    }
  } finally {
    reader.releaseLock();
  }

  return { content, reasoning };
}

/**
 * Example usage (mirrors the Python script):
 *
 * const apiKey = import.meta.env.VITE_MOONSHOT_API_KEY ?? "";
 * await streamKimiChat(
 *   [
 *     { role: "system", content: "You are Kimi." },
 *     { role: "user", content: "Please explain why 1+1=2." },
 *   ],
 *   { apiKey, maxTokens: 1024 * 32, temperature: 1.0 },
 *   {
 *     onReasoning: (t) => { console.log(t); },  // or show in a "thinking" UI
 *     onContent: (t) => { process.stdout.write(t); },
 *   }
 * );
 */

export interface ChatMessage {
  role: "user" | "assistant" | "system";
  content: string;
}

export type ToolCallName = "show_tuning_modal" | "show_style_modal" | "suggest_exercises";

export interface ToolCallResult {
  name: ToolCallName;
  args?: Record<string, unknown>;
}

export interface ChatResult {
  content: string;
  toolCalls: ToolCallResult[];
}

const TOOLS = [
  {
    type: "function" as const,
    function: {
      name: "show_tuning_modal",
      description:
        "Show tuning analysis with the user's pitch accuracy from the last 30 seconds. Prefer calling this whenever the user asks about tuning, pitch, intonation, sharp/flat, or how well they are hitting notes.",
      parameters: { type: "object", properties: {} },
    },
  },
  {
    type: "function" as const,
    function: {
      name: "show_style_modal",
      description:
        "Show the recording modal for style analysis and feedback. Call this when the user asks about vocal style, vocal twin, who they sound like, or similar artists. Also call this whenever you suggest an exercise in chat: always pick at least one exercise for the user to try, tell them which one to do, and call show_style_modal so the modal appears and they can record and get your feedback.",
      parameters: { type: "object", properties: {} },
    },
  },
  {
    type: "function" as const,
    function: {
      name: "suggest_exercises",
      description:
        "When you suggest one or more exercises in your reply (e.g. 'try humming scales' or 'here are some exercises: Lip trills, Vowel slides'), call this with the exact exercise names so the user can add them to their list with one click. Call it whenever you recommend specific exercises by name.",
      parameters: {
        type: "object",
        properties: {
          exercises: {
            type: "array",
            items: { type: "string" },
            description: "1–10 exercise names you are suggesting in this message",
          },
        },
        required: ["exercises"],
      },
    },
  },
];

// K2-Think API response includes reasoning_content (internal thoughts) and content (final answer)
interface K2ThinkMessage {
  role: "assistant";
  content: string;
  reasoning_content?: string;
  tool_calls?: Array<{
    id: string;
    type: "function";
    function: {
      name: string;
      arguments: string;
    };
  }>;
}

export interface SendChatMessageOptions {
  /** AbortSignal to cancel the request (e.g. for timeout). */
  signal?: AbortSignal;
}

export async function sendChatMessage(
  messages: ChatMessage[],
  pitchContext: string,
  options?: SendChatMessageOptions
): Promise<ChatResult> {
  const apiKey = import.meta.env.VITE_K2_API_KEY;
  if (!apiKey) {
    throw new Error("Missing VITE_K2_API_KEY environment variable");
  }

  const systemParts: string[] = [
    "You are an enthusiastic, supportive vocal coach. Your goal is to help the user get better at singing. Be warm and encouraging. Explain things in plain language and avoid jargon; if you use a term like 'intonation' or 'support', briefly say what it means in everyday words. Celebrate progress and keep advice practical so they can act on it.",
    "Respond in plain text only. Do not use markdown: no **, no #, no ```, no bullet or numbered lists. Use short, simple sentences.",
    "Whenever you suggest an exercise (or exercises) in chat, you must: (1) pick at least one specific exercise for the user to try right away, (2) tell them clearly which one to do, (3) call show_style_modal so the recording modal appears and they can record and get your feedback, and (4) call suggest_exercises with the exact exercise name(s) you recommended so the user can add them to their list. Do not only suggest exercises in text — always make the modal appear so they can try at least one with feedback.",
    "Give concrete vocal exercises and clear, jargon-free explanations. When the user asks for style or vocal twin, call show_style_modal. When they ask about tuning or pitch, call show_tuning_modal. After they do an exercise they can ask to analyze style or pitch and you call the relevant tool.",
    "Tool order: Prefer show_tuning_modal or show_style_modal first when relevant. After they do an exercise, call the suggest_exercises tool to add the exercise to the user's list.",
    pitchContext,
  ].filter(Boolean);

  const systemMessage: ChatMessage = {
    role: "system",
    content: systemParts.join("\n\n"),
  };

  const response = await fetch(
    "https://api.k2think.ai/v1/chat/completions",
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        model: "MBZUAI-IFM/K2-Think-v2",
        messages: [systemMessage, ...messages],
        tools: TOOLS,
        tool_choice: "auto",
        stream: false,
        max_tokens: 1024,
        temperature: 1.0,
      }),
      signal: options?.signal,
    }
  );

  if (!response.ok) {
    const text = await response.text().catch(() => "");
    throw new Error(`K2-Think API error (${response.status}): ${text}`);
  }

  const data = await response.json();
  console.log("[K2-Think full response]:", data);
  const choice = data.choices?.[0]?.message as K2ThinkMessage | undefined;

  // Use content (final answer) only — never reasoning_content (internal thoughts)
  // If model used </think>, take the part after it (final answer); otherwise use full content (e.g. raw JSON from grading).
  const rawContent = choice?.content ?? "";
  const content: string = rawContent.includes("</think>")
    ? (rawContent.split("</think>")[1]?.trim() ?? "")
    : rawContent.trim();

  if (choice?.reasoning_content) {
    console.debug("[K2-Think Reasoning]:", choice.reasoning_content);
  }

  const toolCalls: ToolCallResult[] = [];
  if (choice?.tool_calls && Array.isArray(choice.tool_calls)) {
    for (const tc of choice.tool_calls) {
      const name = tc.function?.name;
      if (!name) continue;
      let args: Record<string, unknown> | undefined;
      try {
        if (typeof tc.function?.arguments === "string") {
          args = JSON.parse(tc.function.arguments) as Record<string, unknown>;
        }
      } catch {
        // ignore
      }
      if (
        name === "show_tuning_modal" ||
        name === "show_style_modal" ||
        name === "suggest_exercises"
      ) {
        toolCalls.push({ name, args });
      }
    }
  }

  return { content, toolCalls };
}

/** True if the model's content looks like a function-call instruction instead of a real reply. */
export function isFunctionCallContent(content: string): boolean {
  const s = content.trim();
  return (
    s.includes("FN_CALL") ||
    /show_style_modal\s*\(\s*\)/.test(s) ||
    /show_tuning_modal\s*\(\s*\)/.test(s) ||
    s === "show_style_modal()" ||
    s === "show_tuning_modal()"
  );
}

export const DEFAULT_EXERCISES = ["Humming scales", "Scales on ah", "Scales on zzz"] as const;

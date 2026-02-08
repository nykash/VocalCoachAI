export interface ChatMessage {
  role: "user" | "assistant" | "system";
  content: string;
}

export type ToolCall = "show_tuning_modal" | "show_style_modal";

export interface ChatResult {
  content: string;
  toolCalls: ToolCall[];
}

const TOOLS = [
  {
    type: "function" as const,
    function: {
      name: "show_tuning_modal",
      description:
        "Show a tuning analysis modal with the user's pitch accuracy data from the last 30 seconds. Call this when the user asks about their tuning, pitch accuracy, intonation, whether they are sharp or flat, or any question about how well they are hitting the notes.",
      parameters: { type: "object", properties: {} },
    },
  },
  {
    type: "function" as const,
    function: {
      name: "show_style_modal",
      description:
        "Show a singing style analysis modal that analyzes the user's recorded audio to identify their vocal style, closest artist match, and style attributes. Call this when the user asks about their singing style, vocal tone, what artist they sound like, their vocal timbre, or any question about the character/quality of their voice.",
      parameters: { type: "object", properties: {} },
    },
  },
];

export async function sendChatMessage(
  messages: ChatMessage[],
  pitchContext: string
): Promise<ChatResult> {
  const apiKey = import.meta.env.VITE_K2_API_KEY;
  if (!apiKey) {
    throw new Error("Missing VITE_K2_API_KEY environment variable");
  }

  const systemMessage: ChatMessage = {
    role: "system",
    content: pitchContext,
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
        stream: false,
      }),
    }
  );

  if (!response.ok) {
    const text = await response.text().catch(() => "");
    throw new Error(`K2-Think API error (${response.status}): ${text}`);
  }

  const data = await response.json();
  const choice = data.choices?.[0]?.message;
  const content: string = choice?.content ?? "";

  const toolCalls: ToolCall[] = [];
  if (choice?.tool_calls && Array.isArray(choice.tool_calls)) {
    for (const tc of choice.tool_calls) {
      const name = tc.function?.name;
      if (name === "show_tuning_modal" || name === "show_style_modal") {
        toolCalls.push(name);
      }
    }
  }

  return { content, toolCalls };
}

export interface ChatMessage {
  role: "user" | "assistant" | "system";
  content: string;
}

export async function sendChatMessage(
  messages: ChatMessage[],
  pitchContext: string
): Promise<string> {
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
        stream: false,
      }),
    }
  );

  if (!response.ok) {
    const text = await response.text().catch(() => "");
    throw new Error(`K2-Think API error (${response.status}): ${text}`);
  }

  const data = await response.json();
  return data.choices?.[0]?.message?.content ?? "";
}

import { useState, useCallback } from "react";
import { sendChatMessage, type ChatMessage } from "@/lib/chatApi";

export interface DisplayMessage {
  role: "user" | "assistant";
  content: string;
}

export function useChat() {
  const [messages, setMessages] = useState<DisplayMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const sendMessage = useCallback(
    async (text: string, pitchContext: string) => {
      const userMsg: DisplayMessage = { role: "user", content: text };
      setMessages((prev) => [...prev, userMsg]);
      setIsLoading(true);
      setError(null);

      try {
        // Build conversation for the API (excluding system messages)
        const apiMessages: ChatMessage[] = [
          ...messages.map((m) => ({ role: m.role, content: m.content })),
          { role: "user" as const, content: text },
        ];

        const reply = await sendChatMessage(apiMessages, pitchContext);
        const assistantMsg: DisplayMessage = {
          role: "assistant",
          content: reply,
        };
        setMessages((prev) => [...prev, assistantMsg]);
      } catch (err) {
        const msg =
          err instanceof Error ? err.message : "Failed to get response";
        setError(msg);
      } finally {
        setIsLoading(false);
      }
    },
    [messages]
  );

  return { messages, sendMessage, isLoading, error };
}

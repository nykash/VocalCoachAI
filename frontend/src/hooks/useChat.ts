import { useState, useCallback } from "react";
import {
  sendChatMessage,
  type ChatMessage,
  type ToolCall,
} from "@/lib/chatApi";

export interface DisplayMessage {
  role: "user" | "assistant";
  content: string;
}

export function useChat() {
  const [messages, setMessages] = useState<DisplayMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pendingToolCalls, setPendingToolCalls] = useState<ToolCall[]>([]);

  const sendMessage = useCallback(
    async (text: string, pitchContext: string) => {
      const userMsg: DisplayMessage = { role: "user", content: text };
      setMessages((prev) => [...prev, userMsg]);
      setIsLoading(true);
      setError(null);
      setPendingToolCalls([]);

      try {
        const apiMessages: ChatMessage[] = [
          ...messages.map((m) => ({ role: m.role, content: m.content })),
          { role: "user" as const, content: text },
        ];

        const result = await sendChatMessage(apiMessages, pitchContext);

        if (result.content) {
          const assistantMsg: DisplayMessage = {
            role: "assistant",
            content: result.content,
          };
          setMessages((prev) => [...prev, assistantMsg]);
        }

        if (result.toolCalls.length > 0) {
          setPendingToolCalls(result.toolCalls);
          // If the LLM only called tools with no text, add a brief message
          if (!result.content) {
            setMessages((prev) => [
              ...prev,
              {
                role: "assistant",
                content: result.toolCalls.includes("show_tuning_modal")
                  ? "Here's your tuning analysis!"
                  : "Let me analyze your singing style!",
              },
            ]);
          }
        }
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

  const clearToolCalls = useCallback(() => {
    setPendingToolCalls([]);
  }, []);

  return { messages, sendMessage, isLoading, error, pendingToolCalls, clearToolCalls };
}

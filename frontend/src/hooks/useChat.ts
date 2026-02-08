import { useState, useCallback, useRef, useEffect } from "react";
import {
  sendChatMessage,
  isFunctionCallContent,
  type ChatMessage,
  type ToolCallResult,
} from "@/lib/chatApi";
import { fetchVaeTags, type VaeTagResult } from "@/lib/analysisApi";
import { getExerciseInstructionsForPrompt } from "@/lib/exerciseInstructions";
import type { PitchHistorySummary } from "@/hooks/usePitchHistory";

export interface DisplayMessage {
  role: "user" | "assistant";
  content: string;
}

export interface UseChatOptions {
  getRecordedBlob?: () => Blob | null;
  onSetExercises?: (exercises: string[]) => void;
}

export function formatStyleResultForLLM(result: VaeTagResult): string {
  const parts: string[] = [];
  if (result.top_artist) parts.push(`Closest artist match: ${result.top_artist}`);
  if (result.top_3_artists?.length)
    parts.push(`Similar artists: ${result.top_3_artists.join(", ")}`);
  if (result.top_3_attributes?.length) {
    const attrs = result.top_3_attributes
      .map((a) => `${a.tag} (${(a.confidence * 100).toFixed(0)}%)`)
      .join(", ");
    parts.push(`Style attributes: ${attrs}`);
  }
  return parts.length ? parts.join(". ") : "No style data.";
}

export function formatPitchSummaryForLLM(summary: PitchHistorySummary | null): string {
  if (!summary || summary.totalDetections === 0) return "No pitch data.";
  const parts: string[] = [
    `Total detections: ${summary.totalDetections}`,
    `Overall avg deviation: ${summary.overallAvgDeviation} cents`,
  ];
  if (summary.notes.length > 0) {
    const noteStr = summary.notes
      .slice(0, 10)
      .map((n) => `${n.note} (${n.count} hits, avg ${n.avgDeviation}¢)`)
      .join("; ");
    parts.push(`Notes: ${noteStr}`);
  }
  if (summary.currentPitch) {
    const sign = summary.currentPitch.centsOff >= 0 ? "+" : "";
    parts.push(`Current: ${summary.currentPitch.noteLabel} ${sign}${summary.currentPitch.centsOff}¢`);
  }
  return parts.join(". ");
}

export interface ExerciseGradeResult {
  grade: number;
  feedback: string;
  similar_exercises: string[];
}

/** Remove markdown formatting so we display plain text only. */
function stripMarkdown(content: string): string {
  if (!content?.trim()) return content;
  return content
    .replace(/\*\*([^*]+)\*\*/g, "$1")
    .replace(/\*([^*]+)\*/g, "$1")
    .replace(/__([^_]+)__/g, "$1")
    .replace(/_([^_]+)_/g, "$1")
    .replace(/^#+\s+/gm, "")
    .replace(/```[\s\S]*?```/g, "")
    .replace(/`([^`]+)`/g, "$1")
    .replace(/^\s*[-*]\s+/gm, "• ")
    .trim();
}

/** Strip <think>...</think> blocks (model reasoning). If </think> appears anywhere, keep only the text *before* the first </think> (the main reply); drop anything after (e.g. trailing "vocal twin" line). */
function stripThinkTags(content: string): string {
  if (!content?.trim()) return content;
  // If </think> appears, treat it as delimiter: show only what's before it (main message)
  if (/<\/think>/i.test(content)) {
    const before = content.split(/<\/think>/i)[0]?.trim() ?? "";
    if (before.length > 0) {
      return before.replace(/<think>[\s\S]*?$/gi, "").trim();
    }
  }
  // Otherwise just remove <think>...</think> blocks
  return content
    .replace(/<think>[\s\S]*?<\/think>/gi, "")
    .replace(/<think>[\s\S]*/gi, "")
    .trim();
}

/** Apply suggest_exercises from any API result so Add button appears from any turn. */
function applySuggestExercises(
  result: { toolCalls?: Array<{ name: string; args?: Record<string, unknown> }> },
  setSuggestedExercises: (list: string[]) => void
) {
  if (!result.toolCalls) return;
  for (const tc of result.toolCalls) {
    if (tc.name === "suggest_exercises" && Array.isArray(tc.args?.exercises)) {
      const list = tc.args.exercises.filter((e): e is string => typeof e === "string").slice(0, 10);
      if (list.length > 0) setSuggestedExercises(list);
      break;
    }
  }
}

/** True if the reply is a placeholder we must not show (model echoed the prompt). */
function isPlaceholderReply(text: string): boolean {
  const t = text.trim().toLowerCase();
  return (
    t === "your 1-2 sentence reply here" ||
    t === "your full reply inside the quotes for final_message" ||
    /^your 1-2 sentence reply here\.?$/i.test(t) ||
    t.length < 15 && /reply here|sentence reply/i.test(t)
  );
}

/** Extract only the final_message from model output; avoids showing thought chain. */
function extractFinalMessage(content: string): string {
  if (!content?.trim()) return content;
  const noThink = stripThinkTags(content);
  const trimmed = noThink.trim();
  try {
    const parsed = JSON.parse(trimmed);
    if (typeof parsed?.final_message === "string") return parsed.final_message.trim();
  } catch {
    // Not valid JSON; try to find "final_message": "..." in the text
    const match = trimmed.match(/"final_message"\s*:\s*"((?:[^"\\]|\\.)*)"/);
    if (match?.[1]) return match[1].replace(/\\"/g, '"').trim();
  }
  return trimmed;
}

export function useChat(options: UseChatOptions = {}) {
  const { getRecordedBlob, onSetExercises } = options;
  const [messages, setMessages] = useState<DisplayMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pendingToolCalls, setPendingToolCalls] = useState<ToolCallResult[]>([]);
  const [suggestedExercises, setSuggestedExercises] = useState<string[]>([]);
  const messagesRef = useRef<DisplayMessage[]>([]);
  useEffect(() => {
    messagesRef.current = messages;
  }, [messages]);

  const sendMessage = useCallback(
    async (text: string, pitchContext: string) => {
      const userMsg: DisplayMessage = { role: "user", content: text };
      setMessages((prev) => [...prev, userMsg]);
      setIsLoading(true);
      setError(null);
      setPendingToolCalls([]);
      setSuggestedExercises([]);

      try {
        const apiMessages: ChatMessage[] = [
          ...messages.map((m) => ({ role: m.role, content: m.content })),
          { role: "user" as const, content: text },
        ];

        const result = await sendChatMessage(apiMessages, pitchContext);
        applySuggestExercises(result, setSuggestedExercises);

        const contentIsFunctionCall = result.content
          ? isFunctionCallContent(result.content)
          : false;
        const styleRequestInContent =
          result.content?.includes("show_style_modal") && contentIsFunctionCall;
        const hasStyleTool =
          result.toolCalls.some((t) => t.name === "show_style_modal") || styleRequestInContent;
        const modalToolCalls = result.toolCalls.filter(
          (t) => t.name === "show_tuning_modal" || t.name === "show_style_modal"
        );
        const effectiveToolCalls =
          hasStyleTool && !result.toolCalls.some((t) => t.name === "show_style_modal")
            ? [...modalToolCalls, { name: "show_style_modal" as const }]
            : modalToolCalls;

        // If the model returned show_style_modal, run the tool and get the LLM to answer with the real data
        if (hasStyleTool && getRecordedBlob) {
          const blob = getRecordedBlob();
          if (blob && blob.size > 0) {
            try {
              const styleResult = await fetchVaeTags(blob);
              const toolResultText = formatStyleResultForLLM(styleResult);
              const followUpMessages: ChatMessage[] = [
                ...apiMessages,
                {
                  role: "system" as const,
                  content: `Style analysis result: ${toolResultText}\n\n${getExerciseInstructionsForPrompt()}\n\nUsing the conversation above, reply to what the user actually asked. Respond with a JSON object containing only final_message as a key and your reply as the value. Answer the question the user previously asked that we could not directly answer without the style analysis. DON'T just blindly repeat the results, make sure you answer the question the user asked but cite numbers and details that are relevant to the question! If they asked how to do something though, you should always give them feedback after listening to the recording and then give potentially more instructions on how to do it. Then suggest adding an excersize if applicable as a tool call! Make sure to end the message by asking if the user would like to add the excersize to their list or for you to suggest another exercise. Always start with responding to the audio the user sent and then give them feedback and suggestions based on that afterwards.`,
                },
                {
                  role: "user" as const,
                  content: `The user asked: "${text}". You called show_style_modal and got the style analysis result. Answer their question directly with helpful vocal advice (2-4 sentences). Do not say "vocal twin" or "turn on your mic" as the main reply unless they asked who they sound like. If they asked for technique (e.g. less breathy, brighter), give concrete tips. Plain text only, no JSON. Consider suggesting an excersize if applicable to add to the user's list.`,
                },
              ];
              console.log('input messages', [
                ...apiMessages,
                {
                  role: "system" as const,
                  content: `Style analysis result: ${toolResultText}\n\n${getExerciseInstructionsForPrompt()}\n\nUsing the conversation above, reply to what the user actually asked. Respond with a JSON object containing only final_message as a key and your reply as the value. Answer the question the user previously asked that we could not directly answer without the style analysis. DON'T just blindly repeat the results, make sure you answer the questiont the user asked! Then suggest adding an excersize if applicable as a tool call!`,
                },
                {
                  role: "user" as const,
                  content: `The user asked: "${text}". You called show_style_modal and got the style analysis result. Answer their question directly with helpful vocal advice (2-4 sentences). Do not say "vocal twin" or "turn on your mic" as the main reply unless they asked who they sound like. If they asked for technique (e.g. less breathy, brighter), give concrete tips. Plain text only, no JSON. Consider suggesting an excersize if applicable to add to the user's list.`,
                },
              ]);
              console.log("followUpMessages", followUpMessages);
              const followUp = await sendChatMessage(followUpMessages, pitchContext);
              applySuggestExercises(followUp, setSuggestedExercises);
              const raw =
                followUp.content && !isFunctionCallContent(followUp.content)
                  ? followUp.content
                  : toolResultText;
              const reply = extractFinalMessage(raw);
              let display = stripMarkdown(reply || raw);
              if (isPlaceholderReply(display)) display = toolResultText;
              setMessages((prev) => [
                ...prev,
                { role: "assistant", content: display },
              ]);
            } catch (styleErr) {
              const errMsg =
                styleErr instanceof Error ? styleErr.message : "Style analysis failed";
              setMessages((prev) => [
                ...prev,
                {
                  role: "assistant",
                  content: `I couldn't analyze your recording right now (${errMsg}). Try again in a moment!`,
                },
              ]);
            }
          } else {
            const noRecordFollowUp: ChatMessage[] = [
              ...apiMessages,
              {
                role: "user" as const,
                content: `The user asked: "${text}". You called show_style_modal but there is no recording. Answer their question directly with helpful vocal advice (2-4 sentences). Do not say "vocal twin" or "turn on your mic" as the main reply unless they asked who they sound like. If they asked for technique (e.g. less breathy, brighter), give concrete tips. Plain text only, no JSON.`,
              },
            ];
            try {
              const noRecordReply = await sendChatMessage(noRecordFollowUp, pitchContext);
              applySuggestExercises(noRecordReply, setSuggestedExercises);
              const rawReply = noRecordReply.content?.trim() ?? "";
              const clean = stripMarkdown(extractFinalMessage(rawReply) || rawReply);
              const fallback =
                "I'd love to find your vocal twin — turn on your mic, sing a bit, then ask again!";
              setMessages((prev) => [
                ...prev,
                {
                  role: "assistant",
                  content: clean && !isPlaceholderReply(clean) ? clean : fallback,
                },
              ]);
            } catch {
              setMessages((prev) => [
                ...prev,
                {
                  role: "assistant",
                  content:
                    "I'd love to find your vocal twin — turn on your mic, sing a bit, then ask again!",
                },
              ]);
            }
          }
          setPendingToolCalls(effectiveToolCalls);
          setIsLoading(false);
          return;
        }

        if (hasStyleTool && !getRecordedBlob) {
          const noRecordFollowUp: ChatMessage[] = [
            ...apiMessages,
            {
              role: "user" as const,
              content: `The user asked: "${text}". You called show_style_modal but there is no recording. Answer their question directly with helpful vocal advice (2-4 sentences). Do not say "vocal twin" or "turn on your mic" as the main reply unless they asked who they sound like. If they asked for technique (e.g. less breathy, brighter), give concrete tips. Plain text only, no JSON.`,
            },
          ];
            try {
            const noRecordReply = await sendChatMessage(noRecordFollowUp, pitchContext);
            applySuggestExercises(noRecordReply, setSuggestedExercises);
            const rawReply = noRecordReply.content?.trim() ?? "";
            const clean = stripMarkdown(extractFinalMessage(rawReply) || rawReply);
            const fallback =
              "I'd love to find your vocal twin — turn on your mic, sing a bit, then ask again!";
            setMessages((prev) => [
              ...prev,
              {
                role: "assistant",
                content: clean && !isPlaceholderReply(clean) ? clean : fallback,
              },
            ]);
          } catch {
            setMessages((prev) => [
              ...prev,
              {
                role: "assistant",
                content:
                  "I'd love to find your vocal twin — turn on your mic, sing a bit, then ask again!",
              },
            ]);
          }
          setPendingToolCalls(effectiveToolCalls);
          setIsLoading(false);
          return;
        }

        // Normal content: only show if it's not a function-call placeholder; prefer final_message if present
        let addedAssistantMessage = false;
        if (result.content && !contentIsFunctionCall) {
          const displayContent = extractFinalMessage(result.content) || result.content;
          setMessages((prev) => [
            ...prev,
            { role: "assistant", content: stripMarkdown(displayContent) },
          ]);
          addedAssistantMessage = true;
        }

        if (effectiveToolCalls.length > 0) {
          setPendingToolCalls(effectiveToolCalls);
          if (!result.content || contentIsFunctionCall) {
            const hasTuning = effectiveToolCalls.some((t) => t.name === "show_tuning_modal");
            const needPlaceholder = !result.content || (hasTuning && !hasStyleTool);
            if (needPlaceholder) {
              setMessages((prev) => [
                ...prev,
                {
                  role: "assistant",
                  content: hasTuning
                    ? "Here's Your Tuning Analysis!"
                    : "Here’s Your Singing Style Analysis!",
                },
              ]);
              addedAssistantMessage = true;
            }
          }
        }

        // When model returned function-call-style text (e.g. FN_CALL=True + show_tuning_modal()) but API did not parse tool_calls, show a short placeholder so the response is not blank
        if (!addedAssistantMessage && contentIsFunctionCall && result.content) {
          const mentionsTuning = /show_tuning_modal|tuning|pitch/i.test(result.content);
          const mentionsStyle = /show_style_modal|style|vocal twin/i.test(result.content);
          setMessages((prev) => [
            ...prev,
            {
              role: "assistant",
              content: mentionsTuning
                ? "Here's Your Tuning Analysis!"
                : mentionsStyle
                  ? "Here's Your Singing Style Analysis!"
                  : "Opening…",
            },
          ]);
        }
      } catch (err) {
        const msg =
          err instanceof Error ? err.message : "Failed to get response";
        setError(msg);
      } finally {
        setIsLoading(false);
      }
    },
    [messages, getRecordedBlob, onSetExercises]
  );

  const clearToolCalls = useCallback(() => {
    setPendingToolCalls([]);
  }, []);

  /** Call after the style modal is closed with a result so the LLM can answer with that data. */
  const submitStyleResultForReply = useCallback(
    async (
      result: VaeTagResult,
      pitchContext: string,
      pitchSummary?: PitchHistorySummary | null
    ) => {
      const current = messagesRef.current;
      const apiMessages: ChatMessage[] = current.map((m) => ({
        role: m.role,
        content: m.content,
      }));
      const styleText = formatStyleResultForLLM(result);
      const pitchText = formatPitchSummaryForLLM(pitchSummary ?? null);
      const toolResultText =
        pitchText && pitchText !== "No pitch data."
          ? `Style analysis: ${styleText}. Pitch correctness (from recording): ${pitchText}`
          : `Style analysis: ${styleText}`;
      const followUpMessages: ChatMessage[] = [
        ...apiMessages,
        {
          role: "user" as const,
          content: `Style analysis result: ${toolResultText}\n\n${getExerciseInstructionsForPrompt()}\n\nUsing the conversation above, reply to what the user actually asked. Respond with a JSON object containing only: {"final_message": "your reply in 1-2 sentences"}. If they asked who they sound like or for their vocal twin, mention the closest artist. If they asked for an exercise and then style analysis, give the exercise and briefly include the style result. When pitch data is present, you may mention intonation or pitch accuracy. Match your reply to their request. No other text.`,
        },
      ];
      setIsLoading(true);
      setError(null);
      try {
        const followUp = await sendChatMessage(followUpMessages, pitchContext);
        applySuggestExercises(followUp, setSuggestedExercises);
        const raw =
          followUp.content && !isFunctionCallContent(followUp.content)
            ? followUp.content
            : toolResultText;
        const reply = extractFinalMessage(raw);
        let display = stripMarkdown(reply || raw);
        if (isPlaceholderReply(display)) display = toolResultText;
        setMessages((prev) => [...prev, { role: "assistant", content: display }]);
      } catch (err) {
        const msg =
          err instanceof Error ? err.message : "Failed to get response";
        setError(msg);
      } finally {
        setIsLoading(false);
      }
    },
    []
  );

  /** Grade an exercise using style + pitch results; returns grade 0-100, feedback, and similar exercises. Uses a timeout to avoid hanging. */
  const GRADING_TIMEOUT_MS = 90_000;

  const requestExerciseGrade = useCallback(
    async (
      exerciseName: string,
      styleResult: VaeTagResult | null,
      pitchSummary: PitchHistorySummary | null,
      pitchContext: string
    ): Promise<ExerciseGradeResult | null> => {
      const styleText = styleResult ? formatStyleResultForLLM(styleResult) : "Not available.";
      const pitchText = formatPitchSummaryForLLM(pitchSummary);
      const userContent = `Grade this vocal exercise performance. Exercise: "${exerciseName}". Evaluation criteria: style (tone, timbre, artist similarity) and pitch correctness (intonation, cents deviation). Style analysis: ${styleText}. Pitch analysis: ${pitchText}. Respond with ONLY a JSON object, no other text: {"grade": number 0-100, "feedback": "one short paragraph of feedback", "similar_exercises": ["exercise name 1", "exercise name 2", "exercise name 3"]}. Grade based on both style and pitch when available; if one is missing, grade on what you have and note the gap.`;
      const messages: ChatMessage[] = [{ role: "user", content: userContent }];
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), GRADING_TIMEOUT_MS);
      try {
        const result = await sendChatMessage(messages, pitchContext, {
          signal: controller.signal,
        });
        clearTimeout(timeoutId);
        const raw = result.content?.trim() ?? "";
        const noThink = stripThinkTags(raw);
        let parsed: { grade?: number; feedback?: string; similar_exercises?: string[] };
        try {
          parsed = JSON.parse(noThink) as typeof parsed;
        } catch {
          const jsonMatch = noThink.match(/\{[\s\S]*\}/);
          parsed = jsonMatch ? (JSON.parse(jsonMatch[0]) as typeof parsed) : {};
        }
        const grade = typeof parsed.grade === "number" ? Math.max(0, Math.min(100, parsed.grade)) : 0;
        const feedback = typeof parsed.feedback === "string" ? parsed.feedback : "";
        const similar_exercises = Array.isArray(parsed.similar_exercises)
          ? parsed.similar_exercises.filter((e): e is string => typeof e === "string").slice(0, 5)
          : [];
        return { grade, feedback, similar_exercises };
      } catch (err) {
        clearTimeout(timeoutId);
        if (err instanceof Error && err.name === "AbortError") {
          console.warn("Exercise grading request timed out");
        }
        return null;
      }
    },
    []
  );

  return {
    messages,
    setMessages,
    sendMessage,
    isLoading,
    error,
    pendingToolCalls,
    clearToolCalls,
    suggestedExercises,
    submitStyleResultForReply,
    requestExerciseGrade,
  };
}

import { useState, useEffect, useRef, useMemo } from "react";
import { SlidersHorizontal, Gauge, Music, Search, Loader2, Mic, Music2, Radio, Plus } from "lucide-react";
import { useTuneMeModal } from "@/contexts/TuneMeModalContext";
import { useVocalRangeModal } from "@/contexts/VocalRangeModalContext";
import { useChat, type ExerciseGradeResult, type DisplayMessage } from "@/hooks/useChat";
import type { PitchHistorySummary } from "@/hooks/usePitchHistory";
import type { VocalRangeResult } from "@/components/VocalRangeModal";
import type { VaeTagResult } from "@/lib/analysisApi";
import { DEFAULT_EXERCISES } from "@/lib/chatApi";
import { stripFnCallFromDisplay } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import TuningModal from "@/components/TuningModal";
import StyleModal from "@/components/StyleModal";
import SingAlong from "@/pages/SingAlong";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";

const STORAGE_EXERCISES = "vocalCoach_exercises";
const STORAGE_VOCAL_RANGE = "vocalCoach_vocalRange";
const STORAGE_CHATS = "vocalCoach_chats";
export const STORAGE_TWIN_ARTIST = "vocalCoach_twinArtist";

function loadTwinArtist(): string | null {
  try {
    const s = localStorage.getItem(STORAGE_TWIN_ARTIST);
    return s && s.trim() ? s.trim() : null;
  } catch {
    return null;
  }
}

export interface StoredChat {
  id: string;
  title: string;
  messages: DisplayMessage[];
  updatedAt: number;
}

function loadChats(): StoredChat[] {
  try {
    const raw = localStorage.getItem(STORAGE_CHATS);
    if (raw) {
      const parsed = JSON.parse(raw) as unknown;
      if (Array.isArray(parsed)) {
        return parsed.filter(
          (c): c is StoredChat =>
            c != null &&
            typeof c === "object" &&
            typeof (c as StoredChat).id === "string" &&
            typeof (c as StoredChat).title === "string" &&
            Array.isArray((c as StoredChat).messages) &&
            typeof (c as StoredChat).updatedAt === "number"
        );
      }
    }
  } catch {
    // ignore
  }
  return [];
}

function saveChats(chats: StoredChat[]) {
  try {
    localStorage.setItem(STORAGE_CHATS, JSON.stringify(chats));
  } catch {
    // ignore
  }
}

function loadExercises(): string[] {
  try {
    const raw = localStorage.getItem(STORAGE_EXERCISES);
    if (raw) {
      const parsed = JSON.parse(raw) as unknown;
      if (Array.isArray(parsed) && parsed.length >= 1 && parsed.every((e) => typeof e === "string"))
        return parsed;
    }
  } catch {
    // ignore
  }
  return [...DEFAULT_EXERCISES];
}

function loadVocalRange(): VocalRangeResult | null {
  try {
    const raw = localStorage.getItem(STORAGE_VOCAL_RANGE);
    if (raw) {
      const parsed = JSON.parse(raw) as {
        lowNote?: string;
        highNote?: string;
        chestRange?: { lowNote: string; highNote: string };
        headRange?: { lowNote: string; highNote: string };
      };
      if (parsed?.lowNote != null && parsed?.highNote != null)
        return {
          lowNote: String(parsed.lowNote),
          highNote: String(parsed.highNote),
          start: String(parsed.lowNote),
          end: String(parsed.highNote),
          chestRange:
            parsed.chestRange?.lowNote != null && parsed.chestRange?.highNote != null
              ? { lowNote: String(parsed.chestRange.lowNote), highNote: String(parsed.chestRange.highNote) }
              : undefined,
          headRange:
            parsed.headRange?.lowNote != null && parsed.headRange?.highNote != null
              ? { lowNote: String(parsed.headRange.lowNote), highNote: String(parsed.headRange.highNote) }
              : undefined,
        };
    }
  } catch {
    // ignore
  }
  return null;
}

const SUGGESTED_QUESTIONS = [
  "Give me a vocal exercise, then analyze my style",
  "Who is my natural vocal twin?",
  "How can I sing more in tune?",
  "What should I practice? Then check my pitch",
  "What's my vocal range?",
];

const SIDEBAR_FEATURES = [
  { id: "tune", label: "Tune Me", icon: SlidersHorizontal, kind: "modal" as const },
  { id: "sing", label: "Sing Along", icon: Music, kind: "view" as const },
  { id: "range", label: "Vocal Range", icon: Gauge, kind: "modal" as const },
] as const;

const EMPTY_PITCH_SUMMARY: PitchHistorySummary = {
  totalDetections: 0,
  notes: [],
  overallAvgDeviation: 0,
  currentPitch: null,
};

const Index = () => {
  const { showTuneMeModal } = useTuneMeModal();
  const { showVocalRangeModal } = useVocalRangeModal();
  const [mainView, setMainView] = useState<"chat" | "sing-along">("chat");
  const [twinArtist, setTwinArtist] = useState<string | null>(() => loadTwinArtist());
  const [exercises, setExercises] = useState<string[]>(() => loadExercises());
  const [vocalRange, setVocalRange] = useState<VocalRangeResult | null>(() => loadVocalRange());

  // When returning from Sing Along, refresh twin artist from localStorage (karaoke may have saved it)
  useEffect(() => {
    if (mainView === "chat") setTwinArtist(loadTwinArtist());
  }, [mainView]);

  const { messages, setMessages, sendMessage, isLoading, error, pendingToolCalls, clearToolCalls, suggestedExercises, submitStyleResultForReply, requestExerciseGrade } =
    useChat({
      onSetExercises: (next) => {
        localStorage.setItem(STORAGE_EXERCISES, JSON.stringify(next));
        setExercises(next);
      },
    });

  const [chatHistory, setChatHistory] = useState<StoredChat[]>(() => loadChats());
  const [currentChatId, setCurrentChatId] = useState<string | null>(null);
  const [input, setInput] = useState("");
  const [tuningOpen, setTuningOpen] = useState(false);
  const [styleOpen, setStyleOpen] = useState(false);
  const [lastStyleResult, setLastStyleResult] = useState<VaeTagResult | null>(null);
  const [lastPitchSummary, setLastPitchSummary] = useState<PitchHistorySummary | null>(null);
  const [gradingExercise, setGradingExercise] = useState<string | null>(null);
  const [gradeResult, setGradeResult] = useState<ExerciseGradeResult | null>(null);
  const [gradeLoading, setGradeLoading] = useState(false);
  const [exerciseForStyleModal, setExerciseForStyleModal] = useState<string | null>(null);
  /** Last grade per exercise name (for card display when opened from card, not chat) */
  const [exerciseScores, setExerciseScores] = useState<Record<string, number>>({});
  const chatScrollBottomRef = useRef<HTMLDivElement>(null);

  const lastAssistantMessage = useMemo(() => {
    const m = [...messages].reverse().find((msg) => msg.role === "assistant");
    const raw = (m?.content ?? "").trim();
    const content = stripFnCallFromDisplay(raw);
    return content || undefined;
  }, [messages]);

  useEffect(() => {
    chatScrollBottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  // Persist current chat to localStorage when messages change (from sending or loading)
  useEffect(() => {
    if (messages.length === 0) return;
    const title = messages[0]?.role === "user" ? messages[0].content.slice(0, 50) : "New Chat";
    const updatedAt = Date.now();
    if (currentChatId) {
      setChatHistory((prev) => {
        const next = prev.map((c) =>
          c.id === currentChatId ? { ...c, title, messages, updatedAt } : c
        );
        saveChats(next);
        return next;
      });
    } else {
      const id = `chat-${updatedAt}`;
      setCurrentChatId(id);
      setChatHistory((prev) => {
        const next = [{ id, title, messages, updatedAt }, ...prev];
        saveChats(next);
        return next;
      });
    }
  }, [messages, currentChatId]);

  const handleNewChat = () => {
    setCurrentChatId(null);
    setMessages([]);
  };

  const handleLoadChat = (chat: StoredChat) => {
    setCurrentChatId(chat.id);
    setMessages(chat.messages);
  };

  useEffect(() => {
    if (pendingToolCalls.length === 0) return;
    for (const tc of pendingToolCalls) {
      if (tc.name === "show_tuning_modal") setTuningOpen(true);
      if (tc.name === "show_style_modal") setStyleOpen(true);
    }
    clearToolCalls();
  }, [pendingToolCalls, clearToolCalls]);

  const handleSend = () => {
    const text = input.trim();
    if (!text || isLoading) return;
    setInput("");
    sendMessage(text, "");
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleSuggestion = (question: string) => {
    if (isLoading) return;
    sendMessage(question, "");
  };

  const openVocalRangeModal = () => {
    showVocalRangeModal((result) => {
      localStorage.setItem(STORAGE_VOCAL_RANGE, JSON.stringify(result));
      setVocalRange(result);
    });
  };

  /** Open the style modal to do this exercise; after user records and modal closes, we grade and show the grade modal. */
  const handleStartExercise = (exerciseName: string) => {
    setExerciseForStyleModal(exerciseName);
    setStyleOpen(true);
  };

  const handleGradeFromResult = async (
    exerciseName: string,
    styleResult: VaeTagResult,
    pitchSummary: PitchHistorySummary | null
  ) => {
    setGradingExercise(exerciseName);
    setGradeResult(null);
    setGradeLoading(true);
    try {
      const result = await requestExerciseGrade(
        exerciseName,
        styleResult,
        pitchSummary,
        ""
      );
      setGradeResult(result ?? null);
    } finally {
      setGradeLoading(false);
      setExerciseForStyleModal(null);
    }
  };

  const closeGradeModal = () => {
    if (gradingExercise && gradeResult != null) {
      setExerciseScores((prev) => ({ ...prev, [gradingExercise]: gradeResult.grade }));
    }
    setGradingExercise(null);
    setGradeResult(null);
  };

  return (
    <div className="flex min-h-screen page-gradient">
      {/* Left sidebar: fixed height, scrolls independently of main content */}
      <aside className="sticky top-0 flex h-screen w-64 shrink-0 flex-col overflow-y-auto border-r border-border/60 bg-card/40 py-6 pl-5 pr-5">
        <h2 className="mb-3 text-center text-xl font-bold text-foreground">
          Vocal Coach AI
        </h2>
        <div className="mb-4 rounded-xl px-4 py-3 text-sm font-medium bg-muted/80 text-muted-foreground">
          <p className="text-xs font-semibold uppercase tracking-wider mb-1.5">
            Twin Artist
          </p>
          <p className="text-sm font-medium text-foreground min-h-[1.5rem]">
            {twinArtist ?? "—"}
          </p>
        </div>
        <div className="mb-6 rounded-xl px-4 py-3 text-sm font-medium bg-muted/80 text-muted-foreground">
          {vocalRange ? (
            <div className="space-y-1.5">
              <div className="font-semibold text-foreground/90">Chest</div>
              <div className="text-xs">
                {vocalRange.chestRange != null
                  ? `${vocalRange.chestRange.lowNote} – ${vocalRange.chestRange.highNote}`
                  : "—"}
              </div>
              <div className="font-semibold text-foreground/90 pt-0.5">Head</div>
              <div className="text-xs">
                {vocalRange.headRange != null
                  ? `${vocalRange.headRange.lowNote} – ${vocalRange.headRange.highNote}`
                  : "—"}
              </div>
              <div className="text-xs text-muted-foreground pt-1 border-t border-border/60">
                Overall: {vocalRange.lowNote} – {vocalRange.highNote}
              </div>
            </div>
          ) : (
            "Range: —"
          )}
        </div>
        <nav className="flex flex-col gap-4" aria-label="Main actions">
          {SIDEBAR_FEATURES.map((item) => {
            const { id, label, icon: Icon, kind } = item;
            const isSingAlong = kind === "view";
            const isActive = isSingAlong && mainView === "sing-along";
            const className =
              `card-hover-lift flex h-14 w-full items-center gap-4 rounded-xl glass-card px-4 py-3 transition-colors hover:bg-primary/10 hover:text-primary hover:shadow-lg hover:shadow-primary/10 text-left ${isActive ? "bg-primary/10 text-primary" : "text-muted-foreground"}`;
            return (
              <div key={id} className="flex flex-col gap-0">
                {isSingAlong ? (
                  <button
                    type="button"
                    onClick={() => setMainView("sing-along")}
                    className={className}
                    aria-label={label}
                  >
                    <Icon className="h-6 w-6 shrink-0" />
                    <span className="text-base font-medium truncate">{label}</span>
                  </button>
                ) : (
                  <button
                    type="button"
                    onClick={() =>
                      id === "tune" ? showTuneMeModal() : openVocalRangeModal()
                    }
                    className={className}
                    aria-label={label}
                  >
                    <Icon className="h-6 w-6 shrink-0" />
                    <span className="text-base font-medium truncate">{label}</span>
                  </button>
                )}
              </div>
            );
          })}
        </nav>
        <div className="mt-6 flex flex-1 min-h-0 flex-col">
          <button
            type="button"
            onClick={handleNewChat}
            className="mb-4 w-full rounded-xl border-2 border-border bg-muted/50 py-3 text-base font-semibold text-foreground hover:bg-muted hover:border-foreground/20 transition-colors"
          >
            New Chat
          </button>
          <p className="mb-2 text-center text-sm font-bold text-foreground uppercase tracking-wider">
            Previous Chats
          </p>
          <ScrollArea className="flex-1 min-h-0">
            <div className="flex flex-col gap-1 pr-2">
              {chatHistory.map((chat) => (
                <button
                  key={chat.id}
                  type="button"
                  onClick={() => handleLoadChat(chat)}
                  className={`w-full rounded-lg px-3 py-2.5 text-left text-sm truncate transition-colors ${
                    currentChatId === chat.id
                      ? "bg-primary/15 text-primary font-medium"
                      : "text-muted-foreground hover:bg-muted/60 hover:text-foreground"
                  }`}
                  title={chat.title}
                >
                  {chat.title || "New Chat"}
                </button>
              ))}
            </div>
          </ScrollArea>
        </div>
      </aside>

      {/* Main content */}
      <div className="flex flex-1 flex-col justify-start items-center pl-2 pr-4 pt-36 pb-8 min-w-0 overflow-y-auto">
        {mainView === "sing-along" ? (
          <SingAlong embedded onBack={() => setMainView("chat")} />
        ) : (
        <div className="w-full max-w-4xl flex flex-col items-center">
        {/* Spacing */}
        <div className="mb-8 w-full" aria-hidden />
        {/* Chat area */}
        <div className="flex flex-1 flex-col items-center w-full">
          {messages.length > 0 && (
            <ScrollArea className="mb-6 w-full max-w-2xl h-[40vh] min-h-[200px] max-h-[50vh] rounded-2xl glass-card px-5 py-4">
              <div className="space-y-4 min-h-full">
                {messages.map((msg, i) => (
                  <div
                    key={i}
                    className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
                  >
                    <div
                      className={`max-w-[85%] rounded-2xl px-4 py-3 text-base whitespace-pre-wrap ${
                        msg.role === "user"
                          ? "bg-primary text-primary-foreground shadow-sm"
                          : "bg-muted/80 text-foreground"
                      }`}
                    >
                      {stripFnCallFromDisplay(msg.content)}
                    </div>
                  </div>
                ))}
                {isLoading && (
                  <div className="flex justify-start">
                    <div className="flex items-center gap-2 rounded-2xl bg-muted/80 px-4 py-3 text-base text-muted-foreground">
                      <Loader2 className="h-5 w-5 animate-spin" />
                      Thinking...
                    </div>
                  </div>
                )}
                {error && (
                  <div className="rounded-2xl bg-destructive/10 px-4 py-3 text-base text-destructive">
                    {error}
                  </div>
                )}
                <div ref={chatScrollBottomRef} />
              </div>
            </ScrollArea>
          )}

          {suggestedExercises.length > 0 && (
            <div className="mb-4 w-full max-w-2xl">
              <p className="text-xs font-medium text-muted-foreground mb-2 text-center">Suggested exercise — add to your list</p>
              <div className="flex flex-wrap justify-center gap-2">
                {suggestedExercises.map((ex, j) => {
                  const alreadyAdded = exercises.includes(ex);
                  return (
                    <div key={j} className="flex items-center gap-2 rounded-full bg-card/90 border border-border/60 pl-4 pr-1 py-1.5">
                      <span className="text-sm text-foreground">{ex}</span>
                      <Button
                        type="button"
                        variant="outline"
                        size="sm"
                        className="shrink-0 h-8 rounded-full gap-1"
                        disabled={alreadyAdded}
                        onClick={() => {
                          if (alreadyAdded) return;
                          setExercises((prev) => {
                            const next = [...prev, ex];
                            localStorage.setItem(STORAGE_EXERCISES, JSON.stringify(next));
                            return next;
                          });
                        }}
                      >
                        <Plus className="h-4 w-4" />
                        {alreadyAdded ? "Added" : "Add"}
                      </Button>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Spacing (replaces Ask your personal vocal coach title) */}
          <div className="mb-3 w-full" aria-hidden />
          <div className="w-full max-w-4xl">
            <div className="flex gap-4 rounded-2xl glass-card px-6 py-4 focus-within:ring-2 focus-within:ring-primary/30">
              <Input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask about your voice..."
                disabled={isLoading}
                className="h-16 border-0 bg-transparent !text-2xl shadow-none focus-visible:ring-0 focus-visible:ring-offset-0 placeholder:text-muted-foreground/80 placeholder:text-xl"
              />
              <Button
                size="icon"
                className="h-16 w-16 shrink-0 rounded-xl bg-primary text-primary-foreground hover:bg-primary/90"
                onClick={handleSend}
                disabled={!input.trim() || isLoading}
              >
                <Search className="h-7 w-7" />
              </Button>
            </div>

            {messages.length === 0 && (
              <div className="mt-6 flex flex-wrap justify-center gap-3">
                {SUGGESTED_QUESTIONS.map((q) => (
                  <button
                    key={q}
                    type="button"
                    onClick={() => handleSuggestion(q)}
                    disabled={isLoading}
                    className="chip-hover rounded-full bg-card/90 px-6 py-3.5 text-base text-muted-foreground hover:bg-primary/10 hover:text-primary disabled:opacity-50 border border-border/60 shadow-sm"
                  >
                    {q}
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* Suggest an exercise: gets one suggestion from AI and shows Add button */}
          <div className="mt-8 flex justify-center">
            <Button
              type="button"
              variant="outline"
              size="lg"
              onClick={() => handleSuggestion("Suggest one vocal exercise I can add to my list.")}
              disabled={isLoading}
              className="rounded-xl gap-2"
            >
              <Plus className="h-5 w-5" />
              Suggest an Exercise
            </Button>
          </div>

          {/* Exercise cards: below search bar */}
          <section className="mt-10 flex w-full flex-nowrap justify-center gap-5" aria-label="Daily Exercises">
            {exercises.map((label, i) => {
              const score = exerciseScores[label];
              const hasScore = score != null;
              const CardIcon = [Mic, Music2, Radio][i % 3];
              return (
                <button
                  key={`${i}-${label}`}
                  type="button"
                  onClick={() => handleStartExercise(label)}
                  disabled={gradeLoading}
                  className="card-hover-lift glass-card relative flex min-w-0 flex-1 min-h-[160px] flex-col items-center justify-center gap-3 rounded-2xl px-6 py-6 text-center disabled:opacity-50 disabled:pointer-events-none"
                >
                  {hasScore && (
                    <span className="absolute top-4 right-4 flex h-10 w-10 items-center justify-center rounded-full bg-primary/15 text-sm font-bold text-primary">
                      {score}%
                    </span>
                  )}
                  <CardIcon className="h-10 w-10 shrink-0 text-primary/80" aria-hidden />
                  <span className="text-lg font-semibold text-foreground leading-tight">
                    {label}
                  </span>
                  <span className="text-sm text-muted-foreground">
                    {gradeLoading && gradingExercise === label
                      ? "Grading…"
                      : hasScore
                        ? "Try again"
                        : "Start"}
                  </span>
                </button>
              );
            })}
          </section>
        </div>
        </div>
        )}
      </div>

      <TuningModal
        open={tuningOpen}
        onOpenChange={setTuningOpen}
        summary={EMPTY_PITCH_SUMMARY}
      />
      <StyleModal
        open={styleOpen}
        onOpenChange={(open) => {
          if (!open) setExerciseForStyleModal(null);
          setStyleOpen(open);
        }}
        getRecordedBlob={() => null}
        exerciseName={exerciseForStyleModal}
        instructionsContent={exerciseForStyleModal ? undefined : lastAssistantMessage}
        onCloseWithResult={(result, pitchSummaryFromModal) => {
          if (result) {
            setLastStyleResult(result);
            if (exerciseForStyleModal) {
              // From exercise card: grade with style + pitch from the same recording; do not add to chat
              handleGradeFromResult(
                exerciseForStyleModal,
                result,
                pitchSummaryFromModal ?? lastPitchSummary
              );
            } else {
              // From chat (e.g. "analyze my style"): send result to chat with pitch from recording when available
              submitStyleResultForReply(result, "", pitchSummaryFromModal ?? undefined);
            }
            if (pitchSummaryFromModal != null) setLastPitchSummary(pitchSummaryFromModal);
          }
        }}
      />

      {/* Exercise grade result modal */}
      <Dialog open={gradingExercise !== null} onOpenChange={(open) => !open && closeGradeModal()}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>
              {gradingExercise ? `How Did You Do: ${gradingExercise}?` : "Exercise Feedback"}
            </DialogTitle>
            <DialogDescription>
              Evaluation based on your singing style and pitch accuracy.
            </DialogDescription>
          </DialogHeader>
          {gradeLoading && (
            <div className="flex items-center gap-2 py-4 text-muted-foreground">
              <Loader2 className="h-5 w-5 animate-spin" />
              <span className="text-sm">Analyzing your performance…</span>
            </div>
          )}
          {!gradeLoading && gradeResult && (
            <div className="space-y-4">
              <div className="flex items-baseline gap-2">
                <span className="text-3xl font-bold text-foreground">{gradeResult.grade}</span>
                <span className="text-muted-foreground">/ 100</span>
              </div>
              <p className="text-sm text-foreground">{gradeResult.feedback}</p>
              {gradeResult.similar_exercises.length > 0 && (
                <div>
                  <p className="text-xs font-medium text-muted-foreground mb-2">Similar exercises to try</p>
                  <ul className="space-y-2">
                    {gradeResult.similar_exercises.map((ex, j) => {
                      const alreadyAdded = exercises.includes(ex);
                      return (
                        <li key={j} className="flex items-center justify-between gap-2 text-sm text-foreground">
                          <span>{ex}</span>
                          <Button
                            type="button"
                            variant="outline"
                            size="sm"
                            className="shrink-0 h-8 gap-1"
                            disabled={alreadyAdded}
                            onClick={() => {
                              if (alreadyAdded) return;
                              setExercises((prev) => {
                                const next = [...prev, ex];
                                localStorage.setItem(STORAGE_EXERCISES, JSON.stringify(next));
                                return next;
                              });
                            }}
                          >
                            <Plus className="h-4 w-4" />
                            {alreadyAdded ? "Added" : "Add"}
                          </Button>
                        </li>
                      );
                    })}
                  </ul>
                </div>
              )}
            </div>
          )}
          {!gradeLoading && !gradeResult && gradingExercise && (
            <p className="text-sm text-muted-foreground">
              Could not get a grade (request timed out or failed). Try again.
            </p>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default Index;

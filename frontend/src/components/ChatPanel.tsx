import { useState, useRef, useEffect } from "react";
import { MessageCircle, Send, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetDescription,
  SheetTrigger,
} from "@/components/ui/sheet";
import { useChat } from "@/hooks/useChat";
import { stripFnCallFromDisplay } from "@/lib/utils";
import TuningModal from "@/components/TuningModal";
import StyleModal from "@/components/StyleModal";
import type { PitchHistorySummary } from "@/hooks/usePitchHistory";

interface ChatPanelProps {
  pitchContext: string;
  isListening: boolean;
  getHistorySummary: () => PitchHistorySummary;
  getRecordedBlob: () => Blob | null;
}

export default function ChatPanel({
  pitchContext,
  isListening,
  getHistorySummary,
  getRecordedBlob,
}: ChatPanelProps) {
  const { messages, sendMessage, isLoading, error, pendingToolCalls, clearToolCalls, submitStyleResultForReply } =
    useChat({ getRecordedBlob });
  const [input, setInput] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);
  const [tuningOpen, setTuningOpen] = useState(false);
  const [styleOpen, setStyleOpen] = useState(false);

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, isLoading]);

  // Auto-open modals based on LLM tool calls
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
    sendMessage(text, pitchContext);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <>
      <Sheet>
        <SheetTrigger asChild>
          <Button
            size="icon"
            className="fixed bottom-6 right-6 z-40 h-14 w-14 rounded-full shadow-lg"
          >
            <MessageCircle className="h-6 w-6" />
          </Button>
        </SheetTrigger>

        <SheetContent side="right" className="flex flex-col p-0 sm:max-w-md">
          <SheetHeader className="border-b px-6 py-4">
            <SheetTitle>Singing Coach</SheetTitle>
            <SheetDescription>
              {isListening
                ? "Ask me about your singing — I can see your pitch data!"
                : "Turn on your mic so I can analyze your singing."}
            </SheetDescription>
          </SheetHeader>

          {/* Messages area */}
          <ScrollArea className="flex-1 px-4 py-4">
            <div ref={scrollRef} className="space-y-4">
              {messages.length === 0 && (
                <p className="text-center text-sm text-muted-foreground pt-8">
                  Ask a question about your singing!
                </p>
              )}

              {messages.map((msg, i) => (
                <div
                  key={i}
                  className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
                >
                  <div
                    className={`max-w-[85%] rounded-lg px-3 py-2 text-sm whitespace-pre-wrap ${
                      msg.role === "user"
                        ? "bg-primary text-primary-foreground"
                        : "bg-muted text-foreground"
                    }`}
                  >
                    {stripFnCallFromDisplay(msg.content)}
                  </div>
                </div>
              ))}

              {isLoading && (
                <div className="flex justify-start">
                  <div className="flex items-center gap-2 rounded-lg bg-muted px-3 py-2 text-sm text-muted-foreground">
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Thinking...
                  </div>
                </div>
              )}

              {error && (
                <div className="rounded-lg bg-destructive/10 px-3 py-2 text-sm text-destructive">
                  {error}
                </div>
              )}
            </div>
          </ScrollArea>

          {/* Input area */}
          <div className="border-t px-4 py-3">
            <div className="flex gap-2">
              <Input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="How is my pitch?"
                disabled={isLoading}
              />
              <Button
                size="icon"
                onClick={handleSend}
                disabled={!input.trim() || isLoading}
              >
                <Send className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </SheetContent>
      </Sheet>

      {/* Modals */}
      <TuningModal
        open={tuningOpen}
        onOpenChange={setTuningOpen}
        summary={getHistorySummary()}
      />
      <StyleModal
        open={styleOpen}
        onOpenChange={setStyleOpen}
        getRecordedBlob={getRecordedBlob}
        onCloseWithResult={(result) => {
          if (result) submitStyleResultForReply(result, pitchContext);
        }}
      />
    </>
  );
}

# Singing Coach Chatbot — Architecture & Modal Reference

## Overview

The Microphone Sound Analyzer page (`/`, `Index.tsx`) includes an LLM-powered singing coach chatbot. The chatbot uses the K2-Think API to answer questions about the user's singing. When the user asks about **tuning** or **singing style**, the LLM calls a function and the frontend automatically opens the corresponding modal.

---

## System Architecture

```
User types question in ChatPanel
        │
        ▼
  useChat.sendMessage(text, pitchContext)
        │
        ▼
  chatApi.sendChatMessage()
    ├── Sends conversation + pitch context as system message
    ├── Includes tool definitions: show_tuning_modal, show_style_modal
    └── POST https://api.k2think.ai/v1/chat/completions
        │
        ▼
  K2-Think LLM response
    ├── content: text reply (may be empty if only tool call)
    └── tool_calls: ["show_tuning_modal"] or ["show_style_modal"]
        │
        ▼
  useChat sets pendingToolCalls state
        │
        ▼
  ChatPanel useEffect detects pendingToolCalls
    ├── "show_tuning_modal" → opens TuningModal
    └── "show_style_modal" → opens StyleModal
```

---

## File Map

| File | Purpose |
|------|---------|
| `src/lib/chatApi.ts` | K2-Think API client with function calling |
| `src/hooks/usePitchHistory.ts` | Rolling 30s pitch tracker, exposes structured summary |
| `src/hooks/useChat.ts` | Chat state, sends messages, surfaces tool calls |
| `src/hooks/useAudioAnalyser.ts` | Mic input, provides `getRecordedBlob()` |
| `src/components/ChatPanel.tsx` | Sheet drawer UI, renders both modals |
| `src/components/TuningModal.tsx` | Pitch accuracy modal |
| `src/components/StyleModal.tsx` | VAE singing style modal |
| `src/pages/Index.tsx` | Wires everything together on the Analyzer page |

---

## How LLM Function Calling Works

In `src/lib/chatApi.ts`, two tools are defined and sent with every API request:

```ts
const TOOLS = [
  {
    type: "function",
    function: {
      name: "show_tuning_modal",
      description: "Show a tuning analysis modal with the user's pitch accuracy data from the last 30 seconds. Call this when the user asks about their tuning, pitch accuracy, intonation, whether they are sharp or flat, or any question about how well they are hitting the notes.",
      parameters: { type: "object", properties: {} },
    },
  },
  {
    type: "function",
    function: {
      name: "show_style_modal",
      description: "Show a singing style analysis modal that analyzes the user's recorded audio to identify their vocal style, closest artist match, and style attributes. Call this when the user asks about their singing style, vocal tone, what artist they sound like, their vocal timbre, or any question about the character/quality of their voice.",
      parameters: { type: "object", properties: {} },
    },
  },
];
```

The LLM decides which tool to call based on the user's message. The response is parsed in `sendChatMessage()`:

```ts
// From the API response:
const choice = data.choices?.[0]?.message;
const content: string = choice?.content ?? "";       // text reply
const toolCalls: ToolCall[] = [];                     // parsed tool calls
if (choice?.tool_calls && Array.isArray(choice.tool_calls)) {
  for (const tc of choice.tool_calls) {
    const name = tc.function?.name;
    if (name === "show_tuning_modal" || name === "show_style_modal") {
      toolCalls.push(name);
    }
  }
}
return { content, toolCalls };
```

In `useChat.ts`, `pendingToolCalls` state is set when tool calls come back. In `ChatPanel.tsx`, a `useEffect` watches `pendingToolCalls` and auto-opens the corresponding modal:

```ts
useEffect(() => {
  if (pendingToolCalls.length === 0) return;
  for (const tc of pendingToolCalls) {
    if (tc === "show_tuning_modal") setTuningOpen(true);
    if (tc === "show_style_modal") setStyleOpen(true);
  }
  clearToolCalls();
}, [pendingToolCalls, clearToolCalls]);
```

---

## TuningModal

**File:** `src/components/TuningModal.tsx`

### Props

```ts
interface TuningModalProps {
  open: boolean;                          // controlled open state
  onOpenChange: (open: boolean) => void;  // close handler
  summary: PitchHistorySummary;           // pitch data from usePitchHistory
}
```

### Data Source

`PitchHistorySummary` is returned by `usePitchHistory.getHistorySummary()` and contains:

```ts
interface PitchHistorySummary {
  totalDetections: number;          // total pitch readings in 30s window
  notes: NoteSummary[];             // per-note breakdown, sorted by count desc
  overallAvgDeviation: number;      // mean |centsOff| across all detections
  currentPitch: PitchResult | null; // the live pitch reading right now
}

interface NoteSummary {
  note: string;          // e.g. "A4"
  count: number;         // how many times detected
  avgDeviation: number;  // mean centsOff (signed) for this note
}
```

### Current Display

1. **Accuracy score** — `max(0, 100 - avgDeviation * 2)`, color-coded green/yellow/red.
2. **Current pitch** — live note, frequency, and cents offset.
3. **Note breakdown** — each note with a bar chart (% of total), count, and avg deviation.

### How to Customize TuningModal

The modal receives `summary` as a prop. All pitch data is already computed — you just render it. To add new sections:

```tsx
// Inside the TuningModal component, after the existing sections:
export default function TuningModal({ open, onOpenChange, summary }: TuningModalProps) {
  const { totalDetections, notes, overallAvgDeviation, currentPitch } = summary;

  // Example: add a "sharp vs flat" tendency
  const sharpCount = notes.filter(n => n.avgDeviation > 5).length;
  const flatCount = notes.filter(n => n.avgDeviation < -5).length;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        {/* ...existing header and sections... */}

        {/* YOUR NEW SECTION */}
        <div>
          <h4 className="text-sm font-semibold mb-1">Tendency</h4>
          <p className="text-sm text-muted-foreground">
            {sharpCount > flatCount
              ? "You tend to sing sharp. Try relaxing your throat."
              : flatCount > sharpCount
                ? "You tend to sing flat. Focus on breath support."
                : "Your intonation is well balanced!"}
          </p>
        </div>
      </DialogContent>
    </Dialog>
  );
}
```

Key points:
- `summary` is a snapshot taken at the moment the modal opens (called in ChatPanel as `getHistorySummary()`).
- `notes` is sorted by detection count descending — the most-sung note is first.
- `avgDeviation` is signed: positive = sharp, negative = flat.
- `overallAvgDeviation` is the mean of absolute values (always positive).

---

## StyleModal

**File:** `src/components/StyleModal.tsx`

### Props

```ts
interface StyleModalProps {
  open: boolean;                          // controlled open state
  onOpenChange: (open: boolean) => void;  // close handler
  getRecordedBlob: () => Blob | null;     // grabs recorded mic audio as WAV
}
```

### Data Source

When the modal opens, it:
1. Calls `getRecordedBlob()` to get a WAV blob of whatever the mic has recorded so far (without stopping the mic).
2. Sends that blob to the VAE backend via `fetchVaeTags(blob)` from `src/lib/analysisApi.ts`.
3. Receives a `VaeTagResult`:

```ts
interface VaeTagResult {
  artist_probs: Record<string, number>;               // all artist probabilities
  top_artist: string | null;                           // best match
  top_3_artists: string[];                             // top 3 artist matches
  attributes: { tag: string; confidence: number }[];   // all style attributes
  top_3_attributes: { tag: string; confidence: number }[]; // top 3 style tags
  n_chunks: number;                                    // audio chunks analyzed
}
```

### Current Display

1. **Closest artist match** — `top_artist`
2. **Similar artists** — `top_3_artists` as pill badges
3. **Style tags** — `top_3_attributes` with confidence percentages

### How to Customize StyleModal

The modal manages its own async lifecycle (loading/error/result). The key extension point is inside the `{result && !loading && (...)}` block:

```tsx
export default function StyleModal({ open, onOpenChange, getRecordedBlob }: StyleModalProps) {
  const [result, setResult] = useState<VaeTagResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch runs automatically when modal opens (useEffect on `open`)

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        {/* ...existing header, loading, error sections... */}

        {result && !loading && (
          <div className="space-y-4">
            {/* ...existing artist + tags sections... */}

            {/* YOUR NEW SECTION: full probability breakdown */}
            <div>
              <h4 className="text-sm font-semibold mb-1">All attributes</h4>
              {result.attributes
                .sort((a, b) => b.confidence - a.confidence)
                .map(({ tag, confidence }) => (
                  <div key={tag} className="flex justify-between text-sm">
                    <span>{tag}</span>
                    <span className="text-muted-foreground">
                      {(confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                ))}
            </div>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}
```

Key points:
- `getRecordedBlob()` returns the full recording buffer without stopping the mic. It comes from `useAudioAnalyser`.
- The VAE endpoint is `POST /analyze/vae-tags?temperature=5` on the backend (`VITE_API_URL` or `localhost:8000`).
- The blob is sent as `FormData` with field name `"audio"`.
- If the mic hasn't been started or no audio was captured, `getRecordedBlob()` returns `null` and the modal shows an error message.

---

## Adding a New Tool-Triggered Modal

To add a third modal (e.g. `show_breathing_modal`):

### 1. Define the tool in `src/lib/chatApi.ts`

Add to the `TOOLS` array:
```ts
{
  type: "function",
  function: {
    name: "show_breathing_modal",
    description: "Show a breathing analysis modal. Call when the user asks about breath support or breathing technique.",
    parameters: { type: "object", properties: {} },
  },
}
```

Update the `ToolCall` type:
```ts
export type ToolCall = "show_tuning_modal" | "show_style_modal" | "show_breathing_modal";
```

Update the parsing loop to include the new name.

### 2. Create the modal component

Create `src/components/BreathingModal.tsx` following the same pattern as TuningModal or StyleModal (Dialog with `open`/`onOpenChange` props).

### 3. Wire it into ChatPanel

In `src/components/ChatPanel.tsx`:
- Import the new modal
- Add `const [breathingOpen, setBreathingOpen] = useState(false);`
- Add to the `useEffect`: `if (tc === "show_breathing_modal") setBreathingOpen(true);`
- Render `<BreathingModal open={breathingOpen} onOpenChange={setBreathingOpen} ... />`

### 4. Update the system prompt

In `usePitchHistory.ts` `formatContext()`, add the new tool to the instructions so the LLM knows it exists.

---

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `VITE_K2_API_KEY` | API key for K2-Think LLM (set in `frontend/.env`) |
| `VITE_API_URL` | Backend URL for VAE analysis (defaults to `http://localhost:8000`) |

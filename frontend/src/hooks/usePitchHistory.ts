import { useRef, useCallback } from "react";
import type { PitchResult } from "@/lib/pitchDetection";

interface PitchSnapshot extends PitchResult {
  timestamp: number;
}

export interface NoteSummary {
  note: string;
  count: number;
  avgDeviation: number;
}

export interface PitchHistorySummary {
  totalDetections: number;
  notes: NoteSummary[];
  overallAvgDeviation: number;
  currentPitch: PitchResult | null;
}

const HISTORY_WINDOW_MS = 30_000;

/** Build a PitchHistorySummary from an array of pitch results (e.g. from analyzing a recording). */
export function buildPitchSummaryFromResults(results: PitchResult[]): PitchHistorySummary {
  const noteMap = new Map<string, { count: number; totalDeviation: number }>();
  let totalDev = 0;
  for (const r of results) {
    totalDev += Math.abs(r.centsOff);
    const existing = noteMap.get(r.noteLabel);
    if (existing) {
      existing.count++;
      existing.totalDeviation += r.centsOff;
    } else {
      noteMap.set(r.noteLabel, { count: 1, totalDeviation: r.centsOff });
    }
  }
  const notes: NoteSummary[] = [...noteMap.entries()]
    .sort((a, b) => b[1].count - a[1].count)
    .map(([note, { count, totalDeviation }]) => ({
      note,
      count,
      avgDeviation: Math.round(totalDeviation / count),
    }));
  return {
    totalDetections: results.length,
    notes,
    overallAvgDeviation: results.length > 0 ? Math.round(totalDev / results.length) : 0,
    currentPitch: results.length > 0 ? results[results.length - 1] : null,
  };
}

export function usePitchHistory(pitchResult: PitchResult | null) {
  const historyRef = useRef<PitchSnapshot[]>([]);

  // Push new result into history
  if (pitchResult) {
    const now = Date.now();
    historyRef.current.push({ ...pitchResult, timestamp: now });
    // Evict entries older than 30s
    const cutoff = now - HISTORY_WINDOW_MS;
    historyRef.current = historyRef.current.filter(
      (s) => s.timestamp >= cutoff
    );
  }

  const currentPitch = pitchResult;

  const getHistorySummary = useCallback((): PitchHistorySummary => {
    const history = historyRef.current;
    const noteMap = new Map<string, { count: number; totalDeviation: number }>();
    let totalDev = 0;

    for (const snap of history) {
      totalDev += Math.abs(snap.centsOff);
      const existing = noteMap.get(snap.noteLabel);
      if (existing) {
        existing.count++;
        existing.totalDeviation += snap.centsOff;
      } else {
        noteMap.set(snap.noteLabel, {
          count: 1,
          totalDeviation: snap.centsOff,
        });
      }
    }

    const notes: NoteSummary[] = [...noteMap.entries()]
      .sort((a, b) => b[1].count - a[1].count)
      .map(([note, { count, totalDeviation }]) => ({
        note,
        count,
        avgDeviation: Math.round(totalDeviation / count),
      }));

    return {
      totalDetections: history.length,
      notes,
      overallAvgDeviation:
        history.length > 0 ? Math.round(totalDev / history.length) : 0,
      currentPitch,
    };
  }, [currentPitch]);

  const formatContext = useCallback((): string => {
    const lines: string[] = [
      "You are an enthusiastic, supportive singing coach. The user is practicing singing. Your job is to help them get better: explain in plain language (avoid jargon, or briefly define it), be encouraging, and give actionable feedback. Here is their real-time pitch data:",
      "",
    ];

    if (currentPitch) {
      const sign = currentPitch.centsOff >= 0 ? "+" : "";
      lines.push(
        `Current pitch: ${currentPitch.noteLabel} (${currentPitch.frequency.toFixed(1)} Hz, ${sign}${currentPitch.centsOff}\u00A2 off, clarity: ${currentPitch.clarity.toFixed(2)})`,
        ""
      );
    } else {
      lines.push("Current pitch: No pitch detected", "");
    }

    const history = historyRef.current;
    if (history.length > 0) {
      const noteMap = new Map<
        string,
        { count: number; totalDeviation: number }
      >();
      for (const snap of history) {
        const existing = noteMap.get(snap.noteLabel);
        if (existing) {
          existing.count++;
          existing.totalDeviation += snap.centsOff;
        } else {
          noteMap.set(snap.noteLabel, {
            count: 1,
            totalDeviation: snap.centsOff,
          });
        }
      }

      lines.push("Recent history (last 30s):");
      const sorted = [...noteMap.entries()].sort(
        (a, b) => b[1].count - a[1].count
      );
      for (const [note, { count, totalDeviation }] of sorted) {
        const avgDev = Math.round(totalDeviation / count);
        const sign = avgDev >= 0 ? "+" : "";
        lines.push(
          `- ${note}: ${count} detections, avg deviation: ${sign}${avgDev}\u00A2`
        );
      }
      lines.push("");
    }

    lines.push(
      "Use this data to give warm, encouraging feedback. Focus on what will help them improve: explain pitch and tuning in simple terms, celebrate what they are doing well, and suggest one or two concrete next steps. Keep language clear and avoid jargon.",
      "",
      "You have two tools available:",
      "- show_tuning_modal: Call this when the user asks about their tuning, pitch, or intonation.",
      "- show_style_modal: Call this when the user asks about their singing style, vocal tone, what artist they sound like, or artists with similar voices.",
      "Always call the relevant tool when the user's question matches. You may also include a short text response alongside the tool call."
    );

    return lines.join("\n");
  }, [currentPitch]);

  return { currentPitch, formatContext, getHistorySummary };
}

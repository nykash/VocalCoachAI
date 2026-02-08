/** Detailed instructions per exercise: what it is, why do it, how to do it. Shared by StyleModal (UI) and useChat (reprompt content). */
export const EXERCISE_INSTRUCTIONS: Record<string, string> = {
  "Humming scales": `What: Humming scales means singing a scale (do-re-mi-fa-sol-la-ti-do) with your lips closed on an "m" sound — no words, just a steady hum.

Why: Humming warms up your voice gently, builds breath control, and helps you feel resonance in your face and nose without straining. It's one of the safest ways to start a practice.

How: (1) Relax your jaw and keep your lips lightly closed. (2) Breathe in through your nose, then hum a comfortable note. (3) Go up the major scale one note at a time, then back down. (4) Keep the volume even and the breath steady. Stay in a range that feels easy.

We'll analyze your tone and pitch from the recording.`,

  "Scales on ah": `What: "Scales on ah" means singing a major scale (do-re-mi-fa-sol-la-ti-do) on the vowel "ah" (as in "father") — one vowel for the whole scale.

Why: This builds even tone across your range, strengthens breath support, and helps you keep the same vocal quality on every note. It's a core exercise for consistency and intonation.

How: (1) Take a good breath and support from your belly. (2) Sing the first note on "ah" and hold it steady. (3) Move up the scale, keeping the "ah" clear and the volume even. (4) Then come back down. (5) Stay in a comfortable range; avoid pushing high or low.

We'll evaluate your vocal style and pitch accuracy.`,

  "Scales on zzz": `What: Singing a scale on a sustained "zzz" (like a bee buzz) — you keep the buzz going while moving up and down the scale.

Why: The buzz encourages forward resonance and helps you feel where the sound should sit. It's great for reducing tension and finding a clear, focused tone.

How: (1) Place your tongue behind your teeth and make a steady "zzz" sound. (2) Keep the buzz consistent as you sing one note, then move to the next. (3) Go up the major scale and then back down. (4) Don't squeeze; let the buzz flow on your breath.

We'll analyze your style and pitch from the recording.`,
};

export const DEFAULT_INSTRUCTION =
  "Sing clearly in a quiet space so we can hear your tone and pitch. Use a comfortable range and steady breath. We'll analyze your vocal style and intonation from the recording.";

/** Instructions text to include in the model reprompt when we have a style result (so the model can reference what the user was asked to do). */
export function getExerciseInstructionsForPrompt(): string {
  const lines = [
    "Exercise instructions (for reference when replying):",
    ...Object.entries(EXERCISE_INSTRUCTIONS).map(
      ([name, text]) => `--- ${name} ---\n${text}`
    ),
  ];
  return lines.join("\n\n");
}

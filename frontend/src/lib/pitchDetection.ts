export interface PitchResult {
  frequency: number;
  clarity: number;
  noteName: string;
  octave: number;
  noteLabel: string;
  /** Same note in flat notation (e.g. "Gb4" for F#4) */
  noteLabelFlat: string;
  centsOff: number;
}

const NOTE_NAMES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
const NOTE_NAMES_FLAT = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"];

export function frequencyToNote(frequency: number): {
  noteName: string;
  octave: number;
  noteLabel: string;
  noteLabelFlat: string;
  centsOff: number;
} {
  const semitonesFromA4 = 12 * Math.log2(frequency / 440);
  const roundedSemitones = Math.round(semitonesFromA4);
  const centsOff = Math.round((semitonesFromA4 - roundedSemitones) * 100);

  // A4 is MIDI note 69
  const midiNote = 69 + roundedSemitones;
  const octave = Math.floor(midiNote / 12) - 1;
  const noteIndex = ((midiNote % 12) + 12) % 12;
  const noteName = NOTE_NAMES_SHARP[noteIndex];
  const noteNameFlat = NOTE_NAMES_FLAT[noteIndex];

  return {
    noteName,
    octave,
    noteLabel: `${noteName}${octave}`,
    noteLabelFlat: `${noteNameFlat}${octave}`,
    centsOff,
  };
}

export function detectPitch(
  buffer: Float32Array,
  sampleRate: number,
): PitchResult | null {
  const n = buffer.length;
  // Normalize audio
  let maxAmp = 0;
  for (let i = 0; i < n; i++) {
    maxAmp = Math.max(maxAmp, Math.abs(buffer[i]));
  }
  if (maxAmp < 1e-10) return null;

  const normalized = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    normalized[i] = buffer[i] / maxAmp;
  }

  // RMS gate — skip if signal is too quiet
  let rms = 0;
  for (let i = 0; i < n; i++) {
    rms += normalized[i] * normalized[i];
  }
  rms = Math.sqrt(rms / n);
  if (rms < 0.003) return null; // Even lower threshold for maximum sensitivity

  // Pre-emphasis filter to amplify high frequencies (helps with clarity)
  const emphasized = new Float32Array(n);
  emphasized[0] = normalized[0];
  for (let i = 1; i < n; i++) {
    emphasized[i] = normalized[i] - 0.97 * normalized[i - 1];
  }

  // Autocorrelation over lag range for 70–500 Hz (broader range for voices)
  const minLag = Math.floor(sampleRate / 500);
  const maxLag = Math.ceil(sampleRate / 70);

  // Normalized autocorrelation
  const correlations = new Float32Array(maxLag + 1);
  for (let lag = minLag; lag <= maxLag; lag++) {
    let sum = 0;
    let sumSq1 = 0;
    let sumSq2 = 0;
    for (let i = 0; i < n - lag; i++) {
      const s1 = emphasized[i];
      const s2 = emphasized[i + lag];
      sum += s1 * s2;
      sumSq1 += s1 * s1;
      sumSq2 += s2 * s2;
    }
    const denom = Math.sqrt(sumSq1 * sumSq2);
    correlations[lag] = denom > 0 ? sum / denom : 0;
  }

  // Find the best peak in the autocorrelation
  let bestLag = minLag;
  let bestCorr = correlations[minLag];
  for (let lag = minLag + 1; lag <= maxLag; lag++) {
    if (correlations[lag] > bestCorr) {
      bestCorr = correlations[lag];
      bestLag = lag;
    }
  }

  // Lowered clarity threshold for breathy/softer voices - be very permissive
  if (bestCorr < 0.4) return null;

  // Parabolic interpolation for sub-sample accuracy
  let refinedLag = bestLag;
  if (bestLag > minLag && bestLag < maxLag) {
    const y0 = correlations[bestLag - 1];
    const y1 = correlations[bestLag];
    const y2 = correlations[bestLag + 1];
    const denom = y0 - 2 * y1 + y2;
    if (Math.abs(denom) > 1e-10) {
      const shift = (y0 - y2) / (2 * denom);
      if (Math.abs(shift) < 1) {
        refinedLag = bestLag + shift;
      }
    }
  }

  const frequency = sampleRate / refinedLag;
  
  // Validate frequency is in reasonable human voice range
  if (frequency < 70 || frequency > 500) return null;

  const note = frequencyToNote(frequency);

  return {
    frequency,
    clarity: bestCorr,
    ...note,
  };
}

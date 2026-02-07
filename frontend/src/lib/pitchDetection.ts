export interface PitchResult {
  frequency: number;
  clarity: number;
  noteName: string;
  octave: number;
  noteLabel: string;
  centsOff: number;
}

const NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];

export function frequencyToNote(frequency: number): {
  noteName: string;
  octave: number;
  noteLabel: string;
  centsOff: number;
} {
  const semitonesFromA4 = 12 * Math.log2(frequency / 440);
  const roundedSemitones = Math.round(semitonesFromA4);
  const centsOff = Math.round((semitonesFromA4 - roundedSemitones) * 100);

  // A4 is MIDI note 69
  const midiNote = 69 + roundedSemitones;
  const octave = Math.floor(midiNote / 12) - 1;
  const noteIndex = ((midiNote % 12) + 12) % 12;
  const noteName = NOTE_NAMES[noteIndex];

  return {
    noteName,
    octave,
    noteLabel: `${noteName}${octave}`,
    centsOff,
  };
}

export function detectPitch(
  buffer: Float32Array,
  sampleRate: number,
): PitchResult | null {
  const n = buffer.length;

  // RMS gate — skip if signal is too quiet
  let rms = 0;
  for (let i = 0; i < n; i++) {
    rms += buffer[i] * buffer[i];
  }
  rms = Math.sqrt(rms / n);
  if (rms < 0.01) return null;

  // Autocorrelation over lag range for 80–1100 Hz
  const minLag = Math.floor(sampleRate / 1100);
  const maxLag = Math.ceil(sampleRate / 80);

  // Normalized autocorrelation
  const correlations = new Float32Array(maxLag + 1);
  for (let lag = minLag; lag <= maxLag; lag++) {
    let sum = 0;
    let sumSq1 = 0;
    let sumSq2 = 0;
    for (let i = 0; i < n - lag; i++) {
      sum += buffer[i] * buffer[i + lag];
      sumSq1 += buffer[i] * buffer[i];
      sumSq2 += buffer[i + lag] * buffer[i + lag];
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

  // Clarity gate — reject noisy / non-tonal signals
  if (bestCorr < 0.8) return null;

  // Parabolic interpolation for sub-sample accuracy
  let refinedLag = bestLag;
  if (bestLag > minLag && bestLag < maxLag) {
    const y0 = correlations[bestLag - 1];
    const y1 = correlations[bestLag];
    const y2 = correlations[bestLag + 1];
    const shift = (y0 - y2) / (2 * (y0 - 2 * y1 + y2));
    if (Math.abs(shift) < 1) {
      refinedLag = bestLag + shift;
    }
  }

  const frequency = sampleRate / refinedLag;
  const note = frequencyToNote(frequency);

  return {
    frequency,
    clarity: bestCorr,
    ...note,
  };
}

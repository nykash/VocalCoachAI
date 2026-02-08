/** Build a WAV Blob from mono Float32 samples (range -1..1). */
export function float32ToWavBlob(samples: Float32Array, sampleRate: number): Blob {
  const numChannels = 1;
  const bitsPerSample = 16;
  const bytesPerSample = bitsPerSample / 8;
  const blockAlign = numChannels * bytesPerSample;
  const dataSize = samples.length * blockAlign;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);
  let offset = 0;
  const write = (value: number, bytes: number, littleEndian = true) => {
    if (bytes === 2) view.setInt16(offset, value, littleEndian);
    else if (bytes === 4) view.setInt32(offset, value, littleEndian);
    offset += bytes;
  };
  const writeStr = (str: string) => {
    for (let i = 0; i < str.length; i++) view.setUint8(offset++, str.charCodeAt(i));
  };
  writeStr("RIFF");
  write(36 + dataSize, 4);
  writeStr("WAVE");
  writeStr("fmt ");
  write(16, 4);
  write(1, 2);
  write(numChannels, 2);
  write(sampleRate, 4);
  write(sampleRate * blockAlign, 4);
  write(blockAlign, 2);
  write(bitsPerSample, 2);
  writeStr("data");
  write(dataSize, 4);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    offset += 2;
  }
  return new Blob([buffer], { type: "audio/wav" });
}

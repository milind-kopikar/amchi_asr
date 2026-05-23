/**
 * WAV encoder for the live-recording demo.
 *
 * The browser's MediaRecorder produces WebM/Opus on Chrome and m4a/AAC on
 * Safari. The RunPod ASR handler expects 16 kHz mono PCM WAV. This module
 * decodes the recorded blob via AudioContext, downsamples to 16 kHz,
 * mixes down to mono, writes a standards-compliant WAV header, and
 * returns a base64 string suitable for sending to /api/transcribe.
 *
 * The module is split into:
 *   - pure helpers (no DOM access — directly unit-testable)
 *   - browser-only helpers that use AudioContext (must run in the browser)
 *
 * @module wav-encoder
 */

/** Target sample rate the RunPod handler expects. */
export const TARGET_SAMPLE_RATE = 16_000;

// ---------------------------------------------------------------------------
// Pure helpers — fully testable in Node
// ---------------------------------------------------------------------------

/**
 * Build the 44-byte PCM WAV header.
 *
 * @param sampleRate    Sample rate in Hz (must be > 0).
 * @param numSamples    Number of PCM samples (per channel).
 * @param numChannels   Channel count. 1 = mono. (The encoder always
 *                       produces mono, but the helper is general.)
 * @returns A 44-byte ArrayBuffer containing the WAV header.
 * @throws RangeError when any parameter is invalid.
 */
export function buildWavHeader(
  sampleRate: number,
  numSamples: number,
  numChannels: number,
): ArrayBuffer {
  if (!Number.isInteger(sampleRate) || sampleRate <= 0) {
    throw new RangeError(`sampleRate must be a positive integer, got ${sampleRate}`);
  }
  if (!Number.isInteger(numSamples) || numSamples < 0) {
    throw new RangeError(`numSamples must be a non-negative integer, got ${numSamples}`);
  }
  if (!Number.isInteger(numChannels) || numChannels <= 0) {
    throw new RangeError(`numChannels must be a positive integer, got ${numChannels}`);
  }

  const bytesPerSample = 2; // 16-bit
  const blockAlign = numChannels * bytesPerSample;
  const byteRate = sampleRate * blockAlign;
  const dataSize = numSamples * blockAlign;

  const buffer = new ArrayBuffer(44);
  const view = new DataView(buffer);

  // RIFF header
  writeAscii(view, 0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeAscii(view, 8, "WAVE");

  // fmt chunk
  writeAscii(view, 12, "fmt ");
  view.setUint32(16, 16, true);           // chunk size for PCM
  view.setUint16(20, 1, true);            // PCM
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 16, true);           // bits per sample

  // data chunk
  writeAscii(view, 36, "data");
  view.setUint32(40, dataSize, true);

  return buffer;
}

/**
 * Convert Float32 PCM samples in the range [-1, 1] to signed 16-bit PCM.
 * Values outside the range are clamped (not wrapped).
 *
 * @param samples Float32Array of input samples.
 * @returns Int16Array of clamped, scaled samples.
 */
export function floatTo16BitPcm(samples: Float32Array): Int16Array {
  const out = new Int16Array(samples.length);
  for (let i = 0; i < samples.length; i++) {
    let s = samples[i];
    if (s > 1) s = 1;
    else if (s < -1) s = -1;
    out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return out;
}

/**
 * Downsample a Float32 buffer to a lower sample rate using simple averaging.
 * This is good enough for speech (no anti-aliasing filter), and is well-suited
 * for the 48 kHz → 16 kHz step common in browsers.
 *
 * @param samples       Source samples.
 * @param sourceRate    Source sample rate in Hz.
 * @param targetRate    Target sample rate in Hz. Must be <= sourceRate.
 * @returns Float32Array at targetRate.
 * @throws RangeError when targetRate > sourceRate or rates are non-positive.
 */
export function downsample(
  samples: Float32Array,
  sourceRate: number,
  targetRate: number,
): Float32Array {
  if (sourceRate <= 0 || targetRate <= 0) {
    throw new RangeError("sample rates must be positive");
  }
  if (targetRate > sourceRate) {
    throw new RangeError(
      `targetRate (${targetRate}) must be <= sourceRate (${sourceRate})`,
    );
  }
  if (targetRate === sourceRate) return samples;

  const ratio = sourceRate / targetRate;
  const newLength = Math.floor(samples.length / ratio);
  const out = new Float32Array(newLength);

  for (let i = 0; i < newLength; i++) {
    const start = Math.floor(i * ratio);
    const end = Math.floor((i + 1) * ratio);
    let sum = 0;
    let count = 0;
    for (let j = start; j < end && j < samples.length; j++) {
      sum += samples[j];
      count++;
    }
    out[i] = count > 0 ? sum / count : 0;
  }
  return out;
}

/**
 * Mix down multi-channel Float32 PCM to mono by averaging across channels.
 *
 * @param channels Array of per-channel Float32Arrays. All must be same length.
 * @returns Float32Array of the same length, holding the mono mixdown.
 *           Returns the input directly if there's only one channel.
 * @throws RangeError when channels is empty or channels have mismatched lengths.
 */
export function mixToMono(channels: Float32Array[]): Float32Array {
  if (channels.length === 0) {
    throw new RangeError("channels array must not be empty");
  }
  if (channels.length === 1) return channels[0];

  const length = channels[0].length;
  for (const ch of channels) {
    if (ch.length !== length) {
      throw new RangeError(
        `Channel length mismatch: ${ch.length} vs ${length}`,
      );
    }
  }

  const mono = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    let sum = 0;
    for (const ch of channels) {
      sum += ch[i];
    }
    mono[i] = sum / channels.length;
  }
  return mono;
}

/**
 * Assemble a complete WAV file from Float32 mono samples + target sample rate.
 *
 * @param samples       Float32Array of mono samples at ``sampleRate``.
 * @param sampleRate    Sample rate in Hz.
 * @returns Uint8Array containing the full WAV file (header + data).
 */
export function buildWav(samples: Float32Array, sampleRate: number): Uint8Array {
  const pcm16 = floatTo16BitPcm(samples);
  const header = buildWavHeader(sampleRate, samples.length, 1);
  const out = new Uint8Array(header.byteLength + pcm16.byteLength);
  out.set(new Uint8Array(header), 0);
  out.set(new Uint8Array(pcm16.buffer), header.byteLength);
  return out;
}

/**
 * Base64-encode a Uint8Array. Works in both Node and the browser.
 *
 * @param bytes Bytes to encode.
 * @returns ASCII base64 string with no line breaks.
 */
export function bytesToBase64(bytes: Uint8Array): string {
  // Browser path
  if (typeof btoa === "function") {
    let binary = "";
    const chunkSize = 0x8000; // avoid call-stack overflow on large arrays
    for (let i = 0; i < bytes.length; i += chunkSize) {
      const slice = bytes.subarray(i, i + chunkSize);
      binary += String.fromCharCode(...slice);
    }
    return btoa(binary);
  }
  // Node path (used by unit tests)
  // Buffer is global in Node; we cast to keep TypeScript happy.
  const NodeBuffer = (globalThis as unknown as { Buffer: { from(b: Uint8Array): { toString(enc: string): string } } }).Buffer;
  return NodeBuffer.from(bytes).toString("base64");
}

// ---------------------------------------------------------------------------
// Browser-only helpers (use AudioContext)
// ---------------------------------------------------------------------------

/**
 * Decode a recorded Blob (WebM/Opus from Chrome, m4a from Safari, etc.) into
 * a WAV at 16 kHz mono, base64-encoded, ready to POST to /api/transcribe.
 *
 * Must be called from the browser (uses AudioContext).
 *
 * @param blob         Recording from MediaRecorder.
 * @param targetRate   Target sample rate (default ``TARGET_SAMPLE_RATE``).
 * @returns Base64 string of the WAV bytes.
 * @throws Error if AudioContext is unavailable or decoding fails.
 */
export async function audioBlobToBase64Wav(
  blob: Blob,
  targetRate: number = TARGET_SAMPLE_RATE,
): Promise<string> {
  const AudioContextCtor =
    typeof window !== "undefined" &&
    (window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext);
  if (!AudioContextCtor) {
    throw new Error("AudioContext is not available in this environment");
  }

  const audioCtx = new AudioContextCtor();
  try {
    const arrayBuffer = await blob.arrayBuffer();
    const decoded: AudioBuffer = await audioCtx.decodeAudioData(arrayBuffer);

    // Pull each channel as Float32Array
    const channels: Float32Array[] = [];
    for (let c = 0; c < decoded.numberOfChannels; c++) {
      channels.push(decoded.getChannelData(c));
    }

    const mono = mixToMono(channels);
    const resampled =
      decoded.sampleRate === targetRate
        ? mono
        : downsample(mono, decoded.sampleRate, targetRate);
    const wavBytes = buildWav(resampled, targetRate);
    return bytesToBase64(wavBytes);
  } finally {
    await audioCtx.close();
  }
}

// ---------------------------------------------------------------------------
// Internal helpers (not exported, but kept simple for clarity)
// ---------------------------------------------------------------------------

function writeAscii(view: DataView, offset: number, text: string): void {
  for (let i = 0; i < text.length; i++) {
    view.setUint8(offset + i, text.charCodeAt(i));
  }
}

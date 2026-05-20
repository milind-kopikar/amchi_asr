/**
 * Unit tests for the pure helpers in ``wav-encoder.ts``.
 *
 * Only the pure helpers (no AudioContext) are covered here — they run
 * directly in Node under vitest. The ``audioBlobToBase64Wav`` integration
 * function needs a browser/jsdom environment and is intentionally not
 * tested here; smoke-test it in the demo webapp instead.
 *
 * Run with:
 *   npm install   (one-time)
 *   npm test
 *
 * @module wav-encoder.test
 */

import { describe, it, expect } from "vitest";
import {
  buildWav,
  buildWavHeader,
  bytesToBase64,
  downsample,
  floatTo16BitPcm,
  mixToMono,
  TARGET_SAMPLE_RATE,
} from "./wav-encoder";

// ---------------------------------------------------------------------------
// buildWavHeader
// ---------------------------------------------------------------------------

describe("buildWavHeader", () => {
  it("typical: produces a 44-byte RIFF/WAVE/fmt /data header", () => {
    const header = buildWavHeader(16000, 16000, 1);
    expect(header.byteLength).toBe(44);

    const view = new DataView(header);
    // RIFF magic
    expect(String.fromCharCode(view.getUint8(0), view.getUint8(1),
                               view.getUint8(2), view.getUint8(3))).toBe("RIFF");
    expect(String.fromCharCode(view.getUint8(8), view.getUint8(9),
                               view.getUint8(10), view.getUint8(11))).toBe("WAVE");
    // sample rate at offset 24
    expect(view.getUint32(24, true)).toBe(16000);
    // channels at offset 22
    expect(view.getUint16(22, true)).toBe(1);
    // bits/sample at offset 34
    expect(view.getUint16(34, true)).toBe(16);
    // data chunk size = 16000 samples * 2 bytes
    expect(view.getUint32(40, true)).toBe(32000);
  });

  it("typical: stereo at 44.1 kHz", () => {
    const header = buildWavHeader(44100, 1000, 2);
    const view = new DataView(header);
    expect(view.getUint32(24, true)).toBe(44100);
    expect(view.getUint16(22, true)).toBe(2);
    // block align = channels * bytes/sample = 2 * 2 = 4
    expect(view.getUint16(32, true)).toBe(4);
    // byte rate = sampleRate * blockAlign = 44100 * 4 = 176400
    expect(view.getUint32(28, true)).toBe(176400);
  });

  it("zero samples is allowed (empty data chunk)", () => {
    const header = buildWavHeader(16000, 0, 1);
    const view = new DataView(header);
    expect(view.getUint32(40, true)).toBe(0); // data size = 0
  });

  it("rejects non-positive sample rate", () => {
    expect(() => buildWavHeader(0, 100, 1)).toThrow(RangeError);
    expect(() => buildWavHeader(-1, 100, 1)).toThrow(RangeError);
  });

  it("rejects non-integer sample rate", () => {
    expect(() => buildWavHeader(16000.5, 100, 1)).toThrow(RangeError);
  });

  it("rejects negative samples", () => {
    expect(() => buildWavHeader(16000, -1, 1)).toThrow(RangeError);
  });

  it("rejects zero channels", () => {
    expect(() => buildWavHeader(16000, 100, 0)).toThrow(RangeError);
  });
});

// ---------------------------------------------------------------------------
// floatTo16BitPcm
// ---------------------------------------------------------------------------

describe("floatTo16BitPcm", () => {
  it("typical: scales [-1, 1] floats to int16 range", () => {
    const samples = new Float32Array([0, 0.5, -0.5, 1.0, -1.0]);
    const pcm = floatTo16BitPcm(samples);
    expect(pcm[0]).toBe(0);
    expect(pcm[1]).toBe(Math.floor(0.5 * 0x7fff));
    expect(pcm[2]).toBe(Math.floor(-0.5 * 0x8000));
    expect(pcm[3]).toBe(0x7fff);
    expect(pcm[4]).toBe(-0x8000);
  });

  it("clamps values above 1 and below -1", () => {
    const samples = new Float32Array([5.0, -5.0]);
    const pcm = floatTo16BitPcm(samples);
    expect(pcm[0]).toBe(0x7fff);
    expect(pcm[1]).toBe(-0x8000);
  });

  it("empty input returns empty Int16Array", () => {
    const pcm = floatTo16BitPcm(new Float32Array(0));
    expect(pcm.length).toBe(0);
  });

  it("all-zero input returns all-zero output", () => {
    const pcm = floatTo16BitPcm(new Float32Array(10));
    expect(Array.from(pcm)).toEqual([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
  });
});

// ---------------------------------------------------------------------------
// downsample
// ---------------------------------------------------------------------------

describe("downsample", () => {
  it("typical 48 kHz → 16 kHz (3:1)", () => {
    const samples = new Float32Array(48); // 1 ms of data
    for (let i = 0; i < 48; i++) samples[i] = i;
    const out = downsample(samples, 48000, 16000);
    expect(out.length).toBe(16);
    // first output sample = average of inputs 0,1,2 = 1.0
    expect(out[0]).toBeCloseTo(1.0);
  });

  it("source==target returns the input unchanged", () => {
    const samples = new Float32Array([1, 2, 3]);
    const out = downsample(samples, 16000, 16000);
    expect(out).toBe(samples); // same reference
  });

  it("rejects target rate > source rate", () => {
    const samples = new Float32Array(10);
    expect(() => downsample(samples, 16000, 48000)).toThrow(RangeError);
  });

  it("rejects non-positive rates", () => {
    const samples = new Float32Array(10);
    expect(() => downsample(samples, 0, 16000)).toThrow(RangeError);
    expect(() => downsample(samples, 16000, 0)).toThrow(RangeError);
  });

  it("empty input produces empty output", () => {
    const out = downsample(new Float32Array(0), 48000, 16000);
    expect(out.length).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// mixToMono
// ---------------------------------------------------------------------------

describe("mixToMono", () => {
  it("typical: averages two channels", () => {
    const left = new Float32Array([1, 0, -1]);
    const right = new Float32Array([0, 1, 0]);
    const mono = mixToMono([left, right]);
    expect(Array.from(mono)).toEqual([0.5, 0.5, -0.5]);
  });

  it("single channel returns the same reference", () => {
    const ch = new Float32Array([1, 2, 3]);
    const mono = mixToMono([ch]);
    expect(mono).toBe(ch);
  });

  it("rejects empty channel list", () => {
    expect(() => mixToMono([])).toThrow(RangeError);
  });

  it("rejects mismatched channel lengths", () => {
    const a = new Float32Array(10);
    const b = new Float32Array(20);
    expect(() => mixToMono([a, b])).toThrow(RangeError);
  });

  it("averages three channels correctly", () => {
    const a = new Float32Array([3]);
    const b = new Float32Array([6]);
    const c = new Float32Array([9]);
    const mono = mixToMono([a, b, c]);
    expect(mono[0]).toBeCloseTo(6);
  });
});

// ---------------------------------------------------------------------------
// buildWav
// ---------------------------------------------------------------------------

describe("buildWav", () => {
  it("typical: produces header + data bytes with correct length", () => {
    const samples = new Float32Array(16000); // 1 second at 16 kHz
    const wav = buildWav(samples, 16000);
    // 44-byte header + 16000 samples * 2 bytes
    expect(wav.length).toBe(44 + 32000);
    // Magic bytes
    expect(String.fromCharCode(wav[0], wav[1], wav[2], wav[3])).toBe("RIFF");
  });

  it("data section size matches header data-chunk-size field", () => {
    const samples = new Float32Array(100);
    const wav = buildWav(samples, 16000);
    const view = new DataView(wav.buffer, wav.byteOffset, wav.byteLength);
    const dataSize = view.getUint32(40, true);
    expect(dataSize).toBe(wav.length - 44);
  });

  it("empty samples produce a valid 44-byte WAV", () => {
    const wav = buildWav(new Float32Array(0), 16000);
    expect(wav.length).toBe(44);
  });
});

// ---------------------------------------------------------------------------
// bytesToBase64
// ---------------------------------------------------------------------------

describe("bytesToBase64", () => {
  it("typical round-trip", () => {
    const bytes = new Uint8Array([0, 1, 2, 254, 255]);
    const b64 = bytesToBase64(bytes);
    // 5 bytes → 8 base64 chars
    expect(b64.length).toBe(8);
    // Decode back to compare
    const NodeBuffer = (globalThis as unknown as {
      Buffer: { from(s: string, enc: string): { length: number; [i: number]: number };
              };
    }).Buffer;
    const decoded = NodeBuffer.from(b64, "base64");
    expect(decoded[0]).toBe(0);
    expect(decoded[4]).toBe(255);
  });

  it("empty input returns empty string", () => {
    expect(bytesToBase64(new Uint8Array(0))).toBe("");
  });
});

// ---------------------------------------------------------------------------
// TARGET_SAMPLE_RATE constant
// ---------------------------------------------------------------------------

describe("TARGET_SAMPLE_RATE", () => {
  it("is 16 kHz to match the NeMo handler", () => {
    expect(TARGET_SAMPLE_RATE).toBe(16000);
  });
});

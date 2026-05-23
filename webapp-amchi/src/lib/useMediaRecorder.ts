/**
 * useMediaRecorder — React hook that wraps the browser MediaRecorder API.
 *
 * Returns a small state machine (idle / requesting / recording / stopping /
 * ready / error) plus ``start``/``stop`` async functions. The collected
 * audio is exposed as a Blob (browser-native format — WebM/Opus on Chrome,
 * m4a/AAC on Safari). Convert to base64 WAV with
 * ``audioBlobToBase64Wav(blob)`` from ``wav-encoder.ts`` before POSTing to
 * /api/transcribe.
 *
 * Usage:
 * ```tsx
 *   const recorder = useMediaRecorder();
 *   <button onClick={recorder.start}>Record</button>
 *   <button onClick={recorder.stop}>Stop</button>
 *   {recorder.status === "ready" && recorder.audioBlob && <audio src={URL.createObjectURL(recorder.audioBlob)} controls />}
 * ```
 *
 * NOTE: this hook intentionally has no unit tests at the hook level — testing
 * the React hook requires jsdom + a MediaRecorder mock and tests very little.
 * The PURE helpers it depends on (in ``wav-encoder.ts``) ARE unit-tested. To
 * smoke-test this hook, run the demo page in a real browser.
 *
 * @module useMediaRecorder
 */

"use client";

import { useCallback, useEffect, useRef, useState } from "react";

export type RecorderStatus =
  | "idle"
  | "requesting"
  | "recording"
  | "stopping"
  | "ready"
  | "error";

export interface UseMediaRecorderResult {
  /** Current state-machine position. See ``RecorderStatus``. */
  status: RecorderStatus;
  /** The recorded audio blob (browser-native format). Set when status === "ready". */
  audioBlob: Blob | null;
  /** Last error message, populated when status === "error". */
  error: string | null;
  /** Start recording. Resolves once the recorder is actively capturing. */
  start: () => Promise<void>;
  /** Stop recording. Resolves once the blob is available. */
  stop: () => Promise<Blob | null>;
  /** Reset to idle, clearing any blob and error. */
  reset: () => void;
}

export function useMediaRecorder(): UseMediaRecorderResult {
  const [status, setStatus] = useState<RecorderStatus>("idle");
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [error, setError] = useState<string | null>(null);

  const recorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<BlobPart[]>([]);
  const stopResolveRef = useRef<((blob: Blob | null) => void) | null>(null);

  /** Release the underlying microphone stream. */
  const releaseStream = useCallback(() => {
    if (streamRef.current) {
      for (const track of streamRef.current.getTracks()) {
        track.stop();
      }
      streamRef.current = null;
    }
    recorderRef.current = null;
  }, []);

  // Tidy up if the component unmounts mid-recording
  useEffect(() => {
    return () => releaseStream();
  }, [releaseStream]);

  const start = useCallback(async () => {
    setError(null);
    setAudioBlob(null);
    setStatus("requesting");

    if (typeof navigator === "undefined" || !navigator.mediaDevices?.getUserMedia) {
      setError("Microphone is not available in this environment.");
      setStatus("error");
      return;
    }
    if (typeof window === "undefined" || typeof window.MediaRecorder === "undefined") {
      setError("MediaRecorder API is not available in this browser.");
      setStatus("error");
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
        },
      });
      streamRef.current = stream;
      chunksRef.current = [];

      const recorder = new MediaRecorder(stream);
      recorderRef.current = recorder;

      recorder.ondataavailable = (ev: BlobEvent) => {
        if (ev.data && ev.data.size > 0) {
          chunksRef.current.push(ev.data);
        }
      };

      recorder.onstop = () => {
        const mime = recorder.mimeType || "audio/webm";
        const blob = new Blob(chunksRef.current, { type: mime });
        chunksRef.current = [];
        setAudioBlob(blob);
        setStatus("ready");
        releaseStream();
        if (stopResolveRef.current) {
          stopResolveRef.current(blob);
          stopResolveRef.current = null;
        }
      };

      recorder.onerror = (ev: Event) => {
        const message = (ev as unknown as { error?: { message?: string } }).error?.message
          || "MediaRecorder error";
        setError(message);
        setStatus("error");
        releaseStream();
        if (stopResolveRef.current) {
          stopResolveRef.current(null);
          stopResolveRef.current = null;
        }
      };

      recorder.start();
      setStatus("recording");
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setError(`Failed to start recording: ${message}`);
      setStatus("error");
      releaseStream();
    }
  }, [releaseStream]);

  const stop = useCallback((): Promise<Blob | null> => {
    return new Promise((resolve) => {
      const recorder = recorderRef.current;
      if (!recorder || recorder.state === "inactive") {
        resolve(audioBlob);
        return;
      }
      stopResolveRef.current = resolve;
      setStatus("stopping");
      try {
        recorder.stop();
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        setError(`Failed to stop recording: ${message}`);
        setStatus("error");
        releaseStream();
        resolve(null);
      }
    });
  }, [audioBlob, releaseStream]);

  const reset = useCallback(() => {
    releaseStream();
    chunksRef.current = [];
    setAudioBlob(null);
    setError(null);
    setStatus("idle");
  }, [releaseStream]);

  return { status, audioBlob, error, start, stop, reset };
}

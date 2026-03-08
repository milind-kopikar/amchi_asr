/**
 * Deaf Speech ASR Experiment Configurations
 *
 * This file contains metadata for all deaf speech ASR experiments.
 * Per-sample results are loaded dynamically from JSON files in public/data/.
 *
 * Experiment naming convention:
 *   DS-C  Baseline — full fine-tune, 1 speaker, 124 samples, 50 epochs
 *   DS-A  Frozen encoder — only decoder trained, 1 speaker, 100 epochs
 *   DS-B  Extended data — full fine-tune, multiple speakers, 188 samples, 100 epochs
 *   DS-D  Speed perturbation — full fine-tune, 1 speaker, 372 augmented samples, 100 epochs
 *
 * @module deaf-speech-experiments
 */

import type { DeafSpeechExperiment } from "./deaf-speech-types";

/**
 * DS-C: Baseline
 * Full fine-tune, single speaker, 124 samples, 50 epochs.
 * This is the reference point — no encoder freezing, no augmentation.
 */
const DS_C: DeafSpeechExperiment = {
  id: "dsc",
  name: "DS-C: Baseline",
  date: "2026-03-01",
  config: "Full fine-tune (115M params), 1 speaker, 124 samples, 50 epochs",
  samples: 124,
  wer: 0.753,
  bestEpoch: 21,
  valWer: 0.720,
  checkpoint:
    "nemo_experiments/deaf_speech_story4_50epoch/checkpoints/konkani_asr-epoch=21-val_wer=0.720.ckpt",
  notes:
    "Reference baseline: full fine-tune on one speaker's 124 recordings. " +
    "High WER reflects the difficulty of deaf speech recognition, but sets the comparison point.",
  per_sample: [],
};

/**
 * DS-A: Frozen Encoder
 * Only the CTC decoder head is trained (132K params).
 * Single speaker, 124 samples, 100 epochs.
 */
const DS_A: DeafSpeechExperiment = {
  id: "dsa",
  name: "DS-A: Frozen Encoder",
  date: "2026-03-07",
  config: "Frozen encoder (132K trainable params), 1 speaker, 124 samples, 100 epochs",
  samples: 124,
  wer: 0.796,
  bestEpoch: 96,
  valWer: 0.766,
  checkpoint: "results/experiments/deaf_speech_dsa/checkpoints/",
  notes:
    "Freezing the encoder hurt performance (-4.3pp vs DS-C). Unlike Amchi Konkani, " +
    "deaf speech requires full model adaptation — the acoustic patterns are too different " +
    "from standard Marathi for a frozen encoder to generalise.",
  per_sample: [],
};

/**
 * DS-B: Extended Data (Multiple Speakers)
 * Full fine-tune on expanded dataset: 124 original samples plus 64 recordings
 * from multiple additional speakers across other deaf speech stories.
 */
const DS_B: DeafSpeechExperiment = {
  id: "dsb",
  name: "DS-B: Extended Data (Multi-Speaker)",
  date: "2026-03-07",
  config: "Full fine-tune, multiple speakers, 188 samples (124 + 64 from other stories), 100 epochs",
  samples: 124,
  wer: 0.931,
  bestEpoch: 75,
  valWer: 0.850,
  checkpoint: "results/experiments/deaf_speech_dsb/checkpoints/",
  notes:
    "Adding 64 clips from multiple speakers across other stories dramatically hurt performance " +
    "(-17.8pp vs DS-C). The different speakers introduced out-of-distribution acoustic patterns. " +
    "For deaf speech, domain and speaker consistency matters far more than raw data volume.",
  per_sample: [],
};

/**
 * DS-D: Speed Perturbation — BEST RESULT
 * Full fine-tune on the same single speaker, augmented with 0.9x and 1.1x speed variants
 * to create 372 training samples (3x the original 124). All data in-domain.
 */
const DS_D: DeafSpeechExperiment = {
  id: "dsd",
  name: "DS-D: Speed Perturbation (BEST)",
  date: "2026-03-07",
  config:
    "Full fine-tune, 1 speaker, 372 samples (0.9x/1.0x/1.1x speed perturbation), 100 epochs",
  samples: 124,
  wer: 0.347,
  bestEpoch: 96,
  valWer: 0.269,
  checkpoint:
    "results/deaf_speech_dsd/checkpoints/konkani_asr-epoch=96-val_wer=0.269.ckpt",
  notes:
    "Speed perturbation on the same single speaker's 124 recordings creates 3x training data " +
    "with zero domain shift. Achieves 40.6pp improvement over DS-C. " +
    "Key insight: in-domain augmentation beats adding new speakers or out-of-domain data.",
  per_sample: [],
};

/**
 * All experiments keyed by ID
 */
export const DEAF_SPEECH_EXPERIMENTS: Record<string, DeafSpeechExperiment> = {
  dsc: DS_C,
  dsa: DS_A,
  dsb: DS_B,
  dsd: DS_D,
};

/**
 * Get experiment by ID, with fallback to DS-C baseline
 */
export function getExperiment(id: unknown): DeafSpeechExperiment {
  if (typeof id !== "string") return DS_C;
  const exp = DEAF_SPEECH_EXPERIMENTS[id];
  return exp ?? DS_C;
}

/**
 * Get all experiments sorted by test WER ascending (best first)
 */
export function getExperimentsByPerformance(): DeafSpeechExperiment[] {
  return Object.values(DEAF_SPEECH_EXPERIMENTS).sort((a, b) => a.wer - b.wer);
}

/**
 * Get path to the per-sample results JSON in public/data/
 */
export function getResultsPath(experimentId: string): string {
  const paths: Record<string, string> = {
    dsc: "/data/deaf_speech_results_dsc.json",
    dsa: "/data/deaf_speech_results_dsa.json",
    dsb: "/data/deaf_speech_results_dsb.json",
    dsd: "/data/deaf_speech_results_dsd.json",
  };
  return paths[experimentId] ?? paths.dsd;
}

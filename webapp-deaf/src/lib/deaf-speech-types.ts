/**
 * Type definitions for Deaf Speech ASR Experiment Results
 * 
 * These types represent the structure of deaf speech ASR experiments,
 * including raw predictions and optional post-processed outputs.
 * 
 * @module deaf-speech-types
 */

/**
 * A single sample from the test set with audio and transcription
 */
export interface DeafSpeechSample {
  audio: string;
  reference: string;
  prediction: string;
  wer: number;
  postprocessed?: {
    prediction: string;
    mode: "RECONSTRUCT" | "FILL" | "PASSTHROUGH";
    wer: number;
  };
}

/**
 * Summary statistics for an experiment
 */
export interface ExperimentSummary {
  total_samples: number;
  mean_wer: number;
}

/**
 * Raw results from the final_test_results.json file
 */
export interface RawExperimentResults {
  best_checkpoint: string;
  summary: ExperimentSummary;
  per_sample: DeafSpeechSample[];
}

/**
 * Processed experiment data ready for visualization
 */
export interface DeafSpeechExperiment {
  id: string;
  name: string;
  date: string;
  config: string;
  samples: number;
  wer: number;
  bestEpoch: number;
  valWer: number;
  checkpoint: string;
  notes: string;
  per_sample: DeafSpeechSample[];
}

/**
 * Computed statistics for error analysis
 */
export interface ErrorStatistics {
  improved: number;
  worsened: number;
  unchanged: number;
  avgImprovement: number;
  avgWorsening: number;
  successRate: number;
}

/**
 * WER distribution histogram
 */
export interface WERDistribution {
  ranges: Array<{
    label: string;
    min: number;
    max: number;
    count: number;
    percentage: number;
  }>;
}

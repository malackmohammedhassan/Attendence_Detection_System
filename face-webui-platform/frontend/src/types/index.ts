// ─────────────────────────────────────────────────────────────────────────────
//  Shared domain types for the ML Dashboard frontend
// ─────────────────────────────────────────────────────────────────────────────

// ── Model ────────────────────────────────────────────────────────────────────

export interface ModelInfo {
  name: string
  framework: string
  loaded: boolean
  load_time_sec: number
  last_used: string | null
  total_inferences: number
  total_errors: number
  registered_at: string
}

export interface ActivateModelResponse {
  message: string
  model_name: string
}

// ── Inference ────────────────────────────────────────────────────────────────

export interface BoundingBox {
  x1: number
  y1: number
  x2: number
  y2: number
  width: number
  height: number
  area: number
  confidence: number
  label: string
}

export interface InferenceResult {
  model_name: string
  detection_count: number
  detections: BoundingBox[]
  latency_ms: number
  frame_width: number
  frame_height: number
  confidence_threshold: number
  timestamp: number
}

// ── Training ─────────────────────────────────────────────────────────────────

export type TrainingStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'

export interface EpochResult {
  epoch: number
  train_loss: number
  val_loss: number
  train_acc: number
  val_acc: number
  lr: number
  duration_sec: number
}

export interface TrainingJob {
  job_id: string
  model_name: string
  status: TrainingStatus
  current_epoch: number
  total_epochs: number
  progress_pct: number
  best_val_loss: number | null
  best_epoch: number
  error_message: string | null
  started_at: string | null
  completed_at: string | null
  created_at: string
  epoch_history: EpochResult[]
}

export interface TrainingStartRequest {
  model_name: string
  epochs: number
  batch_size: number
  learning_rate: number
  optimizer: 'adam' | 'sgd' | 'adamw' | 'rmsprop'
  weight_decay: number
  early_stopping_patience: number
  checkpoint_every_n_epochs: number
}

// ── Metrics ──────────────────────────────────────────────────────────────────

export interface SystemSnapshot {
  timestamp: string
  cpu: {
    overall_percent: number
    per_core: number[]
    process_percent: number
  }
  memory: {
    used_mb: number
    total_mb: number
    percent: number
    swap_used_mb: number
    swap_total_mb: number
    process_rss_mb: number
  }
  inference: {
    fps: number
    total_frames: number
  }
  uptime_sec: number
}

export interface LatencyPercentiles {
  mean: number
  median: number
  stdev: number
  p50: number
  p90: number
  p95: number
  p99: number
  min: number
  max: number
}

export interface InferenceStats {
  model_name: string
  sample_count: number
  mean_latency_ms: number
  p50_latency_ms: number
  p95_latency_ms: number
  p99_latency_ms: number
  max_latency_ms: number
  min_latency_ms: number
  mean_confidence: number
  mean_detections: number
  fps_estimate: number
}

export interface InferenceRecord {
  model_name: string
  latency_ms: number
  confidence: number
  num_detections: number
  image_width: number
  image_height: number
  timestamp: string
}

export interface MetricsSummary {
  system: SystemSnapshot | null
  collector: {
    total_inferences: number
    error_count: number
    models_tracked: string[]
    training_runs: Record<string, number>
  }
  inference: Record<string, InferenceStats>
}

// ── Benchmark ────────────────────────────────────────────────────────────────

export type BenchmarkStatus = 'pending' | 'running' | 'completed' | 'failed'

export interface BenchmarkLatencyStats {
  model_name: string
  warmup_runs: number
  measure_runs: number
  latency_ms: LatencyPercentiles
  throughput_fps: number
}

export interface BenchmarkResult {
  run_id: string
  model_name: string
  status: BenchmarkStatus
  frame_size: string
  latency_stats: BenchmarkLatencyStats | null
  error_message: string | null
  duration_sec: number | null
  started_at: string | null
  completed_at: string | null
  created_at: string
  // Full evaluation fields
  frames_evaluated?: number
  progress_pct?: number
  precision?: number | null
  recall?: number | null
  f1?: number | null
  false_positives?: number
  true_positives?: number
  false_negatives?: number
  cpu_avg?: number | null
  memory_avg_mb?: number | null
  memory_baseline_mb?: number | null
  memory_peak_mb?: number | null
  memory_growth_mb?: number | null
  is_full_eval?: boolean
  iou_threshold?: number | null
  gt_source?: string | null
  celeba_coverage?: number | null
  avg_fps?: number | null
  // PR curve sweep (research-grade evaluation)
  pr_curve?: Array<{
    threshold: number
    precision: number
    recall: number
    f1: number
    tp: number
    fp: number
    fn: number
  }> | null
  auc_pr?: number | null
  best_f1_threshold?: number | null
  precision_at_recall_90?: number | null
  // Reproducibility
  model_sha256?: string | null
  dataset_adapter?: string | null
  eval_config?: Record<string, unknown> | null
  // Confidence distribution histogram
  confidence_histogram?: {
    bins: number[]
    counts: number[]
    n_bins?: number
    total_detections: number
  } | null
  // Calibration curve: confidence -> actual precision per bin
  calibration_curve?: Array<{
    bin_start: number
    bin_end: number
    bin_center: number
    mean_confidence: number
    actual_precision: number
    count: number
  }> | null
  // Experiment tagging
  run_tag?: string | null
  run_notes?: string | null
}

export interface BenchmarkComparison {
  models_compared: string[]
  frame_size: string
  results: Record<string, BenchmarkResult | { error: string }>
  ranking_by_fps: Array<{ rank: number; model: string; fps: number }>
}

export interface ComparativeModelStats {
  precision: number | null
  recall: number | null
  f1: number | null
  tp: number
  fp: number
  fn: number
  avg_fps: number | null
  auc_pr: number | null
}

export interface ComparativeFrameData {
  frame_idx: number
  gt_count: number
  a_n_dets: number
  b_n_dets: number
  a_tp: number; a_fp: number; a_fn: number
  b_tp: number; b_fp: number; b_fn: number
  a_f1: number
  b_f1: number
  f1_delta: number
  a_mean_conf: number
  b_mean_conf: number
  conf_delta: number
  agreement: 'both_detect' | 'both_miss' | 'a_only' | 'b_only'
}

export interface ComparativeResult {
  run_id: string
  model_names: string[]
  status: BenchmarkStatus
  frame_size: string
  frames_evaluated: number
  num_frames: number
  progress_pct: number
  started_at: string | null
  completed_at: string | null
  created_at: string
  duration_sec: number | null
  error_message: string | null
  run_tag?: string | null
  run_notes?: string | null
  model_stats?: Record<string, ComparativeModelStats> | null
  per_frame_data?: ComparativeFrameData[] | null
  disagreement_analysis?: {
    total_frames: number
    both_detect_rate: number
    both_miss_rate: number
    a_only_rate: number
    b_only_rate: number
    agreement_rate: number
    disagreement_rate: number
    mean_f1_delta: number
    mean_conf_delta: number
    model_a: string
    model_b: string
    dataset_hash?: string | null
    git_commit?: string | null
  } | null
  pr_curves?: Record<string, Array<{ threshold: number; precision: number; recall: number; f1: number }>> | null
  conf_shift_histogram?: {
    bins: number[]
    counts: number[]
    model_a: string
    model_b: string
  } | null
}

// ── WebSocket messages ───────────────────────────────────────────────────────

// ── Training logs ───────────────────────────────────────────────────────────

/** Single stdout/stderr line emitted by the training subprocess. */
export interface WsTrainLog {
  type: 'train_log'
  job_id: string
  line: string
  ts: number
}

/** Batch of existing log lines sent on WS connect. */
export interface WsTrainLogBatch {
  type: 'train_log_batch'
  job_id: string
  lines: string[]
}

// ── WebSocket messages ───────────────────────────────────────────────────────

export type WsMessageType =
  | 'heartbeat'
  | 'connected'
  | 'epoch'
  | 'status_change'
  | 'job_state'
  | 'job_complete'
  | 'train_log'
  | 'train_log_batch'
  | 'metrics_tick'
  | 'inference_result'
  | 'live_result'
  | 'no_model'
  | 'log'
  | 'log_replay'
  | 'ready'
  | 'error'
  | 'ack'
  | 'pong'

export interface WsHeartbeat {
  type: 'heartbeat'
  timestamp: string
  connections: number
}

export interface WsEpochMessage {
  type: 'epoch' | 'status_change' | 'job_state' | 'job_complete'
  job_id: string
  model_name: string
  status: TrainingStatus
  current_epoch: number
  total_epochs: number
  progress_pct: number
  epoch_result?: EpochResult
}

export interface WsMetricsTick {
  type: 'metrics_tick'
  system: SystemSnapshot | null
  // Optional extras sent only by /ws/live
  cpu?: number
  memory_mb?: number
  fps?: number
  timestamp?: number
}

export interface WsLogEntry {
  type: 'log' | 'log_replay'
  level: string
  logger: string
  message: string
  timestamp: string
  location: string
  thread: string
  exc_text: string | null
}

export interface WsInferenceResult {
  type: 'inference_result'
  model_name: string
  detection_count: number
  detections: BoundingBox[]
  latency_ms: number
  frame_width: number
  frame_height: number
  confidence_threshold: number
  timestamp: number
}

/**
 * Emitted by /ws/live for each processed frame.
 * Shape mirrors InferenceResult.as_dict() + raw_detection_count.
 */
export interface WsLiveResult {
  type: 'live_result'
  model_name: string
  detection_count: number
  raw_detection_count: number
  detections: BoundingBox[]
  latency_ms: number
  frame_width: number
  frame_height: number
  confidence_threshold: number
  timestamp: number
}

/**
 * Periodic system-metrics tick emitted by /ws/live every ~1 s.
 * WsLiveMetricsTick is a WsMetricsTick that guarantees the live-only fields.
 */
export type WsLiveMetricsTick = WsMetricsTick & {
  cpu: number
  memory_mb: number
  fps: number
  timestamp: number
}

export type WsMessage =
  | WsHeartbeat
  | WsEpochMessage
  | WsMetricsTick
  | WsLiveResult
  | WsLogEntry
  | WsInferenceResult
  | WsTrainLog
  | WsTrainLogBatch
  | { type: 'no_model' }
  | { type: 'error'; message: string }
  | { type: 'ready'; model: string; has_model?: boolean; client_id?: string }
  | { type: 'ack'; threshold?: number; interval_sec?: number }
  | { type: 'pong' }
  | { type: 'connected'; interval_sec: number }

// ── UI / Store helpers ───────────────────────────────────────────────────────

export interface InferenceSettings {
  confidenceThreshold: number
  stride: number
  scale: number
}

export interface ChartPoint {
  time: string
  value: number
}

export type ConnectionStatus = 'connected' | 'disconnected' | 'connecting' | 'error'

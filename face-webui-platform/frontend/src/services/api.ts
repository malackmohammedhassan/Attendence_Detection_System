import axios from 'axios'
import type { AxiosInstance, AxiosRequestConfig } from 'axios'
import type {
  ModelInfo,
  ActivateModelResponse,
  InferenceResult,
  TrainingJob,
  TrainingStartRequest,
  SystemSnapshot,
  InferenceStats,
  InferenceRecord,
  MetricsSummary,
  BenchmarkResult,
  BenchmarkComparison,
} from '@/types'

// ─────────────────────────────────────────────────────────────────────────────
//  Axios client factory
// ─────────────────────────────────────────────────────────────────────────────

const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? '/api/v1'

function createClient(): AxiosInstance {
  const client = axios.create({
    baseURL: BASE_URL,
    timeout: 30_000,
    headers: { 'Content-Type': 'application/json' },
  })

  // Request interceptor — attach auth token if available in future
  client.interceptors.request.use((config) => {
    return config
  })

  // Response interceptor — normalise errors
  client.interceptors.response.use(
    (res) => res,
    (err) => {
      const message: string =
        err.response?.data?.detail ??
        err.response?.data?.message ??
        err.message ??
        'Unknown error'
      return Promise.reject(new Error(message))
    }
  )

  return client
}

const client = createClient()

// ─────────────────────────────────────────────────────────────────────────────
//  Generic request helper
// ─────────────────────────────────────────────────────────────────────────────

async function get<T>(url: string, config?: AxiosRequestConfig): Promise<T> {
  const res = await client.get<T>(url, config)
  return res.data
}

async function post<T>(url: string, data?: unknown, config?: AxiosRequestConfig): Promise<T> {
  const res = await client.post<T>(url, data, config)
  return res.data
}

// ─────────────────────────────────────────────────────────────────────────────
//  System
// ─────────────────────────────────────────────────────────────────────────────

export const systemApi = {
  health: () => get<{ status: string; service: string; version: string }>('/health'),
  status: () => get<{ status: string; websockets: { active_connections: number } }>('/status'),
}

// ─────────────────────────────────────────────────────────────────────────────
//  Models / Inference
// ─────────────────────────────────────────────────────────────────────────────

export const modelsApi = {
  list: () => get<ModelInfo[]>('/inference/models'),

  getActive: () => get<ModelInfo | null>('/inference/models/active'),

  activate: (modelName: string, confidenceThreshold?: number) =>
    post<ActivateModelResponse>(`/inference/models/${modelName}/activate`, {
      confidence_threshold: confidenceThreshold ?? null,
    }),

  getInfo: (modelName: string) => get<ModelInfo>(`/inference/models/${modelName}`),
}

export const inferenceApi = {
  inferFile: (file: File, confidenceThreshold?: number): Promise<InferenceResult> => {
    const form = new FormData()
    form.append('file', file)
    if (confidenceThreshold !== undefined) {
      form.append('confidence_threshold', String(confidenceThreshold))
    }
    return post<InferenceResult>('/inference/frame', form, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
  },

  inferBase64: (imageB64: string, confidenceThreshold?: number) =>
    post<InferenceResult>('/inference/frame/base64', {
      image_b64: imageB64,
      confidence_threshold: confidenceThreshold ?? null,
    }),
}

// ─────────────────────────────────────────────────────────────────────────────
//  Training
// ─────────────────────────────────────────────────────────────────────────────

export const trainingApi = {
  start: (request: TrainingStartRequest) =>
    post<{ job_id: string; model_name: string; status: string; message: string }>(
      '/train/start',
      request
    ),

  stop: () =>
    post<{ job_id: string | null; message: string }>('/train/stop'),

  cancel: (jobId: string) =>
    post<{ job_id: string; message: string }>(`/train/${jobId}/cancel`),

  getJob: (jobId: string) => get<TrainingJob>(`/train/${jobId}`),

  listJobs: (status?: string) =>
    get<TrainingJob[]>('/train/', { params: status ? { status } : undefined }),

  getActive: () => get<TrainingJob | null>('/train/active'),

  getStatus: () =>
    get<{ active: TrainingJob | null; is_running: boolean; total_jobs: number }>('/train/status'),
}

// ─────────────────────────────────────────────────────────────────────────────
//  Metrics
// ─────────────────────────────────────────────────────────────────────────────

export const metricsApi = {
  getSystem: () => get<SystemSnapshot>('/metrics/system'),

  getSystemHistory: (n?: number) =>
    get<SystemSnapshot[]>('/metrics/system/history', { params: n ? { n } : undefined }),

  getAllInferenceStats: () => get<Record<string, InferenceStats>>('/metrics/inference'),

  getInferenceStats: (modelName: string) =>
    get<InferenceStats>(`/metrics/inference/${modelName}`),

  getInferenceHistory: (modelName: string, limit?: number) =>
    get<InferenceRecord[]>(`/metrics/inference/${modelName}/history`, {
      params: limit ? { limit } : undefined,
    }),

  getTrainingHistory: (modelName: string, lastN?: number) =>
    get<object[]>(`/metrics/training/${modelName}`, {
      params: lastN ? { last_n: lastN } : undefined,
    }),

  getBestEpoch: (modelName: string) => get<object>(`/metrics/training/${modelName}/best`),

  getSummary: () => get<MetricsSummary>('/metrics/summary'),

  resetMetrics: (modelName?: string) =>
    post<{ message: string }>('/metrics/reset', {
      model_name: modelName ?? null,
      confirm: true,
    }),
}

// ─────────────────────────────────────────────────────────────────────────────
//  Benchmark
// ─────────────────────────────────────────────────────────────────────────────

export const benchmarkApi = {
  /** Start a full evaluation (fire-and-forget). Returns {run_id}. */
  start: (
    modelName: string,
    opts?: { numFrames?: number; frameWidth?: number; frameHeight?: number; runTag?: string; runNotes?: string }
  ) =>
    post<{ run_id: string; status: string; message: string }>('/benchmark', {
      model_name: modelName,
      num_frames: opts?.numFrames ?? 200,
      frame_width: opts?.frameWidth ?? 640,
      frame_height: opts?.frameHeight ?? 480,
      run_tag: opts?.runTag ?? null,
      run_notes: opts?.runNotes ?? null,
    }),

  run: (
    modelName: string,
    opts?: { warmupRuns?: number; measureRuns?: number; frameWidth?: number; frameHeight?: number }
  ) =>
    post<BenchmarkResult>('/benchmark/run', {
      model_name: modelName,
      warmup_runs: opts?.warmupRuns ?? null,
      measure_runs: opts?.measureRuns ?? null,
      frame_width: opts?.frameWidth ?? 640,
      frame_height: opts?.frameHeight ?? 480,
    }),

  compare: (
    modelNames: string[],
    opts?: { warmupRuns?: number; measureRuns?: number; frameWidth?: number; frameHeight?: number }
  ) =>
    post<BenchmarkComparison>('/benchmark/compare', {
      model_names: modelNames,
      warmup_runs: opts?.warmupRuns ?? null,
      measure_runs: opts?.measureRuns ?? null,
      frame_width: opts?.frameWidth ?? 640,
      frame_height: opts?.frameHeight ?? 480,
    }),

  listResults: (modelName?: string) =>
    get<BenchmarkResult[]>('/benchmark/results', {
      params: modelName ? { model_name: modelName } : undefined,
    }),

  getResult: (runId: string) => get<BenchmarkResult>(`/benchmark/results/${runId}`),
}

// ─────────────────────────────────────────────────────────────────────────────
//  Export
// ─────────────────────────────────────────────────────────────────────────────

export const exportApi = {
  downloadInferenceJson: (modelName: string) =>
    `${BASE_URL}/export/metrics/${modelName}`,

  downloadInferenceCsv: (modelName: string) =>
    `${BASE_URL}/export/metrics/${modelName}/csv`,

  downloadTrainingJson: (modelName: string) =>
    `${BASE_URL}/export/training/${modelName}`,

  downloadTrainingCsv: (modelName: string) =>
    `${BASE_URL}/export/training/${modelName}/csv`,

  downloadBenchmarkJson: (runId: string) =>
    `${BASE_URL}/export/benchmark/${runId}`,

  downloadBenchmarkHtmlReport: (runId: string) =>
    `${BASE_URL}/export/benchmark/${runId}/report.html`,

  downloadBenchmarkAllJson: (modelName?: string) =>
    `${BASE_URL}/export/benchmark/all${modelName ? `?model_name=${encodeURIComponent(modelName)}` : ''}`,

  downloadBenchmarkAllCsv: (modelName?: string) =>
    `${BASE_URL}/export/benchmark/all/csv${modelName ? `?model_name=${encodeURIComponent(modelName)}` : ''}`,

  downloadReport: () => `${BASE_URL}/export/report`,
}

// ─────────────────────────────────────────────────────────────────────────────
//  Comparative benchmark API
// ─────────────────────────────────────────────────────────────────────────────

export const comparativeApi = {
  start: (
    modelNames: string[],
    opts?: { numFrames?: number; frameWidth?: number; frameHeight?: number; runTag?: string; runNotes?: string }
  ) =>
    post<{ run_id: string; status: string; message: string }>('/benchmark/comparative', {
      model_names: modelNames,
      num_frames:  opts?.numFrames  ?? 200,
      frame_width: opts?.frameWidth ?? 640,
      frame_height: opts?.frameHeight ?? 480,
      run_tag:  opts?.runTag  ?? null,
      run_notes: opts?.runNotes ?? null,
    }),

  getResult: (runId: string) =>
    get<import('@/types').ComparativeResult>(`/benchmark/comparative/${runId}`),

  list: () =>
    get<import('@/types').ComparativeResult[]>('/benchmark/comparative'),
}

export default client

import { create } from 'zustand'
import { devtools } from 'zustand/middleware'
import { format } from 'date-fns'
import type {
  SystemSnapshot,
  InferenceStats,
  TrainingJob,
  EpochResult,
  WsLogEntry,
  ChartPoint,
  ConnectionStatus,
  MetricsSummary,
} from '@/types'

const MAX_CHART_POINTS = 120   // ~2 min at 1s interval
const MAX_LOG_ENTRIES = 500
const MAX_TRAIN_LOG_LINES = 2000

// ─────────────────────────────────────────────────────────────────────────────
//  Metrics store — system snapshots, inference stats, training progress, logs
// ─────────────────────────────────────────────────────────────────────────────

interface MetricsState {
  // System performance
  latestSnapshot: SystemSnapshot | null
  fpsHistory: ChartPoint[]
  latencyHistory: ChartPoint[]
  cpuHistory: ChartPoint[]
  memoryHistory: ChartPoint[]
  metricsWsStatus: ConnectionStatus

  // Inference stats (keyed by model name)
  inferenceStats: Record<string, InferenceStats>
  metricsSummary: MetricsSummary | null

  // Training
  activeJob: TrainingJob | null
  trainingJobs: TrainingJob[]
  trainingWsStatus: ConnectionStatus

  // Logs (application / system)
  logs: WsLogEntry[]
  logsWsStatus: ConnectionStatus
  logFilter: string

  // Training subprocess logs (raw stdout/stderr lines)
  trainLogs: string[]
  trainLogsJobId: string | null
  trainLogsWsStatus: ConnectionStatus

  // Actions
  pushSnapshot: (snap: SystemSnapshot, latency_ms?: number) => void
  setMetricsWsStatus: (status: ConnectionStatus) => void
  setInferenceStats: (stats: Record<string, InferenceStats>) => void
  setMetricsSummary: (summary: MetricsSummary) => void
  setActiveJob: (job: TrainingJob | null) => void
  updateActiveJob: (patch: Partial<TrainingJob>) => void
  appendEpoch: (epoch: EpochResult) => void
  setTrainingJobs: (jobs: TrainingJob[]) => void
  setTrainingWsStatus: (status: ConnectionStatus) => void
  appendLog: (entry: WsLogEntry) => void
  prependLogs: (entries: WsLogEntry[]) => void
  setLogsWsStatus: (status: ConnectionStatus) => void
  setLogFilter: (filter: string) => void
  clearLogs: () => void

  appendTrainLog: (line: string, jobId: string) => void
  setTrainLogs: (lines: string[], jobId: string) => void
  prependTrainLogs: (lines: string[], jobId: string) => void
  clearTrainLogs: () => void
  setTrainLogsWsStatus: (status: ConnectionStatus) => void
}

function makeTimeLabel(): string {
  return format(new Date(), 'HH:mm:ss')
}

function appendCapped<T>(arr: T[], item: T, max: number): T[] {
  const next = [...arr, item]
  return next.length > max ? next.slice(next.length - max) : next
}

export const useMetricsStore = create<MetricsState>()(
  devtools(
    (set) => ({
      // ── initial state ────────────────────────────────────────────────
      latestSnapshot: null,
      fpsHistory: [],
      latencyHistory: [],
      cpuHistory: [],
      memoryHistory: [],
      metricsWsStatus: 'disconnected',

      inferenceStats: {},
      metricsSummary: null,

      activeJob: null,
      trainingJobs: [],
      trainingWsStatus: 'disconnected',

      logs: [],
      logsWsStatus: 'disconnected',
      logFilter: '',

      trainLogs: [],
      trainLogsJobId: null,
      trainLogsWsStatus: 'disconnected',

      // ── actions ──────────────────────────────────────────────────────
      pushSnapshot: (snap, latency_ms) =>
        set((state) => {
          const label = makeTimeLabel()
          return {
            latestSnapshot: snap,
            fpsHistory: appendCapped(
              state.fpsHistory,
              { time: label, value: snap.inference.fps },
              MAX_CHART_POINTS
            ),
            latencyHistory:
              latency_ms !== undefined
                ? appendCapped(
                    state.latencyHistory,
                    { time: label, value: latency_ms },
                    MAX_CHART_POINTS
                  )
                : state.latencyHistory,
            cpuHistory: appendCapped(
              state.cpuHistory,
              { time: label, value: snap.cpu.overall_percent },
              MAX_CHART_POINTS
            ),
            memoryHistory: appendCapped(
              state.memoryHistory,
              { time: label, value: snap.memory.percent },
              MAX_CHART_POINTS
            ),
          }
        }),

      setMetricsWsStatus: (status) => set({ metricsWsStatus: status }),

      setInferenceStats: (stats) => set({ inferenceStats: stats }),

      setMetricsSummary: (summary) => set({ metricsSummary: summary }),

      setActiveJob: (job) => set({ activeJob: job }),

      updateActiveJob: (patch) =>
        set((state) => ({
          activeJob: state.activeJob
            ? { ...state.activeJob, ...patch }
            : null,
        })),

      appendEpoch: (epoch) =>
        set((state) => {
          if (!state.activeJob) return {}
          return {
            activeJob: {
              ...state.activeJob,
              current_epoch: epoch.epoch,
              epoch_history: [...state.activeJob.epoch_history, epoch],
            },
          }
        }),

      setTrainingJobs: (jobs) => set({ trainingJobs: jobs }),

      setTrainingWsStatus: (status) => set({ trainingWsStatus: status }),

      appendLog: (entry) =>
        set((state) => {
          const logs = [entry, ...state.logs]
          return { logs: logs.slice(0, MAX_LOG_ENTRIES) }
        }),

      prependLogs: (entries) =>
        set((state) => {
          const logs = [...entries, ...state.logs]
          return { logs: logs.slice(0, MAX_LOG_ENTRIES) }
        }),

      setLogsWsStatus: (status) => set({ logsWsStatus: status }),

      setLogFilter: (filter) => set({ logFilter: filter }),

      clearLogs: () => set({ logs: [] }),

      appendTrainLog: (line, jobId) =>
        set((state) => {
          const logs = [...state.trainLogs, line]
          return {
            trainLogs: logs.length > MAX_TRAIN_LOG_LINES ? logs.slice(-MAX_TRAIN_LOG_LINES) : logs,
            trainLogsJobId: jobId,
          }
        }),

      setTrainLogs: (lines, jobId) =>
        set({
          trainLogs: lines.slice(-MAX_TRAIN_LOG_LINES),
          trainLogsJobId: jobId,
        }),

      prependTrainLogs: (lines, jobId) =>
        set((state) => ({
          trainLogs: [...lines, ...state.trainLogs].slice(-MAX_TRAIN_LOG_LINES),
          trainLogsJobId: jobId,
        })),

      clearTrainLogs: () => set({ trainLogs: [], trainLogsJobId: null }),

      setTrainLogsWsStatus: (status) => set({ trainLogsWsStatus: status }),
    }),
    { name: 'MetricsStore' }
  )
)

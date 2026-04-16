/**
 * Training page â€” Phase 4
 *
 * Features:
 *   â€¢ Training configuration form (model, hyperparams)
 *   â€¢ Start / Stop controls with conflict prevention
 *   â€¢ Real-time progress bar + epoch stats
 *   â€¢ Live log panel â€” streams raw subprocess stdout via /train/ws/logs
 *   â€¢ Loss & Accuracy charts (Recharts)
 *   â€¢ Epoch history table
 */

import { useCallback, useEffect, useRef, useState } from 'react'
import {
  PlayCircle,
  Square,
  Terminal,
  BarChart2,
  List,
  ArrowDownCircle,
  Trash2,
  AlertCircle,
} from 'lucide-react'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts'
import { cn, statusColor } from '@/lib/utils'
import { trainingApi } from '@/services/api'
import { createTrainingWs, createTrainLogsWs } from '@/services/websocket'
import { useMetricsStore } from '@/store/useMetricsStore'
import { useModelStore } from '@/store/useModelStore'
import type { TrainingStartRequest, WsMessage, WsEpochMessage, EpochResult, WsTrainLog, WsTrainLogBatch } from '@/types'

// â”€â”€ Constants â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

const DEFAULT_FORM: TrainingStartRequest = {
  model_name: 'scratch_cnn',
  epochs: 10,
  batch_size: 32,
  learning_rate: 0.001,
  optimizer: 'adam',
  weight_decay: 0.0001,
  early_stopping_patience: 3,
  checkpoint_every_n_epochs: 5,
}

// â”€â”€ Helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/** Classify a raw log line to a visual level. */
function classifyLine(line: string): 'info' | 'warning' | 'error' | 'success' | 'dim' | 'default' {
  const upper = line.toUpperCase()
  if (upper.includes('ERROR') || upper.includes('CRITICAL') || upper.includes('EXCEPTION') || upper.includes('TRACEBACK')) return 'error'
  if (upper.includes('WARNING') || upper.includes('WARN')) return 'warning'
  if (upper.includes('BEST MODEL') || upper.includes('TRAINING COMPLETE') || upper.includes('COMPLETE') || upper.includes('SAVED')) return 'success'
  if (upper.includes('EPOCH [')) return 'info'
  if (line.trim() === '' || line.startsWith('#') || /^={3,}/.test(line)) return 'dim'
  return 'default'
}

const LINE_COLOR: Record<ReturnType<typeof classifyLine>, string> = {
  error:   'text-red-400',
  warning: 'text-yellow-400',
  success: 'text-emerald-400',
  info:    'text-blue-400',
  dim:     'text-slate-600',
  default: 'text-slate-300',
}

// â”€â”€ Sub-components â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

interface FieldProps { label: string; children: React.ReactNode }
function Field({ label, children }: FieldProps) {
  return (
    <div>
      <label className="block text-[11px] text-muted-foreground mb-1 uppercase tracking-wide font-medium">
        {label}
      </label>
      {children}
    </div>
  )
}

const inputCls =
  'w-full rounded bg-[hsl(var(--background))] border border-[hsl(var(--border))] px-3 py-1.5 text-sm font-mono text-foreground focus:outline-none focus:ring-1 focus:ring-primary'

function EpochTable({ epochs }: { epochs: EpochResult[] }) {
  if (!epochs.length) return null
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs font-mono">
        <thead>
          <tr className="border-b border-[hsl(var(--border))] text-muted-foreground">
            {['Epoch', 'Train Loss', 'Val Loss', 'Train Acc', 'Val Acc', 'LR'].map((h) => (
              <th key={h} className="text-left py-1.5 pr-4 font-medium">{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {[...epochs].reverse().map((e) => (
            <tr key={e.epoch} className="border-b border-[hsl(var(--border))]/40 hover:bg-muted/20 transition-colors">
              <td className="py-1.5 pr-4 text-primary font-semibold">{e.epoch}</td>
              <td className="py-1.5 pr-4">{e.train_loss.toFixed(4)}</td>
              <td className="py-1.5 pr-4">{e.val_loss.toFixed(4)}</td>
              <td className="py-1.5 pr-4">{(e.train_acc * 100).toFixed(1)}%</td>
              <td className="py-1.5 pr-4">{(e.val_acc * 100).toFixed(1)}%</td>
              <td className="py-1.5 pr-4">{e.lr.toExponential(2)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
//  Live Log Panel
// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

interface LogPanelProps {
  lines: string[]
  onClear: () => void
}

function LogPanel({ lines, onClear }: LogPanelProps) {
  const bottomRef = useRef<HTMLDivElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const [autoScroll, setAutoScroll] = useState(true)

  // Auto-scroll to bottom when new lines arrive
  useEffect(() => {
    if (autoScroll && bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: 'smooth' })
    }
  }, [lines, autoScroll])

  // Detect manual scroll-up to disable auto-scroll
  const handleScroll = useCallback(() => {
    const el = containerRef.current
    if (!el) return
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 40
    setAutoScroll(atBottom)
  }, [])

  if (lines.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-48 gap-2 text-muted-foreground">
        <Terminal className="w-6 h-6 opacity-40" />
        <p className="text-xs">No log output yet â€” start training to see live output.</p>
      </div>
    )
  }

  return (
    <div className="relative">
      {/* Toolbar */}
      <div className="flex items-center justify-between px-3 py-1.5 border-b border-[hsl(var(--border))] bg-[hsl(var(--background))]">
        <span className="text-[10px] font-mono text-muted-foreground">{lines.length} lines</span>
        <div className="flex items-center gap-2">
          <button
            onClick={() => { setAutoScroll(true); bottomRef.current?.scrollIntoView({ behavior: 'smooth' }) }}
            className={cn('flex items-center gap-1 text-[10px] px-2 py-0.5 rounded transition-colors', autoScroll ? 'text-primary bg-primary/10' : 'text-muted-foreground hover:text-foreground')}
            title="Auto-scroll to bottom"
          >
            <ArrowDownCircle className="w-3 h-3" />
            {autoScroll ? 'Auto' : 'Scroll'}
          </button>
          <button
            onClick={onClear}
            className="flex items-center gap-1 text-[10px] text-muted-foreground hover:text-red-400 transition-colors px-2 py-0.5 rounded"
            title="Clear logs"
          >
            <Trash2 className="w-3 h-3" />
            Clear
          </button>
        </div>
      </div>

      {/* Log lines */}
      <div
        ref={containerRef}
        onScroll={handleScroll}
        className="h-72 overflow-y-auto bg-[hsl(var(--background))] font-mono text-[11px] leading-relaxed p-3 space-y-0.5"
        aria-label="Training logs"
      >
        {lines.map((line, i) => {
          const level = classifyLine(line)
          return (
            <div key={i} className={cn('whitespace-pre-wrap break-all', LINE_COLOR[level])}>
              {line || '\u00a0'}
            </div>
          )
        })}
        <div ref={bottomRef} />
      </div>

      {/* Floating scroll-down hint */}
      {!autoScroll && (
        <button
          onClick={() => { setAutoScroll(true); bottomRef.current?.scrollIntoView({ behavior: 'smooth' }) }}
          className="absolute bottom-3 right-3 flex items-center gap-1 bg-primary/90 text-primary-foreground text-[10px] px-2 py-1 rounded-full shadow-lg"
        >
          <ArrowDownCircle className="w-3 h-3" />
          Jump to latest
        </button>
      )}
    </div>
  )
}

// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
//  Main page
// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

type TabId = 'charts' | 'logs' | 'history'

export function Training() {
  // â”€â”€ Form state â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  const [form, setForm] = useState<TrainingStartRequest>(DEFAULT_FORM)
  const [submitting, setSubmitting] = useState(false)
  const [stopping, setStopping] = useState(false)
  const [formError, setFormError] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState<TabId>('charts')

  // â”€â”€ Store â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  const activeJob = useMetricsStore((s) => s.activeJob)
  const setActiveJob = useMetricsStore((s) => s.setActiveJob)
  const appendEpoch = useMetricsStore((s) => s.appendEpoch)
  const trainLogs = useMetricsStore((s) => s.trainLogs)
  const appendTrainLog = useMetricsStore((s) => s.appendTrainLog)
  const setTrainLogs = useMetricsStore((s) => s.setTrainLogs)
  const clearTrainLogs = useMetricsStore((s) => s.clearTrainLogs)
  const trainLogsWsStatus = useMetricsStore((s) => s.trainLogsWsStatus)
  const setTrainLogsWsStatus = useMetricsStore((s) => s.setTrainLogsWsStatus)
  const availableModels = useModelStore((s) => s.availableModels)

  // â”€â”€ WS refs â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  const progressWsRef = useRef<ReturnType<typeof createTrainingWs> | null>(null)
  const logsWsRef = useRef<ReturnType<typeof createTrainLogsWs> | null>(null)

  // â”€â”€ Log WS â€” always connected while page is mounted â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  useEffect(() => {
    const ws = createTrainLogsWs(
      (msg: WsMessage) => {
        if (msg.type === 'train_log') {
          const m = msg as WsTrainLog
          appendTrainLog(m.line, m.job_id)
          // Auto-switch to logs tab while running
          setActiveTab((prev) => prev === 'history' ? prev : 'logs')
        } else if (msg.type === 'train_log_batch') {
          const m = msg as WsTrainLogBatch
          setTrainLogs(m.lines, m.job_id)
        }
      },
      (s) => setTrainLogsWsStatus(s)
    )
    ws.connect()
    logsWsRef.current = ws
    return () => {
      ws.disconnect()
      logsWsRef.current = null
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // â”€â”€ Progress WS â€” connect when a running job exists â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  useEffect(() => {
    if (!activeJob || activeJob.status !== 'running') return

    const ws = createTrainingWs(
      activeJob.job_id,
      (msg: WsMessage) => {
        if (msg.type === 'epoch') {
          const em = msg as WsEpochMessage
          if (em.epoch_result) appendEpoch(em.epoch_result)
        }
        if (msg.type === 'status_change' || msg.type === 'job_complete') {
          trainingApi.getJob(activeJob.job_id).then(setActiveJob).catch(() => {})
        }
      },
      () => {}
    )
    ws.connect()
    progressWsRef.current = ws
    return () => {
      ws.disconnect()
      progressWsRef.current = null
    }
  }, [activeJob?.job_id, activeJob?.status, appendEpoch, setActiveJob])

  // â”€â”€ Fetch active job on mount â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  useEffect(() => {
    trainingApi
      .getActive()
      .then((j) => { if (j) setActiveJob(j) })
      .catch(() => {})
  }, [setActiveJob])

  // â”€â”€ Handlers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setFormError(null)
    setSubmitting(true)
    clearTrainLogs()
    try {
      const started = await trainingApi.start(form)
      const job = await trainingApi.getJob(started.job_id)
      setActiveJob(job)
      setActiveTab('logs')
    } catch (err: unknown) {
      setFormError(err instanceof Error ? err.message : 'Failed to start training')
    } finally {
      setSubmitting(false)
    }
  }

  const handleStop = async () => {
    setStopping(true)
    try {
      await trainingApi.stop()
      if (activeJob) {
        const updated = await trainingApi.getJob(activeJob.job_id)
        setActiveJob(updated)
      }
    } catch {
      // Silently handle â€” the job may have already finished
    } finally {
      setStopping(false)
    }
  }

  // â”€â”€ Derived â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

  const isRunning = activeJob?.status === 'running'
  const progress = activeJob
    ? Math.round(((activeJob.current_epoch ?? 0) / Math.max(activeJob.total_epochs, 1)) * 100)
    : 0
  const chartData = activeJob?.epoch_history ?? []
  const latestEpoch = chartData.at(-1)

  const TABS: { id: TabId; label: string; icon: React.ReactNode }[] = [
    { id: 'charts',  label: 'Charts',  icon: <BarChart2 className="w-3.5 h-3.5" /> },
    { id: 'logs',    label: `Logs${trainLogs.length > 0 ? ` (${trainLogs.length})` : ''}`, icon: <Terminal className="w-3.5 h-3.5" /> },
    { id: 'history', label: `History${chartData.length > 0 ? ` (${chartData.length})` : ''}`, icon: <List className="w-3.5 h-3.5" /> },
  ]

  return (
    <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">

      {/* â•â•â•â•â•â•â•â•â•â•â•â•â•â• Left: Config + quick stats â•â•â•â•â•â•â•â•â•â•â•â•â•â• */}
      <div className="xl:col-span-1 space-y-4">

        {/* Config form */}
        <div className="rounded-lg border border-[hsl(var(--border))] bg-[hsl(var(--card))] p-5">
          <h2 className="text-sm font-semibold text-foreground mb-4">Training Configuration</h2>
          <form onSubmit={handleSubmit} className="space-y-4">

            <Field label="Model">
              <select
                className={inputCls}
                value={form.model_name}
                onChange={(e) => setForm({ ...form, model_name: e.target.value })}
                disabled={isRunning || submitting}
              >
                {availableModels.length > 0
                  ? availableModels.map((m) => <option key={m.name} value={m.name}>{m.name}</option>)
                  : <option value="scratch_cnn">scratch_cnn</option>}
              </select>
            </Field>

            <div className="grid grid-cols-2 gap-3">
              <Field label="Epochs">
                <input type="number" className={inputCls} min={1} max={1000}
                  value={form.epochs}
                  onChange={(e) => setForm({ ...form, epochs: Number(e.target.value) })}
                  disabled={isRunning || submitting} />
              </Field>
              <Field label="Batch Size">
                <input type="number" className={inputCls} min={1}
                  value={form.batch_size}
                  onChange={(e) => setForm({ ...form, batch_size: Number(e.target.value) })}
                  disabled={isRunning || submitting} />
              </Field>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <Field label="Learning Rate">
                <input type="number" className={inputCls} step="0.0001" min="0.00001"
                  value={form.learning_rate}
                  onChange={(e) => setForm({ ...form, learning_rate: Number(e.target.value) })}
                  disabled={isRunning || submitting} />
              </Field>
              <Field label="Patience">
                <input type="number" className={inputCls} min={0}
                  value={form.early_stopping_patience}
                  onChange={(e) => setForm({ ...form, early_stopping_patience: Number(e.target.value) })}
                  disabled={isRunning || submitting} />
              </Field>
            </div>

            <Field label="Optimizer">
              <select className={inputCls} value={form.optimizer}
                onChange={(e) => setForm({ ...form, optimizer: e.target.value as TrainingStartRequest['optimizer'] })}
                disabled={isRunning || submitting}>
                {['adam', 'sgd', 'rmsprop', 'adamw'].map((o) => <option key={o} value={o}>{o}</option>)}
              </select>
            </Field>

            <Field label="Weight Decay">
              <input type="number" className={inputCls} step="0.0001" min="0"
                value={form.weight_decay}
                onChange={(e) => setForm({ ...form, weight_decay: Number(e.target.value) })}
                disabled={isRunning || submitting} />
            </Field>

            {formError && (
              <div className="flex items-start gap-2 rounded-md bg-red-950/40 border border-red-800 p-2.5">
                <AlertCircle className="w-3.5 h-3.5 text-red-400 flex-shrink-0 mt-0.5" />
                <p className="text-xs text-red-300">{formError}</p>
              </div>
            )}

            <button
              type="submit"
              disabled={submitting || isRunning}
              className="w-full flex items-center justify-center gap-2 rounded-md bg-primary px-4 py-2 text-sm font-semibold text-primary-foreground disabled:opacity-50 hover:opacity-90 transition"
            >
              <PlayCircle className="w-4 h-4" />
              {submitting ? 'Startingâ€¦' : 'Start Training'}
            </button>
          </form>
        </div>

        {/* Quick stats */}
        {latestEpoch && (
          <div className="rounded-lg border border-[hsl(var(--border))] bg-[hsl(var(--card))] p-4">
            <p className="text-[11px] text-muted-foreground uppercase tracking-wide font-medium mb-3">
              Latest Epoch Stats
            </p>
            <div className="grid grid-cols-2 gap-2 text-xs">
              {[
                ['Train Loss', latestEpoch.train_loss.toFixed(4)],
                ['Val Loss',   latestEpoch.val_loss.toFixed(4)],
                ['Train Acc',  `${(latestEpoch.train_acc * 100).toFixed(1)}%`],
                ['Val Acc',    `${(latestEpoch.val_acc  * 100).toFixed(1)}%`],
              ].map(([label, val]) => (
                <div key={label} className="rounded-md bg-background px-2.5 py-2">
                  <p className="text-muted-foreground text-[10px]">{label}</p>
                  <p className="font-mono font-bold text-foreground text-sm">{val}</p>
                </div>
              ))}
            </div>
            {activeJob?.best_val_loss != null && (
              <div className="mt-2 text-[10px] font-mono text-muted-foreground text-center">
                Best val loss <span className="text-emerald-400">{activeJob.best_val_loss.toFixed(4)}</span> @ epoch {activeJob.best_epoch}
              </div>
            )}
          </div>
        )}
      </div>

      {/* â•â•â•â•â•â•â•â•â•â•â•â•â•â• Right: progress + tabs â•â•â•â•â•â•â•â•â•â•â•â•â•â• */}
      <div className="xl:col-span-2 space-y-4">

        {/* â”€â”€ Active job card â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */}
        {activeJob ? (
          <div className="rounded-lg border border-[hsl(var(--border))] bg-[hsl(var(--card))] p-5 space-y-4">
            <div className="flex items-start justify-between gap-4">
              <div className="min-w-0">
                <p className="text-[10px] text-muted-foreground uppercase tracking-wide">Current Job</p>
                <p className="font-mono text-sm font-semibold text-foreground truncate">{activeJob.job_id}</p>
                <p className="text-xs text-muted-foreground">{activeJob.model_name}</p>
              </div>
              <div className="flex items-center gap-2 flex-shrink-0">
                <span className={cn('text-[11px] font-semibold px-2 py-0.5 rounded-full', statusColor(activeJob.status))}>
                  {activeJob.status}
                </span>
                {isRunning && (
                  <button
                    onClick={handleStop}
                    disabled={stopping}
                    className="flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-md border border-red-800 text-red-400 hover:bg-red-950/40 disabled:opacity-50 transition-colors font-medium"
                  >
                    <Square className="w-3 h-3" />
                    {stopping ? 'Stoppingâ€¦' : 'Stop'}
                  </button>
                )}
              </div>
            </div>

            {/* Progress bar */}
            <div>
              <div className="flex justify-between text-xs text-muted-foreground mb-1.5">
                <span>Epoch {activeJob.current_epoch} / {activeJob.total_epochs}</span>
                <span className="font-mono font-semibold text-foreground">{progress}%</span>
              </div>
              <div className="h-2.5 rounded-full bg-muted overflow-hidden">
                <div
                  className={cn(
                    'h-full rounded-full transition-all duration-700',
                    activeJob.status === 'completed' ? 'bg-emerald-500' :
                    activeJob.status === 'failed'    ? 'bg-red-500' :
                    activeJob.status === 'cancelled' ? 'bg-yellow-500' :
                    'bg-primary'
                  )}
                  style={{ width: `${progress}%` }}
                />
              </div>
            </div>

            {/* Error */}
            {activeJob.error_message && (
              <div className="flex items-start gap-2 rounded-md bg-red-950/30 border border-red-900 px-3 py-2">
                <AlertCircle className="w-3.5 h-3.5 text-red-400 flex-shrink-0 mt-0.5" />
                <p className="text-xs text-red-300 font-mono">{activeJob.error_message}</p>
              </div>
            )}
          </div>
        ) : (
          <div className="rounded-lg border border-dashed border-[hsl(var(--border))] bg-[hsl(var(--card))] p-10 text-center">
            <p className="text-muted-foreground text-sm">No active training job.</p>
            <p className="text-muted-foreground/60 text-xs mt-1">Configure parameters on the left and click Start Training.</p>
          </div>
        )}

        {/* â”€â”€ Tabs â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */}
        <div className="rounded-lg border border-[hsl(var(--border))] bg-[hsl(var(--card))] overflow-hidden">
          {/* Tab bar */}
          <div className="flex border-b border-[hsl(var(--border))]">
            {TABS.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={cn(
                  'flex items-center gap-1.5 px-4 py-2.5 text-xs font-medium transition-colors border-b-2 -mb-px',
                  activeTab === tab.id
                    ? 'border-primary text-foreground'
                    : 'border-transparent text-muted-foreground hover:text-foreground'
                )}
              >
                {tab.icon}
                {tab.label}
              </button>
            ))}

            {/* WS status indicator (right-aligned in tab bar) */}
            <div className="ml-auto flex items-center gap-1.5 pr-3 text-[10px] text-muted-foreground">
              <span className={cn('h-1.5 w-1.5 rounded-full',
                trainLogsWsStatus === 'connected'    ? 'bg-emerald-400' :
                trainLogsWsStatus === 'connecting'   ? 'bg-yellow-400 animate-pulse' :
                trainLogsWsStatus === 'error'        ? 'bg-red-500' :
                'bg-slate-600'
              )} />
              logs ws
            </div>
          </div>

          {/* â”€â”€ Charts tab â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */}
          {activeTab === 'charts' && (
            <div className="p-4">
              {chartData.length === 0 ? (
                <div className="flex flex-col items-center justify-center h-48 gap-2 text-muted-foreground">
                  <BarChart2 className="w-6 h-6 opacity-40" />
                  <p className="text-xs">Charts will appear here as epochs complete.</p>
                </div>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {/* Loss chart */}
                  <div>
                    <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wide mb-3">Loss</p>
                    <ResponsiveContainer width="100%" height={180}>
                      <LineChart data={chartData} margin={{ top: 4, right: 4, bottom: 0, left: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                        <XAxis dataKey="epoch" tick={{ fontSize: 10, fill: 'hsl(var(--muted-foreground))' }} />
                        <YAxis tick={{ fontSize: 10, fill: 'hsl(var(--muted-foreground))' }} width={40} />
                        <Tooltip
                          contentStyle={{ backgroundColor: 'hsl(var(--card))', border: '1px solid hsl(var(--border))', borderRadius: 6, fontSize: 11 }}
                          labelStyle={{ color: 'hsl(var(--foreground))' }}
                        />
                        <Legend wrapperStyle={{ fontSize: 11 }} />
                        <Line type="monotone" dataKey="train_loss" name="Train" stroke="#60a5fa" dot={false} strokeWidth={2} />
                        <Line type="monotone" dataKey="val_loss"   name="Val"   stroke="#f87171" dot={false} strokeWidth={2} strokeDasharray="4 2" />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>

                  {/* Accuracy chart */}
                  <div>
                    <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wide mb-3">Accuracy</p>
                    <ResponsiveContainer width="100%" height={180}>
                      <LineChart data={chartData} margin={{ top: 4, right: 4, bottom: 0, left: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                        <XAxis dataKey="epoch" tick={{ fontSize: 10, fill: 'hsl(var(--muted-foreground))' }} />
                        <YAxis tick={{ fontSize: 10, fill: 'hsl(var(--muted-foreground))' }} width={40}
                          tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`} />
                        <Tooltip
                          contentStyle={{ backgroundColor: 'hsl(var(--card))', border: '1px solid hsl(var(--border))', borderRadius: 6, fontSize: 11 }}
                          labelStyle={{ color: 'hsl(var(--foreground))' }}
                          formatter={(v: number) => `${(v * 100).toFixed(2)}%`}
                        />
                        <Legend wrapperStyle={{ fontSize: 11 }} />
                        <Line type="monotone" dataKey="train_acc" name="Train" stroke="#34d399" dot={false} strokeWidth={2} />
                        <Line type="monotone" dataKey="val_acc"   name="Val"   stroke="#a78bfa" dot={false} strokeWidth={2} strokeDasharray="4 2" />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* â”€â”€ Logs tab â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */}
          {activeTab === 'logs' && (
            <LogPanel lines={trainLogs} onClear={clearTrainLogs} />
          )}

          {/* â”€â”€ History tab â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */}
          {activeTab === 'history' && (
            <div className="p-4">
              {chartData.length === 0 ? (
                <div className="flex flex-col items-center justify-center h-32 gap-2 text-muted-foreground">
                  <List className="w-5 h-5 opacity-40" />
                  <p className="text-xs">Epoch history will appear here.</p>
                </div>
              ) : (
                <EpochTable epochs={chartData} />
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}


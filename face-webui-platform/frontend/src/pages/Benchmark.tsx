import { useState, useEffect, useRef, useCallback } from 'react'
import {
  Play,
  Loader2,
  BarChart2,
  Target,
  Zap,
  Clock,
  AlertCircle,
  Download,
  FileJson,
  FileText,
  Table2,
  Layers,
} from 'lucide-react'
import {
  BarChart,
  Bar,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  LineChart,
  Line,
  ReferenceDot,
  ReferenceLine,
} from 'recharts'
import { cn } from '@/lib/utils'
import { BenchmarkReport } from '@/components/BenchmarkReport'
import { benchmarkApi, exportApi, comparativeApi } from '@/services/api'
import { useModelStore } from '@/store/useModelStore'
import type { BenchmarkResult, BenchmarkComparison, ComparativeResult } from '@/types'

// â”€â”€ helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

const fmt1 = (v: number | null | undefined, suffix = '') =>
  v != null ? `${v.toFixed(1)}${suffix}` : 'â€”'
const fmt2 = (v: number | null | undefined, suffix = '') =>
  v != null ? `${v.toFixed(2)}${suffix}` : 'â€”'
const fmtPct = (v: number | null | undefined) =>
  v != null ? `${(v * 100).toFixed(1)}%` : 'â€”'

// â”€â”€ shared sub-components â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

const inputCls =
  'w-full rounded bg-[hsl(var(--background))] border border-[hsl(var(--border))] px-3 py-1.5 text-sm font-mono text-foreground focus:outline-none focus:ring-1 focus:ring-primary'

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <label className="block text-[11px] text-muted-foreground mb-1 uppercase tracking-wide">
        {label}
      </label>
      {children}
    </div>
  )
}

interface MetricCardProps {
  label: string
  value: string
  sub?: string
  icon: React.ElementType
  color?: string
  wide?: boolean
}
function MetricCard({ label, value, sub, icon: Icon, color = 'text-primary', wide }: MetricCardProps) {
  return (
    <div className={cn('rounded-lg border border-[hsl(var(--border))] bg-[hsl(var(--card))] p-4', wide && 'col-span-2')}>
      <div className="flex items-center gap-2 mb-2">
        <Icon className={cn('w-4 h-4', color)} />
        <span className="text-[11px] text-muted-foreground uppercase tracking-wide">{label}</span>
      </div>
      <p className={cn('text-2xl font-mono font-bold', color)}>{value}</p>
      {sub && <p className="text-[10px] text-muted-foreground mt-1">{sub}</p>}
    </div>
  )
}

// ── PR Curve Chart ───────────────────────────────────────────────────────────

function PrCurveChart({ result }: { result: BenchmarkResult }) {
  if (!result.pr_curve || result.pr_curve.length === 0) return null

  // Sort by recall ascending so the curve flows left → right
  const data = [...result.pr_curve]
    .sort((a, b) => a.recall - b.recall)
    .map((p) => ({ recall: p.recall, precision: p.precision, f1: p.f1, threshold: p.threshold }))

  const bestPoint = result.pr_curve.reduce(
    (best, p) => (p.f1 > best.f1 ? p : best),
    result.pr_curve[0],
  )

  return (
    <div className="rounded-lg border border-[hsl(var(--border))] bg-[hsl(var(--card))] p-4">
      <div className="flex flex-wrap items-start justify-between gap-2 mb-1">
        <p className="text-sm font-semibold text-foreground">Precision–Recall Curve</p>
        <div className="flex flex-wrap items-center gap-2">
          {result.auc_pr != null && (
            <span className="text-[11px] font-mono px-2 py-0.5 rounded-full bg-emerald-950/40 text-emerald-400">
              AUC-PR&nbsp;{result.auc_pr.toFixed(3)}
            </span>
          )}
          {result.best_f1_threshold != null && (
            <span className="text-[11px] font-mono px-2 py-0.5 rounded-full bg-amber-950/40 text-amber-400">
              Best-F1&nbsp;thr&nbsp;{result.best_f1_threshold.toFixed(3)}
            </span>
          )}
          {result.precision_at_recall_90 != null && (
            <span className="text-[11px] font-mono px-2 py-0.5 rounded-full bg-violet-950/40 text-violet-400">
              P@R≥90%&nbsp;=&nbsp;{result.precision_at_recall_90.toFixed(3)}
            </span>
          )}
        </div>
      </div>
      <p className="text-[10px] text-muted-foreground mb-3">
        50-threshold sweep · IoU&nbsp;=&nbsp;{result.iou_threshold ?? 0.5}&nbsp;·&nbsp;
        <span className="text-amber-400 font-medium">●</span>&nbsp;optimal F1 operating point
      </p>
      <ResponsiveContainer width="100%" height={230}>
        <LineChart data={data} margin={{ top: 8, right: 16, left: -8, bottom: 20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" strokeOpacity={0.4} />
          <XAxis
            dataKey="recall"
            type="number"
            domain={[0, 1]}
            tickCount={6}
            tick={{ fontSize: 10, fill: 'hsl(var(--muted-foreground))' }}
            tickLine={false}
            axisLine={false}
            label={{ value: 'Recall', position: 'insideBottom', offset: -12, fontSize: 11, fill: 'hsl(var(--muted-foreground))' }}
          />
          <YAxis
            domain={[0, 1]}
            tickCount={6}
            tick={{ fontSize: 10, fill: 'hsl(var(--muted-foreground))' }}
            tickLine={false}
            axisLine={false}
            width={36}
            label={{ value: 'Precision', angle: -90, position: 'insideLeft', offset: 14, fontSize: 11, fill: 'hsl(var(--muted-foreground))' }}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: 'hsl(var(--card))',
              border: '1px solid hsl(var(--border))',
              borderRadius: '6px',
              fontSize: 11,
            }}
            formatter={(v: number, key: string) => [
              Number(v).toFixed(3),
              key === 'precision' ? 'Precision' : key,
            ]}
            labelFormatter={(recall: number) => `Recall: ${Number(recall).toFixed(3)}`}
          />
          <Line
            type="monotone"
            dataKey="precision"
            name="Precision"
            stroke="#a78bfa"
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 4, fill: '#a78bfa' }}
          />
          {bestPoint && (
            <ReferenceDot
              x={bestPoint.recall}
              y={bestPoint.precision}
              r={6}
              fill="#f59e0b"
              stroke="#fbbf24"
              strokeWidth={2}
              label={{
                value: `F1=${bestPoint.f1.toFixed(2)}`,
                position: 'top',
                fontSize: 10,
                fill: '#f59e0b',
                offset: 8,
              }}
            />
          )}
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

// ── Confidence Distribution Chart ────────────────────────────────────────────

function ConfidenceHistogram({ result }: { result: BenchmarkResult }) {
  const hist = result.confidence_histogram
  if (!hist || hist.counts.every((c) => c === 0)) return null

  const data = hist.bins.slice(0, -1).map((bin, i) => ({
    bin: bin.toFixed(2),
    count: hist.counts[i],
  }))
  const peakBin = data.reduce((max, d) => (d.count > max.count ? d : max), data[0])
  const mean =
    data.reduce((sum, d, i) => sum + (hist.bins[i] + 0.025) * d.count, 0) /
    Math.max(hist.total_detections, 1)

  return (
    <div className="rounded-lg border border-[hsl(var(--border))] bg-[hsl(var(--card))] p-4">
      <div className="flex flex-wrap items-start justify-between gap-2 mb-1">
        <p className="text-sm font-semibold text-foreground">Detection Confidence Distribution</p>
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-[11px] font-mono px-2 py-0.5 rounded-full bg-slate-800 text-muted-foreground">
            {hist.total_detections.toLocaleString()}&nbsp;detections
          </span>
          <span className="text-[11px] font-mono px-2 py-0.5 rounded-full bg-emerald-950/40 text-emerald-400">
            mean&nbsp;{mean.toFixed(3)}
          </span>
          <span className="text-[11px] font-mono px-2 py-0.5 rounded-full bg-sky-950/40 text-sky-400">
            peak&nbsp;{peakBin.bin}–{(parseFloat(peakBin.bin) + 0.05).toFixed(2)}
          </span>
        </div>
      </div>
      <p className="text-[10px] text-muted-foreground mb-3">
        20-bin histogram · skew toward 1.0&nbsp;=&nbsp;confident model · bimodal&nbsp;=&nbsp;poor calibration
      </p>
      <ResponsiveContainer width="100%" height={230}>
        <BarChart data={data} margin={{ top: 4, right: 8, left: -16, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" strokeOpacity={0.4} />
          <XAxis
            dataKey="bin"
            tick={{ fontSize: 9, fill: 'hsl(var(--muted-foreground))' }}
            tickLine={false}
            axisLine={false}
            interval={3}
          />
          <YAxis
            tick={{ fontSize: 10, fill: 'hsl(var(--muted-foreground))' }}
            tickLine={false}
            axisLine={false}
            width={44}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: 'hsl(var(--card))',
              border: '1px solid hsl(var(--border))',
              borderRadius: '6px',
              fontSize: 11,
            }}
            formatter={(v: number) => [v.toLocaleString(), 'Detections']}
            labelFormatter={(bin: string) =>
              `Conf [${bin},\u00a0${(parseFloat(bin) + 0.05).toFixed(2)})`
            }
          />
          <Bar dataKey="count" name="Detections" fill="#34d399" radius={[2, 2, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

// ── Calibration Curve Chart ──────────────────────────────────────────────────

function CalibrationCurveChart({ result }: { result: BenchmarkResult }) {
  const cal = result.calibration_curve
  if (!cal || cal.length < 2) return null

  const data = cal.map((p) => ({
    mean_confidence: p.mean_confidence,
    actual_precision: p.actual_precision,
    count: p.count,
    label: `[${p.bin_start.toFixed(1)}–${p.bin_end.toFixed(1)})`,
  }))

  const overconfident = data.filter((p) => p.mean_confidence > p.actual_precision + 0.05).length
  const underconfident = data.filter((p) => p.actual_precision > p.mean_confidence + 0.05).length
  const calibrationTag =
    overconfident > underconfident
      ? 'over-confident'
      : underconfident > overconfident
        ? 'under-confident'
        : 'well-calibrated'
  const tagColor =
    calibrationTag === 'well-calibrated'
      ? 'text-emerald-400 bg-emerald-950/40'
      : 'text-amber-400 bg-amber-950/40'

  return (
    <div className="rounded-lg border border-[hsl(var(--border))] bg-[hsl(var(--card))] p-4">
      <div className="flex flex-wrap items-start justify-between gap-2 mb-1">
        <p className="text-sm font-semibold text-foreground">Confidence Calibration</p>
        <div className="flex flex-wrap items-center gap-2">
          <span className={cn('text-[11px] font-mono px-2 py-0.5 rounded-full', tagColor)}>
            {calibrationTag}
          </span>
          <span className="text-[11px] font-mono px-2 py-0.5 rounded-full bg-slate-800 text-muted-foreground">
            {cal.length}&nbsp;bins
          </span>
        </div>
      </div>
      <p className="text-[10px] text-muted-foreground mb-3">
        Confidence vs&nbsp;actual precision&nbsp;·&nbsp;‒‒ dashed = perfect calibration (y&nbsp;=&nbsp;x)
      </p>
      <ResponsiveContainer width="100%" height={230}>
        <LineChart data={data} margin={{ top: 8, right: 16, left: -8, bottom: 20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" strokeOpacity={0.4} />
          <XAxis
            dataKey="mean_confidence"
            type="number"
            domain={[0, 1]}
            tickCount={6}
            tick={{ fontSize: 10, fill: 'hsl(var(--muted-foreground))' }}
            tickLine={false}
            axisLine={false}
            label={{ value: 'Mean Confidence', position: 'insideBottom', offset: -12, fontSize: 11, fill: 'hsl(var(--muted-foreground))' }}
          />
          <YAxis
            domain={[0, 1]}
            tickCount={6}
            tick={{ fontSize: 10, fill: 'hsl(var(--muted-foreground))' }}
            tickLine={false}
            axisLine={false}
            width={36}
            label={{ value: 'Actual Precision', angle: -90, position: 'insideLeft', offset: 14, fontSize: 11, fill: 'hsl(var(--muted-foreground))' }}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: 'hsl(var(--card))',
              border: '1px solid hsl(var(--border))',
              borderRadius: '6px',
              fontSize: 11,
            }}
            formatter={(v: number, key: string) => [
              Number(v).toFixed(3),
              key === 'actual_precision' ? 'Actual Precision' : key,
            ]}
            labelFormatter={(conf: number) => `Mean Conf: ${Number(conf).toFixed(3)}`}
          />
          {/* Perfect calibration diagonal y = x */}
          <ReferenceLine
            segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]}
            stroke="hsl(var(--muted-foreground))"
            strokeDasharray="6 4"
            strokeOpacity={0.5}
          />
          <Line
            type="monotone"
            dataKey="actual_precision"
            name="Actual Precision"
            stroke="#f97316"
            strokeWidth={2}
            dot={{ r: 4, fill: '#f97316', strokeWidth: 0 }}
            activeDot={{ r: 5, fill: '#f97316' }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
// â”€â”€ Full Evaluation Tab â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

interface FullEvalTabProps {
  availableModels: { name: string; framework?: string }[]
}

function FullEvalTab({ availableModels }: FullEvalTabProps) {
  const [model, setModel] = useState<string>(availableModels[0]?.name ?? '')
  const [numFrames, setNumFrames] = useState(200)
  const [frameWidth, setFrameWidth] = useState(640)
  const [frameHeight, setFrameHeight] = useState(480)
  const [runTag, setRunTag] = useState('')
  const [runNotes, setRunNotes] = useState('')

  const [runId, setRunId] = useState<string | null>(null)
  const [result, setResult] = useState<BenchmarkResult | null>(null)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current)
      pollRef.current = null
    }
  }, [])

  const startPolling = useCallback(
    (id: string) => {
      stopPolling()
      pollRef.current = setInterval(async () => {
        try {
          const r = await benchmarkApi.getResult(id)
          setResult(r)
          if (r.status === 'completed' || r.status === 'failed') {
            stopPolling()
            setRunning(false)
            if (r.status === 'failed') {
              setError(r.error_message ?? 'Benchmark failed')
            }
          }
        } catch {
          // transient poll error â€” keep trying
        }
      }, 1000)
    },
    [stopPolling]
  )

  useEffect(() => () => stopPolling(), [stopPolling])

  const handleRun = async () => {
    if (!model) return
    setError(null)
    setResult(null)
    setRunId(null)
    setRunning(true)
    try {
      const resp = await benchmarkApi.start(model, {
        numFrames,
        frameWidth,
        frameHeight,
        runTag: runTag.trim() || undefined,
        runNotes: runNotes.trim() || undefined,
      })
      setRunId(resp.run_id)
      startPolling(resp.run_id)
    } catch (e: unknown) {
      setRunning(false)
      setError(e instanceof Error ? e.message : 'Failed to start benchmark')
    }
  }

  const progress = result?.progress_pct ?? (running ? 5 : 0)
  const ls = result?.latency_stats
  const lm = ls?.latency_ms

  const latencyChartData = lm
    ? [
        { name: 'Mean', ms: lm.mean },
        { name: 'P50', ms: lm.p50 },
        { name: 'P90', ms: lm.p90 },
        { name: 'P95', ms: lm.p95 },
        { name: 'P99', ms: lm.p99 },
      ]
    : []

  const sysChartData =
    result?.cpu_avg != null && result?.memory_avg_mb != null
      ? [{ label: result.model_name, cpu: result.cpu_avg, mem: result.memory_avg_mb }]
      : []

  return (
    <div className="space-y-6">
      {/* Config row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Model */}
        <div className="rounded-lg border border-[hsl(var(--border))] bg-[hsl(var(--card))] p-5">
          <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide mb-3">
            Model
          </p>
          {availableModels.length === 0 ? (
            <p className="text-xs text-muted-foreground">No models loaded yet.</p>
          ) : (
            <select
              value={model}
              onChange={(e) => setModel(e.target.value)}
              className={inputCls}
            >
              {availableModels.map((m) => (
                <option key={m.name} value={m.name}>
                  {m.name}
                </option>
              ))}
            </select>
          )}
        </div>

        {/* Params */}
        <div className="rounded-lg border border-[hsl(var(--border))] bg-[hsl(var(--card))] p-5">
          <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide mb-4">
            Evaluation Parameters
          </p>
          <div className="grid grid-cols-3 gap-3">
            <Field label="Frames">
              <input
                type="number"
                className={inputCls}
                min={10}
                max={2000}
                value={numFrames}
                onChange={(e) => setNumFrames(Number(e.target.value))}
              />
            </Field>
            <Field label="Width px">
              <input
                type="number"
                className={inputCls}
                min={64}
                value={frameWidth}
                onChange={(e) => setFrameWidth(Number(e.target.value))}
              />
            </Field>
            <Field label="Height px">
              <input
                type="number"
                className={inputCls}
                min={64}
                value={frameHeight}
                onChange={(e) => setFrameHeight(Number(e.target.value))}
              />
            </Field>
          </div>
          <div className="mt-3 space-y-2">
            <Field label="Run Tag">
              <input
                type="text"
                className={inputCls}
                maxLength={100}
                placeholder="e.g. baseline, augmented…"
                value={runTag}
                onChange={(e) => setRunTag(e.target.value)}
              />
            </Field>
            <Field label="Notes">
              <textarea
                className={inputCls + ' resize-none'}
                rows={2}
                maxLength={1000}
                placeholder="Optional notes about this run…"
                value={runNotes}
                onChange={(e) => setRunNotes(e.target.value)}
              />
            </Field>
          </div>
        </div>

        {/* Run */}
        <div className="rounded-lg border border-[hsl(var(--border))] bg-[hsl(var(--card))] p-5 flex flex-col justify-between gap-3">
          <div>
            <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide mb-1">
              Run Full Evaluation
            </p>
            <p className="text-xs text-muted-foreground">
              {model
                ? `Evaluate "${model}" over ${numFrames} frames â€” measures fps, latency, precision, recall, f1, cpu & memory.`
                : 'Select a model to start.'}
            </p>
            {error && (
              <div className="mt-2 flex items-center gap-1.5 text-xs text-red-400">
                <AlertCircle className="w-3.5 h-3.5 flex-shrink-0" />
                {error}
              </div>
            )}
          </div>

          <button
            onClick={handleRun}
            disabled={running || !model}
            className="w-full flex items-center justify-center gap-2 rounded bg-primary px-4 py-2.5 text-sm font-semibold text-primary-foreground disabled:opacity-50 hover:opacity-90 transition"
          >
            {running ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Evaluatingâ€¦
              </>
            ) : (
              <>
                <Play className="w-4 h-4" />
                Run Benchmark
              </>
            )}
          </button>
        </div>
      </div>

      {/* Progress bar */}
      {(running || (result && result.status === 'running')) && (
        <div className="rounded-lg border border-[hsl(var(--border))] bg-[hsl(var(--card))] p-5">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs text-muted-foreground uppercase tracking-wide">
              Evaluating framesâ€¦
            </span>
            <span className="text-xs font-mono text-foreground">
              {result?.frames_evaluated ?? 0} / {numFrames} &nbsp;Â·&nbsp;{progress.toFixed(0)}%
            </span>
          </div>
          <div className="h-2 rounded-full bg-slate-800 overflow-hidden">
            <div
              className="h-full rounded-full bg-primary transition-all duration-500"
              style={{ width: `${Math.max(2, progress)}%` }}
            />
          </div>
          {result?.latency_stats && (
            <p className="mt-1.5 text-[10px] text-muted-foreground font-mono">
              live fps ~{result.latency_stats.throughput_fps.toFixed(1)}
            </p>
          )}
        </div>
      )}

      {/* Results */}
      {result && result.status === 'completed' && (
        <div className="space-y-5">
          {/* Metric cards */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <MetricCard
              label="Throughput"
              value={fmt1(ls?.throughput_fps, ' fps')}
              sub={`${result.frames_evaluated ?? numFrames} frames`}
              icon={Zap}
              color="text-emerald-400"
            />
            <MetricCard
              label="Mean Latency"
              value={fmt1(lm?.mean, ' ms')}
              sub={`P95: ${fmt1(lm?.p95, ' ms')}`}
              icon={Clock}
              color="text-blue-400"
            />
            <MetricCard
              label="Precision"
              value={fmtPct(result.precision)}
              sub="TP / (TP + FP)"
              icon={Target}
              color="text-violet-400"
            />
            <MetricCard
              label="Recall"
              value={fmtPct(result.recall)}
              sub="TP / (TP + FN)"
              icon={Target}
              color="text-cyan-400"
            />
            <MetricCard
              label="F1 Score"
              value={fmt2(result.f1 != null ? result.f1 : null)}
              sub="Harmonic mean of P & R"
              icon={BarChart2}
              color="text-amber-400"
            />
            <MetricCard
              label="False Positives"
              value={String(result.false_positives ?? 0)}
              sub={`total across ${result.frames_evaluated ?? numFrames} frames`}
              icon={AlertCircle}
              color="text-red-400"
            />
            <MetricCard
              label="CPU Avg"
              value={fmt1(result.cpu_avg, '%')}
              sub="process cpu usage"
              icon={BarChart2}
              color="text-sky-400"
            />
            <MetricCard
              label="Memory Avg"
              value={fmt1(result.memory_avg_mb, ' MB')}
              sub="process RSS"
              icon={BarChart2}
              color="text-pink-400"
            />
          </div>

          {/* Charts row */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* Latency percentile chart */}
            {latencyChartData.length > 0 && (
              <div className="rounded-lg border border-[hsl(var(--border))] bg-[hsl(var(--card))] p-4">
                <p className="text-sm font-semibold text-foreground mb-4">
                  Latency Distribution (ms)
                </p>
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={latencyChartData} margin={{ top: 4, right: 8, left: -16, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" strokeOpacity={0.4} />
                    <XAxis
                      dataKey="name"
                      tick={{ fontSize: 10, fill: 'hsl(var(--muted-foreground))' }}
                      tickLine={false}
                      axisLine={false}
                    />
                    <YAxis
                      tick={{ fontSize: 10, fill: 'hsl(var(--muted-foreground))' }}
                      tickLine={false}
                      axisLine={false}
                      width={40}
                      unit="ms"
                    />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: 'hsl(var(--card))',
                        border: '1px solid hsl(var(--border))',
                        borderRadius: '6px',
                        fontSize: 11,
                      }}
                      formatter={(v: number) => [`${v.toFixed(2)} ms`, 'Latency']}
                    />
                    <Bar dataKey="ms" fill="#60a5fa" radius={[3, 3, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}

            {/* CPU / Memory chart */}
            {sysChartData.length > 0 && (
              <div className="rounded-lg border border-[hsl(var(--border))] bg-[hsl(var(--card))] p-4">
                <p className="text-sm font-semibold text-foreground mb-4">
                  System Resources (avg)
                </p>
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={sysChartData} margin={{ top: 4, right: 8, left: -16, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" strokeOpacity={0.4} />
                    <XAxis
                      dataKey="label"
                      tick={{ fontSize: 10, fill: 'hsl(var(--muted-foreground))' }}
                      tickLine={false}
                      axisLine={false}
                    />
                    <YAxis
                      tick={{ fontSize: 10, fill: 'hsl(var(--muted-foreground))' }}
                      tickLine={false}
                      axisLine={false}
                      width={44}
                    />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: 'hsl(var(--card))',
                        border: '1px solid hsl(var(--border))',
                        borderRadius: '6px',
                        fontSize: 11,
                      }}
                    />
                    <Legend wrapperStyle={{ fontSize: 11 }} />
                    <Bar dataKey="cpu" name="CPU %" fill="#38bdf8" radius={[3, 3, 0, 0]} />
                    <Bar dataKey="mem" name="Mem MB" fill="#f472b6" radius={[3, 3, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}
          </div>

          {/* PR Curve + Confidence distribution */}
          {(result.pr_curve || result.confidence_histogram) && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <PrCurveChart result={result} />
              <ConfidenceHistogram result={result} />
            </div>
          )}

          {/* Calibration curve */}
          {result.calibration_curve && result.calibration_curve.length >= 2 && (
            <CalibrationCurveChart result={result} />
          )}

          {/* Export buttons */}
          <div className="flex items-center gap-3 flex-wrap">
            <span className="text-[11px] text-muted-foreground uppercase tracking-wide">
              Export:
            </span>
            <a
              href={exportApi.downloadBenchmarkJson(runId!)}
              download
              className="flex items-center gap-1.5 rounded border border-[hsl(var(--border))] px-3 py-1.5 text-xs text-foreground hover:border-primary transition"
            >
              <FileJson className="w-3.5 h-3.5" />
              JSON (this run)
            </a>
            <a
              href={exportApi.downloadBenchmarkHtmlReport(runId!)}
              download
              className="flex items-center gap-1.5 rounded border border-[hsl(var(--border))] px-3 py-1.5 text-xs text-foreground hover:border-primary transition"
            >
              <FileText className="w-3.5 h-3.5" />
              HTML Report
            </a>
            <a
              href={exportApi.downloadBenchmarkAllCsv()}
              download
              className="flex items-center gap-1.5 rounded border border-[hsl(var(--border))] px-3 py-1.5 text-xs text-foreground hover:border-primary transition"
            >
              <Table2 className="w-3.5 h-3.5" />
              CSV (all results)
            </a>
            <a
              href={exportApi.downloadBenchmarkAllJson()}
              download
              className="flex items-center gap-1.5 rounded border border-[hsl(var(--border))] px-3 py-1.5 text-xs text-foreground hover:border-primary transition"
            >
              <Download className="w-3.5 h-3.5" />
              JSON (all results)
            </a>
          </div>
        </div>
      )}
    </div>
  )
}

// â”€â”€ Latency / Compare Tab â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

interface LatencyTabProps {
  availableModels: { name: string; framework?: string }[]
  setHistory: React.Dispatch<React.SetStateAction<BenchmarkResult[]>>
}

function LatencyTab({ availableModels, setHistory }: LatencyTabProps) {
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [params, setParams] = useState({ warmup: 5, measure: 100, w: 640, h: 480 })
  const [running, setRunning] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [comparison, setComparison] = useState<BenchmarkComparison | null>(null)
  const [singleResult, setSingleResult] = useState<BenchmarkResult | null>(null)

  const toggle = (name: string) =>
    setSelected((prev) => {
      const n = new Set(prev)
      n.has(name) ? n.delete(name) : n.add(name)
      return n
    })

  const handleRun = async () => {
    if (selected.size === 0) return setError('Select at least one model.')
    setError(null)
    setRunning(true)
    setComparison(null)
    setSingleResult(null)
    try {
      const names = [...selected]
      const opts = { warmupRuns: params.warmup, measureRuns: params.measure, frameWidth: params.w, frameHeight: params.h }
      if (names.length === 1) {
        const r = await benchmarkApi.run(names[0], opts)
        setSingleResult(r)
        setHistory((prev) => [r, ...prev])
      } else {
        const r = await benchmarkApi.compare(names, opts)
        setComparison(r)
      }
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Benchmark failed')
    } finally {
      setRunning(false)
    }
  }

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Model selection */}
        <div className="rounded-lg border border-[hsl(var(--border))] bg-[hsl(var(--card))] p-5">
          <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide mb-3">
            Select Models
          </p>
          {availableModels.length === 0 ? (
            <p className="text-xs text-muted-foreground">No models loaded yet.</p>
          ) : (
            <div className="space-y-2">
              {availableModels.map((m) => (
                <label key={m.name} className="flex items-center gap-2 cursor-pointer group">
                  <input
                    type="checkbox"
                    className="rounded accent-primary"
                    checked={selected.has(m.name)}
                    onChange={() => toggle(m.name)}
                  />
                  <span className="font-mono text-xs group-hover:text-primary transition">
                    {m.name}
                  </span>
                  <span className="text-[10px] text-muted-foreground ml-auto">{m.framework}</span>
                </label>
              ))}
            </div>
          )}
        </div>

        {/* Params */}
        <div className="rounded-lg border border-[hsl(var(--border))] bg-[hsl(var(--card))] p-5">
          <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide mb-4">
            Parameters
          </p>
          <div className="grid grid-cols-2 gap-3">
            <Field label="Warmup Runs">
              <input type="number" className={inputCls} min={1} value={params.warmup}
                onChange={(e) => setParams({ ...params, warmup: Number(e.target.value) })} />
            </Field>
            <Field label="Measure Runs">
              <input type="number" className={inputCls} min={5} value={params.measure}
                onChange={(e) => setParams({ ...params, measure: Number(e.target.value) })} />
            </Field>
            <Field label="Width px">
              <input type="number" className={inputCls} min={64} value={params.w}
                onChange={(e) => setParams({ ...params, w: Number(e.target.value) })} />
            </Field>
            <Field label="Height px">
              <input type="number" className={inputCls} min={64} value={params.h}
                onChange={(e) => setParams({ ...params, h: Number(e.target.value) })} />
            </Field>
          </div>
        </div>

        {/* Action */}
        <div className="rounded-lg border border-[hsl(var(--border))] bg-[hsl(var(--card))] p-5 flex flex-col justify-between">
          <div>
            <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide mb-2">
              Ready to Run
            </p>
            <p className="text-xs text-muted-foreground">
              {selected.size === 0
                ? 'Select models to benchmark.'
                : selected.size === 1
                  ? `Single model: ${[...selected][0]}`
                  : `Compare ${selected.size} models head-to-head.`}
            </p>
            {error && <p className="mt-2 text-xs text-red-400">{error}</p>}
          </div>
          <button
            onClick={handleRun}
            disabled={running || selected.size === 0}
            className="mt-4 w-full flex items-center justify-center gap-2 rounded bg-primary px-4 py-2.5 text-sm font-semibold text-primary-foreground disabled:opacity-50 hover:opacity-90 transition"
          >
            {running ? (
              <><Loader2 className="w-4 h-4 animate-spin" />Runningâ€¦</>
            ) : (
              <><Play className="w-4 h-4" />Run</>
            )}
          </button>
        </div>
      </div>

      {running && (
        <div className="flex items-center justify-center gap-3 py-10 text-muted-foreground text-sm">
          <Loader2 className="w-5 h-5 animate-spin" />
          Benchmark in progressâ€¦
        </div>
      )}

      {comparison && !running && <BenchmarkReport comparison={comparison} />}
      {singleResult && !running && !comparison && <BenchmarkReport singleResult={singleResult} />}
    </div>
  )
}

// â”€â”€ History table â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

type SortKey = 'fps' | 'f1' | 'auc_pr' | 'precision' | 'recall'
const SORT_LABELS: Record<SortKey, string> = {
  fps: 'FPS',
  f1: 'F1',
  auc_pr: 'AUC-PR',
  precision: 'Precision',
  recall: 'Recall',
}

function HistoryTable({ history }: { history: BenchmarkResult[] }) {
  const [sortKey, setSortKey] = useState<SortKey>('auc_pr')
  const [sortDir, setSortDir] = useState<'desc' | 'asc'>('desc')

  if (history.length === 0) return null

  const fullEvalRuns = history.filter((r) => r.is_full_eval && r.status === 'completed')
  const otherRuns = history.filter((r) => !(r.is_full_eval && r.status === 'completed'))

  const getVal = (r: BenchmarkResult, key: SortKey): number => {
    if (key === 'fps') return r.latency_stats?.throughput_fps ?? r.avg_fps ?? -1
    if (key === 'f1') return r.f1 ?? -1
    if (key === 'auc_pr') return r.auc_pr ?? -1
    if (key === 'precision') return r.precision ?? -1
    if (key === 'recall') return r.recall ?? -1
    return -1
  }

  const sorted = [...fullEvalRuns].sort((a, b) => {
    const diff = getVal(a, sortKey) - getVal(b, sortKey)
    return sortDir === 'desc' ? -diff : diff
  })

  const toggleSort = (key: SortKey) => {
    if (key === sortKey) {
      setSortDir((d) => (d === 'desc' ? 'asc' : 'desc'))
    } else {
      setSortKey(key)
      setSortDir('desc')
    }
  }

  const SortTh = ({ k, label }: { k: SortKey; label: string }) => (
    <th
      className="px-3 py-2 text-left text-[10px] uppercase tracking-wide text-muted-foreground cursor-pointer select-none hover:text-foreground transition whitespace-nowrap"
      onClick={() => toggleSort(k)}
    >
      {label}
      {sortKey === k && <span className="ml-1">{sortDir === 'desc' ? '▼' : '▲'}</span>}
    </th>
  )

  return (
    <div className="space-y-4">
      {sorted.length > 0 && (
        <div className="rounded-lg border border-[hsl(var(--border))] bg-[hsl(var(--card))] overflow-hidden">
          <div className="px-4 py-3 border-b border-[hsl(var(--border))] flex items-center justify-between">
            <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
              Full Evaluation Results
            </p>
            <div className="flex items-center gap-2">
              <span className="text-[10px] text-muted-foreground">Sort by:</span>
              {(Object.keys(SORT_LABELS) as SortKey[]).map((k) => (
                <button
                  key={k}
                  onClick={() => toggleSort(k)}
                  className={cn(
                    'text-[10px] px-2 py-0.5 rounded border transition',
                    sortKey === k
                      ? 'border-primary text-primary'
                      : 'border-[hsl(var(--border))] text-muted-foreground hover:text-foreground'
                  )}
                >
                  {SORT_LABELS[k]}
                </button>
              ))}
              <a
                href={exportApi.downloadBenchmarkAllCsv()}
                download
                className="flex items-center gap-1 text-[10px] text-muted-foreground hover:text-foreground transition"
              >
                <Table2 className="w-3 h-3" /> CSV
              </a>
            </div>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-[hsl(var(--border))]">
                  <th className="px-3 py-2 text-left text-[10px] uppercase tracking-wide text-muted-foreground w-6">#</th>
                  <th className="px-3 py-2 text-left text-[10px] uppercase tracking-wide text-muted-foreground">Model</th>
                  <th className="px-3 py-2 text-left text-[10px] uppercase tracking-wide text-muted-foreground">Tag</th>
                  <SortTh k="fps" label="FPS" />
                  <SortTh k="precision" label="Prec" />
                  <SortTh k="recall" label="Rec" />
                  <SortTh k="f1" label="F1" />
                  <SortTh k="auc_pr" label="AUC-PR" />
                  <th className="px-3 py-2 text-left text-[10px] uppercase tracking-wide text-muted-foreground">Frames</th>
                  <th className="px-3 py-2 text-left text-[10px] uppercase tracking-wide text-muted-foreground">Run ID</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[hsl(var(--border))]">
                {sorted.map((r, idx) => {
                  const fps = r.latency_stats?.throughput_fps ?? r.avg_fps
                  return (
                    <tr key={r.run_id} className="hover:bg-[hsl(var(--muted))]/20 transition">
                      <td className="px-3 py-2 text-muted-foreground font-mono">{idx + 1}</td>
                      <td className="px-3 py-2 font-mono text-foreground">{r.model_name}</td>
                      <td className="px-3 py-2">
                        {r.run_tag ? (
                          <span className="px-1.5 py-0.5 rounded bg-violet-950/40 text-violet-300 text-[10px]">{r.run_tag}</span>
                        ) : (
                          <span className="text-muted-foreground">—</span>
                        )}
                      </td>
                      <td className="px-3 py-2 font-mono text-sky-400">{fps != null ? fps.toFixed(1) : '—'}</td>
                      <td className="px-3 py-2 font-mono">{fmtPct(r.precision)}</td>
                      <td className="px-3 py-2 font-mono">{fmtPct(r.recall)}</td>
                      <td className={cn('px-3 py-2 font-mono', r.f1 != null && r.f1 > 0.7 ? 'text-amber-400' : '')}>
                        {fmtPct(r.f1)}
                      </td>
                      <td className={cn('px-3 py-2 font-mono', r.auc_pr != null && r.auc_pr > 0.7 ? 'text-emerald-400' : '')}>
                        {r.auc_pr != null ? r.auc_pr.toFixed(3) : '—'}
                      </td>
                      <td className="px-3 py-2 font-mono text-muted-foreground">{r.frames_evaluated ?? '—'}</td>
                      <td className="px-3 py-2 font-mono text-[10px] text-muted-foreground">{r.run_id.slice(0, 8)}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Non-full-eval runs (latency benchmarks) */}
      {otherRuns.length > 0 && (
        <div className="rounded-lg border border-[hsl(var(--border))] bg-[hsl(var(--card))] overflow-hidden">
          <div className="px-4 py-3 border-b border-[hsl(var(--border))]">
            <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide">Latency Benchmarks</p>
          </div>
          <div className="divide-y divide-[hsl(var(--border))]">
            {otherRuns.slice(0, 10).map((r) => (
              <div key={r.run_id} className="flex items-center justify-between px-4 py-2.5 text-xs">
                <div className="flex items-center gap-3">
                  <span className="font-mono text-foreground">{r.model_name}</span>
                  <span
                    className={cn(
                      'px-1.5 py-0.5 rounded-full text-[10px] font-medium',
                      r.status === 'completed'
                        ? 'bg-emerald-400/10 text-emerald-400'
                        : r.status === 'running'
                          ? 'bg-blue-400/10 text-blue-400'
                          : 'bg-red-400/10 text-red-400'
                    )}
                  >
                    {r.status}
                  </span>
                </div>
                <div className="flex items-center gap-4 text-muted-foreground font-mono">
                  {r.latency_stats && (
                    <>
                      <span>{r.latency_stats.throughput_fps.toFixed(1)} fps</span>
                      <span>p50 {r.latency_stats.latency_ms.p50.toFixed(1)}ms</span>
                    </>
                  )}
                  <span className="text-[10px]">{r.run_id.slice(0, 8)}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

// â”€â”€ Page â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

// ── Comparative Evaluation Tab ──────────────────────────────────────────────

interface ComparativeTabProps {
  availableModels: { name: string; framework?: string }[]
}

function ComparativeTab({ availableModels }: ComparativeTabProps) {
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [params, setParams] = useState({ numFrames: 200, w: 640, h: 480 })
  const [runTag, setRunTag] = useState('')
  const [runId, setRunId] = useState<string | null>(null)
  const [result, setResult] = useState<ComparativeResult | null>(null)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const stopPolling = useCallback(() => {
    if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null }
  }, [])

  const startPolling = useCallback((id: string) => {
    stopPolling()
    pollRef.current = setInterval(async () => {
      try {
        const r = await comparativeApi.getResult(id)
        setResult(r)
        if (r.status === 'completed' || r.status === 'failed') {
          stopPolling()
          setRunning(false)
          if (r.status === 'failed') setError(r.error_message ?? 'Comparative benchmark failed')
        }
      } catch { /* ignore poll errors */ }
    }, 1500)
  }, [stopPolling])

  useEffect(() => () => stopPolling(), [stopPolling])

  const toggle = (name: string) =>
    setSelected(prev => { const n = new Set(prev); n.has(name) ? n.delete(name) : n.add(name); return n })

  const handleRun = async () => {
    if (selected.size < 2) return setError('Select at least 2 models.')
    setError(null); setResult(null); setRunId(null); setRunning(true)
    try {
      const resp = await comparativeApi.start([...selected], {
        numFrames: params.numFrames,
        frameWidth: params.w,
        frameHeight: params.h,
        runTag: runTag.trim() || undefined,
      })
      setRunId(resp.run_id)
      startPolling(resp.run_id)
    } catch (e: unknown) {
      setRunning(false)
      setError(e instanceof Error ? e.message : 'Failed to start comparative benchmark')
    }
  }

  const progress = result?.progress_pct ?? (running ? 5 : 0)
  const da = result?.disagreement_analysis
  const COLORS = ['#38bdf8', '#a78bfa', '#34d399', '#f59e0b']

  return (
    <div className="space-y-6">
      {/* Config */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Model selection */}
        <div className="rounded-lg border border-[hsl(var(--border))] bg-[hsl(var(--card))] p-5">
          <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide mb-1">Select Models</p>
          <p className="text-[10px] text-muted-foreground mb-3">Choose 2 or more for same-frame comparison</p>
          {availableModels.length === 0 ? (
            <p className="text-xs text-muted-foreground">No models loaded yet.</p>
          ) : (
            <div className="space-y-2">
              {availableModels.map(m => (
                <label key={m.name} className="flex items-center gap-2 cursor-pointer group">
                  <input
                    type="checkbox"
                    className="rounded accent-primary"
                    checked={selected.has(m.name)}
                    onChange={() => toggle(m.name)}
                  />
                  <span className="font-mono text-xs group-hover:text-primary transition">{m.name}</span>
                  <span className="text-[10px] text-muted-foreground ml-auto">{m.framework}</span>
                </label>
              ))}
            </div>
          )}
        </div>

        {/* Params */}
        <div className="rounded-lg border border-[hsl(var(--border))] bg-[hsl(var(--card))] p-5">
          <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide mb-4">Parameters</p>
          <div className="grid grid-cols-3 gap-3">
            <Field label="Frames">
              <input type="number" className={inputCls} min={10} max={2000}
                value={params.numFrames}
                onChange={e => setParams({ ...params, numFrames: Number(e.target.value) })} />
            </Field>
            <Field label="Width px">
              <input type="number" className={inputCls} min={64}
                value={params.w}
                onChange={e => setParams({ ...params, w: Number(e.target.value) })} />
            </Field>
            <Field label="Height px">
              <input type="number" className={inputCls} min={64}
                value={params.h}
                onChange={e => setParams({ ...params, h: Number(e.target.value) })} />
            </Field>
          </div>
          <div className="mt-3">
            <Field label="Run Tag">
              <input type="text" className={inputCls} maxLength={100}
                placeholder="e.g. experiment-v2" value={runTag}
                onChange={e => setRunTag(e.target.value)} />
            </Field>
          </div>
        </div>

        {/* Run */}
        <div className="rounded-lg border border-[hsl(var(--border))] bg-[hsl(var(--card))] p-5 flex flex-col justify-between gap-3">
          <div>
            <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide mb-1">Same-Frame Comparison</p>
            <p className="text-xs text-muted-foreground">
              {selected.size < 2
                ? 'Select at least 2 models.'
                : `Compare ${[...selected].join(' vs ')} on ${params.numFrames} identical frames.`}
            </p>
            {error && (
              <div className="mt-2 flex items-start gap-1.5 text-xs text-red-400">
                <AlertCircle className="w-3.5 h-3.5 flex-shrink-0 mt-0.5" />{error}
              </div>
            )}
          </div>
          <button
            onClick={handleRun}
            disabled={running || selected.size < 2}
            className="w-full flex items-center justify-center gap-2 rounded bg-primary px-4 py-2.5 text-sm font-semibold text-primary-foreground disabled:opacity-50 hover:opacity-90 transition"
          >
            {running
              ? <><Loader2 className="w-4 h-4 animate-spin" />Running…</>
              : <><Layers className="w-4 h-4" />Compare Models</>}
          </button>
        </div>
      </div>

      {/* Progress bar */}
      {(running || result?.status === 'running') && (
        <div className="rounded-lg border border-[hsl(var(--border))] bg-[hsl(var(--card))] p-5">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs text-muted-foreground uppercase tracking-wide">Comparing models…</span>
            <span className="text-xs font-mono text-foreground">{progress.toFixed(0)}%</span>
          </div>
          <div className="h-2 rounded-full bg-slate-800 overflow-hidden">
            <div
              className="h-full rounded-full bg-primary transition-all duration-500"
              style={{ width: `${Math.max(2, progress)}%` }}
            />
          </div>
          <p className="mt-1.5 text-[10px] text-muted-foreground font-mono">
            {result?.frames_evaluated ?? 0} frame-model evaluations complete
            {runId && <span className="ml-2 opacity-60">{runId.slice(0, 8)}</span>}
          </p>
        </div>
      )}

      {/* Results */}
      {result?.status === 'completed' && result.model_stats && da && (
        <div className="space-y-5">

          {/* Per-model stats table */}
          <div className="rounded-lg border border-[hsl(var(--border))] bg-[hsl(var(--card))] overflow-hidden">
            <div className="px-4 py-3 border-b border-[hsl(var(--border))]">
              <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Model Performance Comparison</p>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-[hsl(var(--border))]">
                    {['Model','FPS','Precision','Recall','F1','AUC-PR','TP','FP','FN'].map(h => (
                      <th key={h} className="px-3 py-2 text-left text-[10px] uppercase tracking-wide text-muted-foreground">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-[hsl(var(--border))]">
                  {result.model_names.map((name, idx) => {
                    const s = result.model_stats![name]
                    if (!s) return null
                    const rival = result.model_stats![result.model_names[1 - idx]]
                    const better = (key: keyof typeof s) =>
                      result.model_names.length === 2 && rival != null
                        ? (s[key] as number) > (rival[key] as number)
                        : false
                    return (
                      <tr key={name} className="hover:bg-[hsl(var(--muted))]/20 transition">
                        <td className="px-3 py-2.5 font-mono text-foreground font-medium">
                          {name}
                          <span className="ml-1.5 text-[9px] uppercase" style={{ color: COLORS[idx] }}>
                            {String.fromCharCode(65 + idx)}
                          </span>
                        </td>
                        <td className={cn('px-3 py-2.5 font-mono', better('avg_fps') ? 'text-emerald-400' : '')}>{fmt1(s.avg_fps)}</td>
                        <td className={cn('px-3 py-2.5 font-mono', better('precision') ? 'text-emerald-400' : '')}>{fmtPct(s.precision)}</td>
                        <td className={cn('px-3 py-2.5 font-mono', better('recall') ? 'text-emerald-400' : '')}>{fmtPct(s.recall)}</td>
                        <td className={cn('px-3 py-2.5 font-mono', better('f1') ? 'text-amber-400' : '')}>{fmtPct(s.f1)}</td>
                        <td className={cn('px-3 py-2.5 font-mono', better('auc_pr') ? 'text-emerald-400' : '')}>
                          {s.auc_pr != null ? s.auc_pr.toFixed(3) : '—'}
                        </td>
                        <td className="px-3 py-2.5 font-mono text-muted-foreground">{s.tp}</td>
                        <td className="px-3 py-2.5 font-mono text-red-400">{s.fp}</td>
                        <td className="px-3 py-2.5 font-mono text-orange-400">{s.fn}</td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {/* Disagreement analysis */}
          <div className="rounded-lg border border-[hsl(var(--border))] bg-[hsl(var(--card))] p-4">
            <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground mb-3">
              Disagreement Analysis · {da.total_frames} frames · {da.model_a} vs {da.model_b}
            </p>
            <div className="flex flex-wrap gap-2 mb-4">
              {([
                { label: 'Disagreement', val: da.disagreement_rate, color: 'text-red-400 bg-red-950/40' },
                { label: 'Agreement',    val: da.agreement_rate,    color: 'text-emerald-400 bg-emerald-950/40' },
                { label: 'A-only',       val: da.a_only_rate,       color: 'text-sky-400 bg-sky-950/40' },
                { label: 'B-only',       val: da.b_only_rate,       color: 'text-violet-400 bg-violet-950/40' },
                { label: 'Mean F1 Δ',    val: da.mean_f1_delta,     color: da.mean_f1_delta >= 0 ? 'text-amber-400 bg-amber-950/40' : 'text-rose-400 bg-rose-950/40', raw: true },
                { label: 'Mean Conf Δ',  val: da.mean_conf_delta,   color: 'text-slate-300 bg-slate-800', raw: true },
              ] as { label: string; val: number; color: string; raw?: boolean }[]).map(({ label, val, color, raw }) => (
                <div key={label} className={cn('px-3 py-1.5 rounded-full text-[11px] font-mono font-medium', color)}>
                  {label}: {raw ? ((val >= 0 ? '+' : '') + val.toFixed(4)) : fmtPct(val)}
                </div>
              ))}
            </div>
            <ResponsiveContainer width="100%" height={180}>
              <BarChart
                data={[
                  { name: 'Both Detect', value: da.both_detect_rate },
                  { name: 'Both Miss',   value: da.both_miss_rate },
                  { name: `${da.model_a} Only`, value: da.a_only_rate },
                  { name: `${da.model_b} Only`, value: da.b_only_rate },
                ]}
                margin={{ top: 0, right: 8, left: -24, bottom: 0 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" strokeOpacity={0.4} />
                <XAxis dataKey="name" tick={{ fontSize: 9, fill: 'hsl(var(--muted-foreground))' }} tickLine={false} axisLine={false} />
                <YAxis
                  tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
                  tick={{ fontSize: 9, fill: 'hsl(var(--muted-foreground))' }}
                  tickLine={false} axisLine={false}
                />
                <Tooltip
                  formatter={(v: unknown) => [`${((v as number) * 100).toFixed(1)}%`, 'Rate']}
                  contentStyle={{ backgroundColor: 'hsl(var(--card))', border: '1px solid hsl(var(--border))', borderRadius: '6px', fontSize: 11 }}
                />
                <Bar dataKey="value" radius={[3, 3, 0, 0]}>
                  {['#34d399', '#64748b', '#38bdf8', '#a78bfa'].map((fill, i) => (
                    <Cell key={i} fill={fill} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Per-frame F1 delta line chart */}
          {result.per_frame_data && result.per_frame_data.length > 0 && (
            <div className="rounded-lg border border-[hsl(var(--border))] bg-[hsl(var(--card))] p-4">
              <div className="flex flex-wrap items-start justify-between gap-2 mb-1">
                <p className="text-sm font-semibold text-foreground">Per-Frame F1 Delta</p>
                <span className="text-[11px] font-mono px-2 py-0.5 rounded-full bg-slate-800 text-muted-foreground">
                  {da.model_a} − {da.model_b}
                </span>
              </div>
              <p className="text-[10px] text-muted-foreground mb-3">
                Positive = model A better on that frame · near-zero = similar performance
              </p>
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={result.per_frame_data} margin={{ top: 4, right: 8, left: -16, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" strokeOpacity={0.4} />
                  <XAxis
                    dataKey="frame_idx"
                    tick={{ fontSize: 9, fill: 'hsl(var(--muted-foreground))' }}
                    tickLine={false} axisLine={false}
                    label={{ value: 'Frame', position: 'insideBottom', offset: -8, fontSize: 10, fill: 'hsl(var(--muted-foreground))' }}
                  />
                  <YAxis
                    domain={[-1, 1]}
                    tick={{ fontSize: 9, fill: 'hsl(var(--muted-foreground))' }}
                    tickLine={false} axisLine={false} width={36}
                    label={{ value: 'F1 Δ', angle: -90, position: 'insideLeft', offset: 14, fontSize: 10, fill: 'hsl(var(--muted-foreground))' }}
                  />
                  <Tooltip
                    formatter={(v: unknown) => [Number(v).toFixed(3), 'F1 delta']}
                    contentStyle={{ backgroundColor: 'hsl(var(--card))', border: '1px solid hsl(var(--border))', borderRadius: '6px', fontSize: 11 }}
                  />
                  <ReferenceLine y={0} stroke="hsl(var(--muted-foreground))" strokeDasharray="5 4" strokeOpacity={0.5} />
                  <Line type="monotone" dataKey="f1_delta" stroke="#f59e0b" strokeWidth={1.5} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* PR curve overlay */}
          {result.pr_curves && Object.keys(result.pr_curves).length >= 2 && (
            <div className="rounded-lg border border-[hsl(var(--border))] bg-[hsl(var(--card))] p-4">
              <p className="text-sm font-semibold text-foreground mb-1">PR Curve Overlay</p>
              <p className="text-[10px] text-muted-foreground mb-3">Both models evaluated on identical frames</p>
              <ResponsiveContainer width="100%" height={220}>
                <LineChart margin={{ top: 8, right: 16, left: -8, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" strokeOpacity={0.4} />
                  <XAxis
                    dataKey="recall" type="number" domain={[0, 1]} tickCount={6}
                    tick={{ fontSize: 10, fill: 'hsl(var(--muted-foreground))' }}
                    tickLine={false} axisLine={false}
                    label={{ value: 'Recall', position: 'insideBottom', offset: -12, fontSize: 11, fill: 'hsl(var(--muted-foreground))' }}
                  />
                  <YAxis
                    domain={[0, 1]} tickCount={6}
                    tick={{ fontSize: 10, fill: 'hsl(var(--muted-foreground))' }}
                    tickLine={false} axisLine={false} width={36}
                    label={{ value: 'Precision', angle: -90, position: 'insideLeft', offset: 14, fontSize: 11, fill: 'hsl(var(--muted-foreground))' }}
                  />
                  <Tooltip
                    formatter={(v: unknown, key: string) => [Number(v).toFixed(3), key]}
                    contentStyle={{ backgroundColor: 'hsl(var(--card))', border: '1px solid hsl(var(--border))', borderRadius: '6px', fontSize: 11 }}
                  />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  {Object.entries(result.pr_curves).map(([name, curve], ci) => {
                    const sorted = [...curve].sort((a, b) => a.recall - b.recall)
                    return (
                      <Line
                        key={name}
                        data={sorted}
                        type="monotone"
                        dataKey="precision"
                        name={name}
                        stroke={COLORS[ci % COLORS.length]}
                        strokeWidth={2}
                        dot={false}
                      />
                    )
                  })}
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Confidence shift histogram */}
          {result.conf_shift_histogram && (
            <div className="rounded-lg border border-[hsl(var(--border))] bg-[hsl(var(--card))] p-4">
              <p className="text-sm font-semibold text-foreground mb-1">Confidence Shift Distribution</p>
              <p className="text-[10px] text-muted-foreground mb-3">
                Per-frame mean confidence: {result.conf_shift_histogram.model_a} − {result.conf_shift_histogram.model_b}
              </p>
              <ResponsiveContainer width="100%" height={160}>
                <BarChart
                  data={result.conf_shift_histogram.bins.slice(0, -1).map((b: number, i: number) => ({
                    bin: b.toFixed(2),
                    count: result.conf_shift_histogram!.counts[i],
                  }))}
                  margin={{ top: 0, right: 8, left: -16, bottom: 0 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" strokeOpacity={0.4} />
                  <XAxis dataKey="bin" tick={{ fontSize: 9, fill: 'hsl(var(--muted-foreground))' }} tickLine={false} axisLine={false} interval={4} />
                  <YAxis tick={{ fontSize: 9, fill: 'hsl(var(--muted-foreground))' }} tickLine={false} axisLine={false} width={32} />
                  <Tooltip
                    formatter={(v: unknown) => [v as string | number, 'Frames']}
                    contentStyle={{ backgroundColor: 'hsl(var(--card))', border: '1px solid hsl(var(--border))', borderRadius: '6px', fontSize: 11 }}
                  />
                  <Bar dataKey="count" fill="#38bdf8" radius={[2, 2, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Reproducibility footer */}
          {da.dataset_hash && (
            <div className="rounded-lg border border-[hsl(var(--border))] bg-[hsl(var(--card))] p-4">
              <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground mb-2">Reproducibility</p>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-[11px] font-mono">
                <div><span className="text-muted-foreground">Dataset hash</span><br /><span className="text-foreground">{da.dataset_hash}</span></div>
                {da.git_commit && <div><span className="text-muted-foreground">Git commit</span><br /><span className="text-foreground">{da.git_commit.slice(0, 12)}</span></div>}
                <div><span className="text-muted-foreground">Frames</span><br /><span className="text-foreground">{result.num_frames}</span></div>
                <div><span className="text-muted-foreground">Run ID</span><br /><span className="text-foreground">{result.run_id.slice(0, 12)}</span></div>
              </div>
            </div>
          )}

        </div>
      )}
    </div>
  )
}

export function Benchmark() {
  const availableModels = useModelStore((s) => s.availableModels)
  const [activeTab, setActiveTab] = useState<'full' | 'latency' | 'comparative'>('full')
  const [history, setHistory] = useState<BenchmarkResult[]>([])

  useEffect(() => {
    benchmarkApi.listResults().then(setHistory).catch(() => {})
  }, [])

  const tabs = [
    { id: 'full' as const, label: 'Full Evaluation', icon: Target },
    { id: 'latency' as const, label: 'Latency / Compare', icon: BarChart2 },
    { id: 'comparative' as const, label: 'Comparative Eval', icon: Layers },
  ]

  return (
    <div className="space-y-6">
      {/* Tab bar */}
      <div className="flex gap-1 rounded-lg border border-[hsl(var(--border))] bg-[hsl(var(--card))] p-1 w-fit">
        {tabs.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            onClick={() => setActiveTab(id)}
            className={cn(
              'flex items-center gap-2 rounded px-4 py-2 text-sm font-medium transition',
              activeTab === id
                ? 'bg-primary text-primary-foreground'
                : 'text-muted-foreground hover:text-foreground'
            )}
          >
            <Icon className="w-3.5 h-3.5" />
            {label}
          </button>
        ))}
      </div>

      {/* Active tab content */}
      {activeTab === 'full' ? (
        <FullEvalTab availableModels={availableModels} />
      ) : activeTab === 'latency' ? (
        <LatencyTab
          availableModels={availableModels}
          setHistory={setHistory}
        />
      ) : (
        <ComparativeTab availableModels={availableModels} />
      )}

      {/* History always visible */}
      <HistoryTable history={history} />
    </div>
  )
}



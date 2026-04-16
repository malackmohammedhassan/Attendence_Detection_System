import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts'
import { Trophy, Clock, Zap, AlertCircle, Download } from 'lucide-react'
import { cn, formatMs, formatFps } from '@/lib/utils'
import { exportApi } from '@/services/api'
import type { BenchmarkComparison, BenchmarkResult } from '@/types'

const MODEL_COLORS = ['#60a5fa', '#34d399', '#f59e0b', '#a78bfa', '#f87171']

// ── Single model result card ─────────────────────────────────────────────────

interface ModelResultCardProps {
  result: BenchmarkResult
  rank?: number
}

function ModelResultCard({ result, rank }: ModelResultCardProps) {
  const stats = result.latency_stats

  if (result.status === 'failed') {
    return (
      <div className="rounded-lg border border-red-900/50 bg-red-950/20 p-4">
        <div className="flex items-center gap-2 text-red-400 mb-1">
          <AlertCircle className="w-4 h-4" />
          <span className="font-semibold">{result.model_name}</span>
        </div>
        <p className="text-xs text-red-400/70">{result.error_message ?? 'Benchmark failed'}</p>
      </div>
    )
  }

  return (
    <div className="rounded-lg border border-[hsl(var(--border))] bg-[hsl(var(--card))] p-4">
      {/* Model name + rank */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          {rank === 1 && <Trophy className="w-4 h-4 text-amber-400" />}
          <span className="font-semibold text-sm text-foreground">{result.model_name}</span>
          {rank && (
            <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-slate-800 text-muted-foreground">
              #{rank}
            </span>
          )}
        </div>
        <a
          href={exportApi.downloadBenchmarkJson(result.run_id)}
          download
          className="p-1 text-muted-foreground hover:text-foreground transition-colors"
          title="Download result JSON"
        >
          <Download className="w-3.5 h-3.5" />
        </a>
      </div>

      {/* Stats grid */}
      {stats && (
        <div className="grid grid-cols-2 gap-2">
          <Stat
            icon={Zap}
            label="Throughput"
            value={formatFps(stats.throughput_fps)}
            color="text-emerald-400"
          />
          <Stat
            icon={Clock}
            label="Mean Latency"
            value={formatMs(stats.latency_ms.mean)}
            color="text-blue-400"
          />
          <Stat label="P95" value={formatMs(stats.latency_ms.p95)} />
          <Stat label="P99" value={formatMs(stats.latency_ms.p99)} />
          <Stat label="Min" value={formatMs(stats.latency_ms.min)} />
          <Stat label="Max" value={formatMs(stats.latency_ms.max)} />
          <Stat
            label="Std Dev"
            value={formatMs(stats.latency_ms.stdev)}
            className="col-span-2"
          />
        </div>
      )}

      {/* Full-eval metrics */}
      {result.is_full_eval && (result.precision != null || result.f1 != null) && (
        <div className="grid grid-cols-2 gap-2 mt-2 pt-2 border-t border-[hsl(var(--border))]">
          {result.precision != null && (
            <Stat label="Precision" value={`${(result.precision * 100).toFixed(1)}%`} color="text-violet-400" />
          )}
          {result.recall != null && (
            <Stat label="Recall" value={`${(result.recall * 100).toFixed(1)}%`} color="text-cyan-400" />
          )}
          {result.f1 != null && (
            <Stat label="F1 Score" value={result.f1.toFixed(3)} color="text-amber-400" />
          )}
          {result.false_positives != null && (
            <Stat label="False Pos." value={String(result.false_positives)} color="text-red-400" />
          )}
          {result.cpu_avg != null && (
            <Stat label="CPU Avg" value={`${result.cpu_avg.toFixed(1)}%`} color="text-sky-400" />
          )}
          {result.memory_avg_mb != null && (
            <Stat label="Mem Avg" value={`${result.memory_avg_mb.toFixed(0)} MB`} color="text-pink-400" />
          )}
        </div>
      )}

      {result.duration_sec !== null && (
        <p className="text-[10px] text-muted-foreground mt-2">
          Completed in {result.duration_sec?.toFixed(1)}s
          {result.latency_stats && ` · ${result.latency_stats.measure_runs} samples`}
        </p>
      )}
    </div>
  )
}

interface StatProps {
  icon?: React.ElementType
  label: string
  value: string
  color?: string
  className?: string
}

function Stat({ icon: Icon, label, value, color, className }: StatProps) {
  return (
    <div className={cn('bg-slate-900/50 rounded p-2', className)}>
      <div className="flex items-center gap-1 mb-0.5">
        {Icon && <Icon className={cn('w-3 h-3', color ?? 'text-muted-foreground')} />}
        <span className="text-[10px] text-muted-foreground">{label}</span>
      </div>
      <p className={cn('text-sm font-mono font-semibold', color ?? 'text-foreground')}>{value}</p>
    </div>
  )
}

// ── Comparison bar chart ─────────────────────────────────────────────────────

interface ComparisonBarChartProps {
  comparison: BenchmarkComparison
}

function ComparisonBarChart({ comparison }: ComparisonBarChartProps) {
  const chartData = Object.entries(comparison.results)
    .filter(([, r]) => 'latency_stats' in r && r.latency_stats)
    .map(([name, r]) => {
      const result = r as BenchmarkResult
      return {
        name,
        fps: result.latency_stats?.throughput_fps ?? 0,
        p50: result.latency_stats?.latency_ms.p50 ?? 0,
        p95: result.latency_stats?.latency_ms.p95 ?? 0,
        p99: result.latency_stats?.latency_ms.p99 ?? 0,
      }
    })

  return (
    <div className="rounded-lg border border-[hsl(var(--border))] bg-[hsl(var(--card))] p-4">
      <p className="text-sm font-semibold text-foreground mb-4">Latency Comparison (ms)</p>
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={chartData} margin={{ top: 4, right: 8, left: -16, bottom: 0 }}>
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
            width={36}
            unit="ms"
          />
          <Tooltip
            contentStyle={{
              backgroundColor: 'hsl(var(--card))',
              border: '1px solid hsl(var(--border))',
              borderRadius: '6px',
              fontSize: 11,
            }}
            formatter={(v: number, key: string) => [`${v.toFixed(2)}ms`, key.toUpperCase()]}
          />
          <Legend
            wrapperStyle={{ fontSize: 11, color: 'hsl(var(--muted-foreground))' }}
          />
          <Bar dataKey="p50" name="P50" fill="#60a5fa" radius={[2, 2, 0, 0]} />
          <Bar dataKey="p95" name="P95" fill="#f59e0b" radius={[2, 2, 0, 0]} />
          <Bar dataKey="p99" name="P99" fill="#f87171" radius={[2, 2, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>

      {/* FPS ranking badge row */}
      <div className="flex items-center gap-2 mt-3 flex-wrap">
        <span className="text-[10px] text-muted-foreground uppercase tracking-wide">FPS Rank:</span>
        {comparison.ranking_by_fps.map((r, i) => (
          <div
            key={r.model}
            className="flex items-center gap-1 px-2 py-0.5 rounded-full text-xs"
            style={{ backgroundColor: `${MODEL_COLORS[i]}20`, color: MODEL_COLORS[i] }}
          >
            {i === 0 && <Trophy className="w-3 h-3" />}
            {r.model}: {formatFps(r.fps)}
          </div>
        ))}
      </div>
    </div>
  )
}

// ── Main BenchmarkReport ─────────────────────────────────────────────────────

interface BenchmarkReportProps {
  comparison?: BenchmarkComparison | null
  singleResult?: BenchmarkResult | null
  className?: string
}

export function BenchmarkReport({ comparison, singleResult, className }: BenchmarkReportProps) {
  if (!comparison && !singleResult) {
    return (
      <div className={cn('rounded-lg border border-[hsl(var(--border))] bg-[hsl(var(--card))] p-8', className)}>
        <div className="flex flex-col items-center gap-2 text-muted-foreground">
          <Zap className="w-8 h-8 opacity-40" />
          <p className="text-sm">No benchmark results yet</p>
          <p className="text-xs">Run a benchmark to see results here</p>
        </div>
      </div>
    )
  }

  if (singleResult) {
    return (
      <div className={className}>
        <ModelResultCard result={singleResult} />
      </div>
    )
  }

  if (!comparison) return null

  return (
    <div className={cn('space-y-4', className)}>
      {/* Comparison chart */}
      {Object.values(comparison.results).some(
        (r) => 'latency_stats' in r && r.latency_stats
      ) && <ComparisonBarChart comparison={comparison} />}

      {/* Per-model cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {comparison.ranking_by_fps.map((ranked) => {
          const result = comparison.results[ranked.model]
          if (!result || !('run_id' in result)) return null
          return (
            <ModelResultCard
              key={ranked.model}
              result={result as BenchmarkResult}
              rank={ranked.rank}
            />
          )
        })}
      </div>
    </div>
  )
}

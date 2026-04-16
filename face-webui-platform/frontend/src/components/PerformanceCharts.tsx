import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts'
import { Activity, Cpu, MemoryStick, Zap } from 'lucide-react'
import { cn, formatPercent, formatFps, formatMs } from '@/lib/utils'
import { useMetricsStore } from '@/store/useMetricsStore'
import type { ChartPoint } from '@/types'

// ── Reusable mini chart ──────────────────────────────────────────────────────

interface MiniChartProps {
  data: ChartPoint[]
  color: string
  unit: string
  referenceValue?: number
  yDomain?: [number, number]
}

function MiniChart({ data, color, unit, referenceValue, yDomain }: MiniChartProps) {
  return (
    <ResponsiveContainer width="100%" height={80}>
      <LineChart data={data} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" strokeOpacity={0.4} />
        <XAxis
          dataKey="time"
          tick={{ fontSize: 9, fill: 'hsl(var(--muted-foreground))' }}
          tickLine={false}
          axisLine={false}
          interval="preserveStartEnd"
        />
        <YAxis
          tick={{ fontSize: 9, fill: 'hsl(var(--muted-foreground))' }}
          tickLine={false}
          axisLine={false}
          domain={yDomain ?? ['auto', 'auto']}
          width={32}
        />
        <Tooltip
          contentStyle={{
            backgroundColor: 'hsl(var(--card))',
            border: '1px solid hsl(var(--border))',
            borderRadius: '6px',
            fontSize: 11,
            color: 'hsl(var(--foreground))',
          }}
          labelStyle={{ color: 'hsl(var(--muted-foreground))' }}
          formatter={(v: number) => [`${v.toFixed(2)} ${unit}`, '']}
        />
        {referenceValue !== undefined && (
          <ReferenceLine
            y={referenceValue}
            stroke={color}
            strokeDasharray="4 2"
            strokeOpacity={0.5}
          />
        )}
        <Line
          type="monotone"
          dataKey="value"
          stroke={color}
          strokeWidth={1.5}
          dot={false}
          activeDot={{ r: 3, fill: color }}
          isAnimationActive={false}
        />
      </LineChart>
    </ResponsiveContainer>
  )
}

// ── Stat card ────────────────────────────────────────────────────────────────

interface StatCardProps {
  icon: React.ElementType
  label: string
  value: string
  subtitle?: string
  trend?: 'up' | 'down' | 'neutral'
  color: string
  children?: React.ReactNode
  className?: string
}

function StatCard({
  icon: Icon,
  label,
  value,
  subtitle,
  color,
  children,
  className,
}: StatCardProps) {
  return (
    <div className={cn('rounded-lg border border-[hsl(var(--border))] bg-[hsl(var(--card))] p-4', className)}>
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <Icon className={cn('w-4 h-4', color)} />
          <span className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
            {label}
          </span>
        </div>
        <div className="text-right">
          <p className="text-xl font-bold font-mono text-foreground leading-none">{value}</p>
          {subtitle && (
            <p className="text-[10px] text-muted-foreground mt-0.5">{subtitle}</p>
          )}
        </div>
      </div>
      {children}
    </div>
  )
}

// ── Main component ───────────────────────────────────────────────────────────

interface PerformanceChartsProps {
  className?: string
}

export function PerformanceCharts({ className }: PerformanceChartsProps) {
  const snap = useMetricsStore((s) => s.latestSnapshot)
  const fpsHistory = useMetricsStore((s) => s.fpsHistory)
  const latencyHistory = useMetricsStore((s) => s.latencyHistory)
  const cpuHistory = useMetricsStore((s) => s.cpuHistory)
  const memoryHistory = useMetricsStore((s) => s.memoryHistory)

  const fps = snap?.inference.fps ?? 0
  const latency = latencyHistory.length > 0
    ? latencyHistory[latencyHistory.length - 1]?.value ?? 0
    : 0
  const cpu = snap?.cpu.overall_percent ?? 0
  const memPct = snap?.memory.percent ?? 0
  const memUsed = snap
    ? `${(snap.memory.used_mb / 1024).toFixed(1)}GB / ${(snap.memory.total_mb / 1024).toFixed(1)}GB`
    : '—'

  return (
    <div className={cn('grid grid-cols-1 sm:grid-cols-2 gap-4', className)}>
      {/* FPS */}
      <StatCard
        icon={Zap}
        label="FPS"
        value={formatFps(fps)}
        subtitle={`${snap?.inference.total_frames?.toLocaleString() ?? 0} total frames`}
        color="text-emerald-400"
      >
        <MiniChart data={fpsHistory} color="#34d399" unit="fps" yDomain={[0, 60]} />
      </StatCard>

      {/* Latency */}
      <StatCard
        icon={Activity}
        label="Latency"
        value={latency > 0 ? formatMs(latency) : '—'}
        subtitle="per inference"
        color="text-blue-400"
      >
        <MiniChart data={latencyHistory} color="#60a5fa" unit="ms" />
      </StatCard>

      {/* CPU */}
      <StatCard
        icon={Cpu}
        label="CPU"
        value={formatPercent(cpu)}
        subtitle={`Process: ${formatPercent(snap?.cpu.process_percent ?? 0)}`}
        color="text-amber-400"
      >
        <MiniChart data={cpuHistory} color="#fbbf24" unit="%" yDomain={[0, 100]} />
      </StatCard>

      {/* Memory */}
      <StatCard
        icon={MemoryStick}
        label="Memory"
        value={formatPercent(memPct)}
        subtitle={memUsed}
        color="text-violet-400"
      >
        <MiniChart data={memoryHistory} color="#a78bfa" unit="%" yDomain={[0, 100]} />
      </StatCard>
    </div>
  )
}

// ── Full latency/FPS overlay chart for the Benchmark / Compare pages ─────────

interface OverlayChartProps {
  title: string
  series: Array<{ name: string; data: ChartPoint[]; color: string }>
  unit: string
  yDomain?: [number, number]
  className?: string
}

export function OverlayChart({ title, series, unit, yDomain, className }: OverlayChartProps) {
  const allData = series[0]?.data ?? []

  return (
    <div className={cn('rounded-lg border border-[hsl(var(--border))] bg-[hsl(var(--card))] p-4', className)}>
      <p className="text-sm font-semibold text-foreground mb-3">{title}</p>
      <ResponsiveContainer width="100%" height={200}>
        <LineChart data={allData} margin={{ top: 4, right: 8, left: -16, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" strokeOpacity={0.4} />
          <XAxis
            dataKey="time"
            tick={{ fontSize: 10, fill: 'hsl(var(--muted-foreground))' }}
            tickLine={false}
            axisLine={false}
            interval="preserveStartEnd"
          />
          <YAxis
            tick={{ fontSize: 10, fill: 'hsl(var(--muted-foreground))' }}
            tickLine={false}
            axisLine={false}
            domain={yDomain ?? ['auto', 'auto']}
            width={36}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: 'hsl(var(--card))',
              border: '1px solid hsl(var(--border))',
              borderRadius: '6px',
              fontSize: 11,
            }}
            formatter={(v: number, name: string) => [`${v.toFixed(2)} ${unit}`, name]}
          />
          {series.map((s) => (
            <Line
              key={s.name}
              type="monotone"
              dataKey="value"
              data={s.data}
              name={s.name}
              stroke={s.color}
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
      {series.length > 1 && (
        <div className="flex items-center gap-4 mt-2">
          {series.map((s) => (
            <div key={s.name} className="flex items-center gap-1.5 text-xs text-muted-foreground">
              <div className="w-3 h-0.5 rounded" style={{ backgroundColor: s.color }} />
              {s.name}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

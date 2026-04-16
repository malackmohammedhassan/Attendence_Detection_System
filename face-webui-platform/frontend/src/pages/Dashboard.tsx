import { useEffect } from 'react'
import { Activity, Cpu, Database, Zap } from 'lucide-react'
import { cn, formatFps, formatMs, formatPercent, formatMb } from '@/lib/utils'
import { PerformanceCharts } from '@/components/PerformanceCharts'
import { LogsPanel } from '@/components/LogsPanel'
import { SlidersPanel } from '@/components/SlidersPanel'
import { createMetricsWs } from '@/services/websocket'
import { metricsApi } from '@/services/api'
import { useMetricsStore } from '@/store/useMetricsStore'
import { useModelStore } from '@/store/useModelStore'
import type { WsMetricsTick, WsMessage } from '@/types'

// ── Stat pill ────────────────────────────────────────────────────────────────

interface StatPillProps {
  icon: React.ElementType
  label: string
  value: string
  color?: string
}

function StatPill({ icon: Icon, label, value, color = 'text-primary' }: StatPillProps) {
  return (
    <div className="flex items-center gap-3 rounded-lg border border-[hsl(var(--border))] bg-[hsl(var(--card))] px-4 py-3">
      <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-primary/10">
        <Icon className={cn('w-4 h-4', color)} />
      </div>
      <div>
        <p className="text-[11px] text-muted-foreground uppercase tracking-wide">{label}</p>
        <p className="text-base font-bold font-mono text-foreground leading-tight">{value}</p>
      </div>
    </div>
  )
}

// ── Dashboard page ────────────────────────────────────────────────────────────

export function Dashboard() {
  const snap = useMetricsStore((s) => s.latestSnapshot)
  const pushSnapshot = useMetricsStore((s) => s.pushSnapshot)
  const setMetricsWsStatus = useMetricsStore((s) => s.setMetricsWsStatus)
  const metricsSummary = useMetricsStore((s) => s.metricsSummary)
  const setMetricsSummary = useMetricsStore((s) => s.setMetricsSummary)
  const activeModel = useModelStore((s) => s.activeModel)

  // Connect metrics WebSocket
  useEffect(() => {
    const ws = createMetricsWs(
      (msg: WsMessage) => {
        if (msg.type === 'metrics_tick') {
          const tick = msg as WsMetricsTick
          if (tick.system) pushSnapshot(tick.system)
        }
      },
      setMetricsWsStatus
    )
    ws.connect()
    return () => ws.disconnect()
  }, [pushSnapshot, setMetricsWsStatus])

  // Load summary on mount
  useEffect(() => {
    metricsApi.getSummary().then(setMetricsSummary).catch(() => {})
  }, [setMetricsSummary])

  const trackedModels = metricsSummary?.collector.models_tracked ?? []
  const totalInferences = metricsSummary?.collector.total_inferences ?? 0

  return (
    <div className="space-y-6">
      {/* Hero stats row */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatPill
          icon={Zap}
          label="Live FPS"
          value={snap ? formatFps(snap.inference.fps) : '—'}
          color="text-emerald-400"
        />
        <StatPill
          icon={Cpu}
          label="CPU"
          value={snap ? formatPercent(snap.cpu.overall_percent) : '—'}
          color="text-amber-400"
        />
        <StatPill
          icon={Activity}
          label="Memory"
          value={snap ? formatMb(snap.memory.used_mb) : '—'}
          color="text-violet-400"
        />
        <StatPill
          icon={Database}
          label="Total Infs"
          value={totalInferences.toLocaleString()}
          color="text-blue-400"
        />
      </div>

      {/* Main grid */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        {/* Performance charts — takes up 2/3 */}
        <div className="xl:col-span-2 space-y-4">
          <PerformanceCharts />

          {/* Active model info */}
          {activeModel && (
            <div className="rounded-lg border border-[hsl(var(--border))] bg-[hsl(var(--card))] p-4">
              <p className="text-xs text-muted-foreground uppercase tracking-wide mb-2">
                Active Model
              </p>
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center">
                  <Activity className="w-4 h-4 text-primary" />
                </div>
                <div>
                  <p className="font-mono font-semibold text-foreground">{activeModel}</p>
                  {metricsSummary?.inference[activeModel] && (
                    <p className="text-xs text-muted-foreground">
                      {formatFps(metricsSummary.inference[activeModel].fps_estimate)} estimated ·{' '}
                      {formatMs(metricsSummary.inference[activeModel].mean_latency_ms)} avg latency
                    </p>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* Tracked models summary */}
          {trackedModels.length > 0 && (
            <div className="rounded-lg border border-[hsl(var(--border))] bg-[hsl(var(--card))] p-4">
              <p className="text-xs text-muted-foreground uppercase tracking-wide mb-3">
                Model Inference Stats
              </p>
              <div className="space-y-2">
                {trackedModels.map((name) => {
                  const stats = metricsSummary?.inference[name]
                  if (!stats) return null
                  return (
                    <div
                      key={name}
                      className="flex items-center justify-between py-1.5 border-b border-[hsl(var(--border))] last:border-0"
                    >
                      <span className="font-mono text-xs text-foreground">{name}</span>
                      <div className="flex items-center gap-4 text-xs text-muted-foreground">
                        <span>{formatFps(stats.fps_estimate)}</span>
                        <span>{formatMs(stats.mean_latency_ms)}</span>
                        <span>{stats.sample_count.toLocaleString()} samples</span>
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>
          )}
        </div>

        {/* Right column — sliders + system info */}
        <div className="space-y-4">
          <SlidersPanel />

          {/* System info */}
          {snap && (
            <div className="rounded-lg border border-[hsl(var(--border))] bg-[hsl(var(--card))] p-4">
              <p className="text-xs text-muted-foreground uppercase tracking-wide mb-3">System</p>
              <dl className="space-y-1.5 text-xs">
                {[
                  ['Uptime', `${(snap.uptime_sec / 3600).toFixed(1)}h`],
                  ['Swap Used', formatMb(snap.memory.swap_used_mb)],
                  ['Process CPU', formatPercent(snap.cpu.process_percent)],
                  ['Process RAM', formatMb(snap.memory.process_rss_mb)],
                  ['Total Frames', snap.inference.total_frames.toLocaleString()],
                ].map(([label, value]) => (
                  <div key={label} className="flex justify-between">
                    <dt className="text-muted-foreground">{label}</dt>
                    <dd className="font-mono text-foreground">{value}</dd>
                  </div>
                ))}
              </dl>
            </div>
          )}
        </div>
      </div>

      {/* Logs */}
      <LogsPanel className="h-64" />
    </div>
  )
}

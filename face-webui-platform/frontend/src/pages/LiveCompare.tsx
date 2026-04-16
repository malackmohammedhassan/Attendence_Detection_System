import { Activity, Clock, Box, Crosshair } from 'lucide-react'
import { cn, formatMs, formatPercent } from '@/lib/utils'
import { WebcamViewer } from '@/components/WebcamViewer'
import { SlidersPanel } from '@/components/SlidersPanel'
import { useModelStore } from '@/store/useModelStore'
import type { BoundingBox, InferenceResult } from '@/types'

// ── Detection stat strip ─────────────────────────────────────────────────────

interface DetectionStatsProps {
  result: InferenceResult | null
}

function DetectionStats({ result }: DetectionStatsProps) {
  const items = [
    {
      icon: Box,
      label: 'Detections',
      value: result ? String(result.detection_count) : '—',
      color: 'text-emerald-400',
    },
    {
      icon: Clock,
      label: 'Latency',
      value: result ? formatMs(result.latency_ms) : '—',
      color: 'text-blue-400',
    },
    {
      icon: Crosshair,
      label: 'Confidence',
      value:
        result && result.detections.length > 0
          ? formatPercent(
              result.detections.reduce((acc: number, b: BoundingBox) => acc + b.confidence, 0) / result.detections.length
            )
          : '—',
      color: 'text-amber-400',
    },
    {
      icon: Activity,
      label: 'Model',
      value: result?.model_name ?? '—',
      color: 'text-violet-400',
    },
  ]

  return (
    <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
      {items.map(({ icon: Icon, label, value, color }) => (
        <div
          key={label}
          className="flex items-center gap-2 rounded-lg border border-[hsl(var(--border))] bg-[hsl(var(--card))] px-3 py-2"
        >
          <Icon className={cn('w-3.5 h-3.5 shrink-0', color)} />
          <div className="min-w-0">
            <p className="text-[10px] text-muted-foreground truncate">{label}</p>
            <p className="text-sm font-mono font-semibold text-foreground truncate">{value}</p>
          </div>
        </div>
      ))}
    </div>
  )
}

// ── Bounding box list ────────────────────────────────────────────────────────

function BoxList({ result }: { result: InferenceResult | null }) {
  if (!result || result.detections.length === 0) {
    return (
      <p className="text-xs text-muted-foreground py-2">
        {result ? 'No faces detected in this frame.' : 'Waiting for frames…'}
      </p>
    )
  }

  return (
    <div className="space-y-1.5 max-h-48 overflow-y-auto pr-1">
      {result.detections.map((box: BoundingBox, i: number) => (
        <div
          key={i}
          className="flex items-center justify-between text-xs font-mono rounded border border-[hsl(var(--border))] px-2 py-1"
        >
          <span className="text-muted-foreground">
            #{i + 1} ({box.x1},{box.y1}) {box.width}×{box.height}
          </span>
          <span
            className={cn(
              'font-semibold',
              box.confidence > 0.8
                ? 'text-emerald-400'
                : box.confidence > 0.5
                  ? 'text-amber-400'
                  : 'text-red-400'
            )}
          >
            {formatPercent(box.confidence)}
          </span>
        </div>
      ))}
    </div>
  )
}

// ── Page ─────────────────────────────────────────────────────────────────────

export function LiveCompare() {
  const latestResult = useModelStore((s) => s.latestResult)
  const streamStatus = useModelStore((s) => s.streamStatus)
  const activeModel = useModelStore((s) => s.activeModel)

  return (
    <div className="space-y-4">
      {/* Header row */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-sm font-semibold text-foreground">Live Inference Stream</h2>
          <p className="text-xs text-muted-foreground mt-0.5">
            Webcam frames sent over WebSocket · bounding boxes overlaid in real-time
          </p>
        </div>
        <div className="flex items-center gap-1.5">
          <div
            className={cn(
              'w-2 h-2 rounded-full',
              streamStatus === 'connected' ? 'bg-emerald-400 animate-pulse' : 'bg-muted-foreground'
            )}
          />
          <span className="text-xs text-muted-foreground capitalize">{streamStatus}</span>
        </div>
      </div>

      {/* Main grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Webcam — 2/3 width */}
        <div className="lg:col-span-2">
          <WebcamViewer className="w-full" />
        </div>

        {/* Controls — 1/3 */}
        <div className="space-y-4">
          <SlidersPanel />

          {/* Active model indicator */}
          <div className="rounded-lg border border-[hsl(var(--border))] bg-[hsl(var(--card))] p-4">
            <p className="text-xs text-muted-foreground uppercase tracking-wide mb-2">
              Active Model
            </p>
            {activeModel ? (
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-emerald-400" />
                <span className="font-mono text-sm font-semibold text-foreground">
                  {activeModel}
                </span>
              </div>
            ) : (
              <p className="text-xs text-muted-foreground">
                No model active — select one from the header.
              </p>
            )}
          </div>

          {/* Detected boxes */}
          <div className="rounded-lg border border-[hsl(var(--border))] bg-[hsl(var(--card))] p-4">
            <p className="text-xs text-muted-foreground uppercase tracking-wide mb-3">
              Detected Faces
            </p>
            <BoxList result={latestResult} />
          </div>
        </div>
      </div>

      {/* Stats row */}
      <DetectionStats result={latestResult} />

      {/* Frame meta */}
      {latestResult && (
        <p className="text-[10px] text-muted-foreground font-mono">
          ts={new Date(latestResult.timestamp * 1000).toLocaleTimeString()}
        </p>
      )}
    </div>
  )
}

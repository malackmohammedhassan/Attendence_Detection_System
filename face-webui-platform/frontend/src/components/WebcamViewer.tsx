/**
 * WebcamViewer â€” real-time webcam streaming + detection overlay.
 *
 * Uses the useLiveStream hook to manage:
 *   â€¢ Camera (getUserMedia)
 *   â€¢ Frame pump (video â†’ canvas â†’ JPEG â†’ WS binary)
 *   â€¢ WebSocket connection to /ws/live
 *
 * On each server response it:
 *   â€¢ Draws bounding boxes on the overlay canvas
 *   â€¢ Pushes the latest InferenceResult to useModelStore
 *   â€¢ Pushes the system snapshot to useMetricsStore (for live charts)
 *
 * No memory leaks: all refs and intervals are cleaned up by the hook.
 */

import { useCallback, useRef } from 'react'
import { Camera, CameraOff, Loader2 } from 'lucide-react'
import { cn, formatMs, formatPercent } from '@/lib/utils'
import { useLiveStream } from '@/hooks/useLiveStream'
import { useModelStore } from '@/store/useModelStore'
import { useMetricsStore } from '@/store/useMetricsStore'
import type { BoundingBox, WsLiveResult, WsMetricsTick } from '@/types'

// â”€â”€ Constants â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

const CAPTURE_WIDTH = 640
const CAPTURE_HEIGHT = 480

const BOX_COLOR = '#34d399'           // emerald-400
const BOX_LABEL_BG = 'rgba(16,185,129,0.82)'

// â”€â”€ Box drawing â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

function drawDetections(
  canvas: HTMLCanvasElement,
  boxes: BoundingBox[],
  frameW: number,
  frameH: number
) {
  const ctx = canvas.getContext('2d')
  if (!ctx) return

  ctx.clearRect(0, 0, canvas.width, canvas.height)
  if (boxes.length === 0) return

  const scaleX = canvas.width / frameW
  const scaleY = canvas.height / frameH

  for (const box of boxes) {
    const x = box.x1 * scaleX
    const y = box.y1 * scaleY
    const w = (box.x2 - box.x1) * scaleX
    const h = (box.y2 - box.y1) * scaleY
    const label = `${box.label} ${(box.confidence * 100).toFixed(0)}%`
    const labelW = Math.max(w, ctx.measureText(label).width + 8)

    // Box outline
    ctx.strokeStyle = BOX_COLOR
    ctx.lineWidth = 2
    ctx.strokeRect(x, y, w, h)

    // Corner tick marks
    const tick = Math.min(w, h) * 0.15
    ctx.strokeStyle = '#fff'
    ctx.lineWidth = 2
    ctx.beginPath()
    ctx.moveTo(x, y + tick); ctx.lineTo(x, y); ctx.lineTo(x + tick, y)
    ctx.moveTo(x + w - tick, y); ctx.lineTo(x + w, y); ctx.lineTo(x + w, y + tick)
    ctx.moveTo(x, y + h - tick); ctx.lineTo(x, y + h); ctx.lineTo(x + tick, y + h)
    ctx.moveTo(x + w - tick, y + h); ctx.lineTo(x + w, y + h); ctx.lineTo(x + w, y + h - tick)
    ctx.stroke()

    // Label background + text
    const labelY = y >= 20 ? y - 18 : y + h + 2
    ctx.fillStyle = BOX_LABEL_BG
    ctx.fillRect(x, labelY, labelW, 18)
    ctx.fillStyle = '#fff'
    ctx.font = 'bold 11px "JetBrains Mono", monospace'
    ctx.fillText(label, x + 4, labelY + 13)
  }
}

// â”€â”€ Component props â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

interface WebcamViewerProps {
  className?: string
}

export function WebcamViewer({ className }: WebcamViewerProps) {
  // Store access
  const activeModel = useModelStore((s) => s.activeModel)
  const settings = useModelStore((s) => s.settings)
  const setLatestResult = useModelStore((s) => s.setLatestResult)
  const setStreamStatus = useModelStore((s) => s.setStreamStatus)
  const latestResult = useModelStore((s) => s.latestResult)
  const pushSnapshot = useMetricsStore((s) => s.pushSnapshot)

  // Overlay canvas â€” separate from the hidden capture canvas inside the hook
  const overlayRef = useRef<HTMLCanvasElement>(null)

  // â”€â”€ Callbacks (stable â€” stored as refs inside the hook) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  const handleResult = useCallback(
    (result: WsLiveResult) => {
      setLatestResult(result)
      if (overlayRef.current) {
        drawDetections(
          overlayRef.current,
          result.detections,
          result.frame_width,
          result.frame_height
        )
      }
    },
    [setLatestResult]
  )

  const handleMetrics = useCallback(
    (tick: WsMetricsTick) => {
      if (tick.system) pushSnapshot(tick.system)
    },
    [pushSnapshot]
  )

  const handleStatusChange = useCallback(
    (status: Parameters<typeof setStreamStatus>[0]) => {
      setStreamStatus(status)
      if (status === 'disconnected' || status === 'error') {
        const c = overlayRef.current
        c?.getContext('2d')?.clearRect(0, 0, c.width, c.height)
      }
    },
    [setStreamStatus]
  )

  // â”€â”€ Hook â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  const {
    videoRef,
    captureCanvasRef,
    isStreaming,
    wsStatus,
    cameraError,
    start,
    stop,
  } = useLiveStream({
    confidenceThreshold: settings.confidenceThreshold,
    frameIntervalMs: 100,
    onResult: handleResult,
    onMetrics: handleMetrics,
    onStatusChange: handleStatusChange,
  })

  const canStart = !!activeModel
  const faceCount = latestResult?.detection_count ?? 0

  return (
    <div
      className={cn(
        'rounded-lg border border-[hsl(var(--border))] bg-[hsl(var(--card))] overflow-hidden',
        className
      )}
    >
      {/* â”€â”€ Header â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */}
      <div className="flex items-center justify-between px-4 py-2.5 border-b border-[hsl(var(--border))]">
        <div className="flex items-center gap-2.5">
          <Camera className="w-4 h-4 text-primary" />
          <span className="text-sm font-semibold text-foreground">Live Webcam</span>
          <span
            className={cn(
              'inline-flex h-2 w-2 rounded-full',
              wsStatus === 'connected'    && 'bg-emerald-400',
              wsStatus === 'connecting'   && 'bg-yellow-400 animate-pulse',
              wsStatus === 'error'        && 'bg-red-500',
              wsStatus === 'disconnected' && 'bg-slate-600'
            )}
            title={`WebSocket: ${wsStatus}`}
          />
          <span className="text-[10px] text-muted-foreground capitalize">{wsStatus}</span>
        </div>

        <div className="flex items-center gap-3">
          {isStreaming && latestResult && (
            <span className="text-xs font-mono text-muted-foreground tabular-nums">
              {formatMs(latestResult.latency_ms)}&nbsp;Â·&nbsp;
              {faceCount} face{faceCount !== 1 ? 's' : ''}&nbsp;Â·&nbsp;
              {formatPercent(latestResult.confidence_threshold)} thr
            </span>
          )}

          <button
            onClick={isStreaming ? stop : start}
            disabled={!canStart && !isStreaming}
            className={cn(
              'flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-md border font-medium transition-colors',
              isStreaming
                ? 'border-red-800 text-red-400 hover:bg-red-950/40'
                : 'border-primary/40 text-primary hover:bg-primary/10',
              !canStart && !isStreaming && 'opacity-40 cursor-not-allowed'
            )}
          >
            {isStreaming
              ? <><CameraOff className="w-3.5 h-3.5" />Stop</>
              : <><Camera className="w-3.5 h-3.5" />Start</>
            }
          </button>
        </div>
      </div>

      {/* â”€â”€ Video viewport â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */}
      <div className="relative bg-slate-950 aspect-video">
        {/* Live video feed */}
        <video
          ref={videoRef}
          className="absolute inset-0 w-full h-full object-cover"
          muted
          playsInline
          autoPlay={false}
        />

        {/* Hidden capture canvas â€” must stay in DOM for toBlob() */}
        <canvas
          ref={captureCanvasRef}
          width={CAPTURE_WIDTH}
          height={CAPTURE_HEIGHT}
          className="hidden"
          aria-hidden
        />

        {/* Detection overlay */}
        <canvas
          ref={overlayRef}
          width={CAPTURE_WIDTH}
          height={CAPTURE_HEIGHT}
          className="absolute inset-0 w-full h-full pointer-events-none"
          aria-label="Detection overlay"
        />

        {/* Idle / error placeholder */}
        {!isStreaming && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-slate-950/80">
            {wsStatus === 'connecting' ? (
              <Loader2 className="w-8 h-8 text-primary animate-spin" />
            ) : (
              <CameraOff className="w-8 h-8 text-slate-600" />
            )}
            {cameraError ? (
              <p className="text-xs text-red-400 text-center max-w-[260px]">{cameraError}</p>
            ) : (
              <p className="text-xs text-muted-foreground">
                {canStart
                  ? 'Click Start to begin streaming'
                  : 'Select a model to enable live stream'}
              </p>
            )}
          </div>
        )}

        {/* Live inference HUD */}
        {isStreaming && latestResult && (
          <div className="absolute top-2 left-2 flex flex-col gap-1">
            <div className="bg-black/65 rounded px-2 py-1 text-[11px] font-mono text-emerald-400">
              {faceCount} face{faceCount !== 1 ? 's' : ''} Â· {formatMs(latestResult.latency_ms)}
            </div>
            <div className="bg-black/55 rounded px-2 py-1 text-[10px] font-mono text-slate-300">
              {latestResult.model_name}
            </div>
          </div>
        )}

        {/* LIVE badge */}
        {isStreaming && (
          <div className="absolute top-2 right-2 flex items-center gap-1.5 bg-black/55 rounded px-2 py-1">
            <span className="h-1.5 w-1.5 rounded-full bg-red-500 animate-pulse" />
            <span className="text-[10px] font-mono text-slate-300">LIVE</span>
          </div>
        )}
      </div>

      {/* â”€â”€ Per-face confidence bars â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */}
      {isStreaming && latestResult && latestResult.detections.length > 0 && (
        <div className="px-4 pt-2 pb-3 grid grid-cols-2 sm:grid-cols-4 gap-2">
          {latestResult.detections.slice(0, 4).map((box, i) => (
            <div key={i} className="space-y-0.5">
              <div className="flex justify-between text-[10px] font-mono">
                <span className="text-muted-foreground">face {i + 1}</span>
                <span style={{ color: BOX_COLOR }}>{formatPercent(box.confidence)}</span>
              </div>
              <div className="h-1 rounded-full bg-muted overflow-hidden">
                <div
                  className="h-full rounded-full"
                  style={{ width: `${box.confidence * 100}%`, backgroundColor: BOX_COLOR }}
                />
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}


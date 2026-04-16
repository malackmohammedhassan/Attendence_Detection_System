/**
 * useLiveStream — Real-time webcam + /ws/live integration hook.
 *
 * Manages three things in one place:
 *   1. Camera  — getUserMedia, video element attachment, track cleanup
 *   2. Frame pump — captures from the video → offscreen canvas → JPEG blob
 *                   → ArrayBuffer sent as binary over the WebSocket
 *   3. WebSocket  — WsClient to /ws/live ; dispatches live_result and
 *                   metrics_tick messages to the caller via stable callbacks
 *
 * Design goals:
 *   • No memory leaks — all refs / intervals / tracks cleaned up in stop()
 *     and in the useEffect cleanup.
 *   • No unsafe re-renders — WsClient, stream, interval are stored in refs.
 *     setState is called only when the UI must update (isStreaming / status).
 *   • Threshold sync — when confidenceThreshold changes while streaming the
 *     hook sends {"type":"config","threshold":N} without reconnecting.
 */

import { useCallback, useEffect, useRef, useState } from 'react'
import { createLiveStreamWs } from '@/services/websocket'
import type {
  ConnectionStatus,
  WsLiveResult,
  WsMessage,
  WsMetricsTick,
} from '@/types'

// ── Constants ─────────────────────────────────────────────────────────────────

const CAPTURE_WIDTH = 640
const CAPTURE_HEIGHT = 480
const DEFAULT_FRAME_INTERVAL_MS = 100  // ~10 fps send rate
const JPEG_QUALITY = 0.72              // quality / bandwidth trade-off

// ── Types ─────────────────────────────────────────────────────────────────────

export interface UseLiveStreamOptions {
  /** Inference confidence threshold sent to the server (0–1). */
  confidenceThreshold?: number
  /** How often to capture + send a frame in milliseconds. */
  frameIntervalMs?: number
  /** Called with every processed detection result from the server. */
  onResult?: (result: WsLiveResult) => void
  /** Called with every periodic metrics tick from the server. */
  onMetrics?: (tick: WsMetricsTick) => void
  /** Called when the WebSocket connection status changes. */
  onStatusChange?: (status: ConnectionStatus) => void
  /** Called when camera access fails. */
  onCameraError?: (message: string) => void
}

export interface UseLiveStreamReturn {
  /** Attach to a <video> element to show the webcam feed. */
  videoRef: React.RefObject<HTMLVideoElement>
  /** Hidden <canvas> used for JPEG capture — keep it in the DOM. */
  captureCanvasRef: React.RefObject<HTMLCanvasElement>
  /** True while the camera + WebSocket are both active. */
  isStreaming: boolean
  /** Current WebSocket connection state. */
  wsStatus: ConnectionStatus
  /** Last camera error string, or null. */
  cameraError: string | null
  /** Start camera + WebSocket; resolves after getUserMedia succeeds. */
  start: () => Promise<void>
  /** Stop camera, WebSocket, and frame pump. */
  stop: () => void
}

// ── Hook ──────────────────────────────────────────────────────────────────────

export function useLiveStream(options: UseLiveStreamOptions = {}): UseLiveStreamReturn {
  const {
    confidenceThreshold = 0.5,
    frameIntervalMs = DEFAULT_FRAME_INTERVAL_MS,
  } = options

  // ── Refs — these never cause re-renders ──────────────────────────────────
  const videoRef = useRef<HTMLVideoElement>(null)
  const captureCanvasRef = useRef<HTMLCanvasElement>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const wsRef = useRef<ReturnType<typeof createLiveStreamWs> | null>(null)
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)

  // Keep callbacks stable — stored in refs so we never need to re-subscribe
  const onResultRef = useRef(options.onResult)
  const onMetricsRef = useRef(options.onMetrics)
  const onStatusChangeRef = useRef(options.onStatusChange)
  const onCameraErrorRef = useRef(options.onCameraError)
  const thresholdRef = useRef(confidenceThreshold)

  // Update callback refs when they change (no reconnect needed)
  useEffect(() => { onResultRef.current = options.onResult }, [options.onResult])
  useEffect(() => { onMetricsRef.current = options.onMetrics }, [options.onMetrics])
  useEffect(() => {
    onStatusChangeRef.current = options.onStatusChange
  }, [options.onStatusChange])
  useEffect(() => {
    onCameraErrorRef.current = options.onCameraError
  }, [options.onCameraError])

  // ── Sync threshold changes to the server (no reconnect) ─────────────────
  useEffect(() => {
    thresholdRef.current = confidenceThreshold
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.sendJson({ type: 'config', threshold: confidenceThreshold })
    }
  }, [confidenceThreshold])

  // ── State — only what the JSX actually needs ─────────────────────────────
  const [isStreaming, setIsStreaming] = useState(false)
  const [wsStatus, setWsStatus] = useState<ConnectionStatus>('disconnected')
  const [cameraError, setCameraError] = useState<string | null>(null)

  // ── Internal stop (won't setState unless asked) ──────────────────────────
  const _teardown = useCallback(() => {
    if (intervalRef.current !== null) {
      clearInterval(intervalRef.current)
      intervalRef.current = null
    }
    wsRef.current?.disconnect()
    wsRef.current = null
    streamRef.current?.getTracks().forEach((t) => t.stop())
    streamRef.current = null
  }, [])

  // ── Public stop ─────────────────────────────────────────────────────────
  const stop = useCallback(() => {
    _teardown()
    setIsStreaming(false)
    setWsStatus('disconnected')
  }, [_teardown])

  // ── Public start ────────────────────────────────────────────────────────
  const start = useCallback(async () => {
    // --- 1. Camera --------------------------------------------------------
    setCameraError(null)
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: CAPTURE_WIDTH },
          height: { ideal: CAPTURE_HEIGHT },
          facingMode: 'user',
        },
        audio: false,
      })
      streamRef.current = mediaStream

      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream
        await videoRef.current.play()
      }
    } catch (err) {
      const msg =
        err instanceof DOMException && err.name === 'NotAllowedError'
          ? 'Camera access denied. Allow camera permissions and retry.'
          : err instanceof DOMException && err.name === 'NotFoundError'
            ? 'No camera device found.'
            : 'Could not access the camera.'
      setCameraError(msg)
      onCameraErrorRef.current?.(msg)
      return
    }

    // --- 2. WebSocket -----------------------------------------------------
    const ws = createLiveStreamWs(
      (msg: WsMessage) => {
        if (msg.type === 'live_result') {
          onResultRef.current?.(msg as WsLiveResult)
        } else if (msg.type === 'metrics_tick') {
          onMetricsRef.current?.(msg as WsMetricsTick)
        }
        // 'ready' / 'no_model' / 'error' are informational — no action needed
        // (callers can extend by sharing the onMessage callback if required)
      },
      (status: ConnectionStatus) => {
        setWsStatus(status)
        onStatusChangeRef.current?.(status)

        // When the WS reconnects, re-send current threshold so the server
        // session is consistent with the client UI state
        if (status === 'connected') {
          ws.sendJson({
            type: 'config',
            threshold: thresholdRef.current,
          })
        }
      }
    )
    wsRef.current = ws
    ws.connect()

    // --- 3. Frame pump ----------------------------------------------------
    const interval = setInterval(() => {
      const video = videoRef.current
      const canvas = captureCanvasRef.current
      if (!video || !canvas) return
      if (ws.readyState !== WebSocket.OPEN) return
      // Skip if video is not playing yet
      if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return

      const ctx = canvas.getContext('2d')
      if (!ctx) return

      ctx.drawImage(video, 0, 0, CAPTURE_WIDTH, CAPTURE_HEIGHT)

      canvas.toBlob(
        (blob) => {
          if (!blob) return
          blob.arrayBuffer().then((buf) => {
            // Guard: ws may have been cleaned up between toBlob and this callback
            if (wsRef.current?.readyState === WebSocket.OPEN) {
              wsRef.current.sendBinary(buf)
            }
          }).catch(() => {/* ignore errors on unmounted components */})
        },
        'image/jpeg',
        JPEG_QUALITY
      )
    }, frameIntervalMs)

    intervalRef.current = interval
    setIsStreaming(true)
  }, [frameIntervalMs, _teardown])

  // ── Cleanup on unmount ───────────────────────────────────────────────────
  useEffect(() => {
    return () => {
      _teardown()
    }
  }, [_teardown])

  return {
    videoRef,
    captureCanvasRef,
    isStreaming,
    wsStatus,
    cameraError,
    start,
    stop,
  }
}

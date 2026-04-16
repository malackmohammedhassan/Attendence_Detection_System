import type { WsMessage, ConnectionStatus } from '@/types'

// ─────────────────────────────────────────────────────────────────────────────
//  WebSocket client with auto-reconnect and typed message dispatch
// ─────────────────────────────────────────────────────────────────────────────

const WS_BASE = import.meta.env.VITE_WS_BASE_URL ??
  `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.host}`

export type MessageHandler = (message: WsMessage) => void
export type StatusHandler = (status: ConnectionStatus) => void

interface WsClientOptions {
  /** Path relative to WS_BASE, e.g. '/ws/metrics' */
  path: string
  onMessage: MessageHandler
  onStatusChange: StatusHandler
  /** Reconnect delay in ms (default 2000) */
  reconnectDelay?: number
  /** Max reconnect attempts before giving up (default Infinity) */
  maxRetries?: number
}

export class WsClient {
  private ws: WebSocket | null = null
  private readonly path: string
  private readonly onMessage: MessageHandler
  private readonly onStatusChange: StatusHandler
  private readonly reconnectDelay: number
  private readonly maxRetries: number
  private retryCount = 0
  private intentionallyClosed = false
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null

  constructor(options: WsClientOptions) {
    this.path = options.path
    this.onMessage = options.onMessage
    this.onStatusChange = options.onStatusChange
    this.reconnectDelay = options.reconnectDelay ?? 2000
    this.maxRetries = options.maxRetries ?? Infinity
  }

  get url(): string {
    return `${WS_BASE}${this.path}`
  }

  connect(): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) return

    this.intentionallyClosed = false
    this.onStatusChange('connecting')

    try {
      this.ws = new WebSocket(this.url)
    } catch {
      this.onStatusChange('error')
      this._scheduleReconnect()
      return
    }

    this.ws.onopen = () => {
      this.retryCount = 0
      this.onStatusChange('connected')
    }

    this.ws.onmessage = (event: MessageEvent<string>) => {
      try {
        const msg = JSON.parse(event.data) as WsMessage
        this.onMessage(msg)
      } catch {
        // Not JSON — ignore silently
      }
    }

    this.ws.onerror = () => {
      this.onStatusChange('error')
    }

    this.ws.onclose = () => {
      if (!this.intentionallyClosed) {
        this.onStatusChange('disconnected')
        this._scheduleReconnect()
      }
    }
  }

  sendJson(data: Record<string, unknown>): boolean {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data))
      return true
    }
    return false
  }

  sendBinary(data: ArrayBuffer | Blob): boolean {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(data)
      return true
    }
    return false
  }

  disconnect(): void {
    this.intentionallyClosed = true
    if (this.reconnectTimer !== null) {
      clearTimeout(this.reconnectTimer)
      this.reconnectTimer = null
    }
    this.ws?.close()
    this.ws = null
    this.onStatusChange('disconnected')
  }

  get readyState(): number {
    return this.ws?.readyState ?? WebSocket.CLOSED
  }

  private _scheduleReconnect(): void {
    if (this.intentionallyClosed) return
    if (this.retryCount >= this.maxRetries) {
      this.onStatusChange('error')
      return
    }
    this.retryCount++
    const delay = Math.min(this.reconnectDelay * this.retryCount, 30_000)
    this.reconnectTimer = setTimeout(() => {
      this.connect()
    }, delay)
  }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Pre-configured client factories
// ─────────────────────────────────────────────────────────────────────────────

export function createMetricsWs(
  onMessage: MessageHandler,
  onStatusChange: StatusHandler
): WsClient {
  return new WsClient({ path: '/ws/metrics', onMessage, onStatusChange })
}

export function createLogsWs(
  onMessage: MessageHandler,
  onStatusChange: StatusHandler
): WsClient {
  return new WsClient({ path: '/ws/logs', onMessage, onStatusChange })
}

export function createTrainingWs(
  jobId: string,
  onMessage: MessageHandler,
  onStatusChange: StatusHandler
): WsClient {
  return new WsClient({
    path: `/api/v1/train/ws/${jobId}`,
    onMessage,
    onStatusChange,
  })
}

export function createInferenceStreamWs(
  onMessage: MessageHandler,
  onStatusChange: StatusHandler
): WsClient {
  return new WsClient({
    path: '/api/v1/inference/stream',
    onMessage,
    onStatusChange,
    maxRetries: 5,
  })
}

/**
 * Create a WsClient for the /ws/live real-time streaming endpoint.
 *
 * Protocol:
 *   Send binary  → JPEG frame bytes
 *   Send JSON    → { type: 'config', threshold: 0.5 }
 *   Recv JSON    → WsLiveResult | WsMetricsTick | { type: 'no_model' | 'ready' | 'error' }
 */
/**
 * Create a WsClient for /api/v1/train/ws/logs — streams raw training stdout lines.
 * Sends { type: 'train_log_batch' } on connect, then individual { type: 'train_log' }.
 */
export function createTrainLogsWs(
  onMessage: MessageHandler,
  onStatusChange: StatusHandler
): WsClient {
  return new WsClient({
    path: '/api/v1/train/ws/logs',
    onMessage,
    onStatusChange,
    reconnectDelay: 3000,
    maxRetries: Infinity,
  })
}

export function createLiveStreamWs(
  onMessage: MessageHandler,
  onStatusChange: StatusHandler
): WsClient {
  return new WsClient({
    path: '/ws/live',
    onMessage,
    onStatusChange,
    reconnectDelay: 2000,
    maxRetries: Infinity,
  })
}

// ─────────────────────────────────────────────────────────────────────────────
//  React hook pattern — use inside components via useEffect
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Usage example in a component:
 *
 *   useEffect(() => {
 *     const ws = createMetricsWs(
 *       (msg) => { if (msg.type === 'metrics_tick') pushSnapshot(msg.system) },
 *       (status) => setMetricsWsStatus(status)
 *     )
 *     ws.connect()
 *     return () => ws.disconnect()
 *   }, [])
 */

import { useEffect, useRef, useState } from 'react'
import { Terminal, Trash2, Download, ChevronDown, Wifi } from 'lucide-react'
import { cn, levelColor } from '@/lib/utils'
import { useMetricsStore } from '@/store/useMetricsStore'
import { createLogsWs } from '@/services/websocket'
import type { WsLogEntry, WsMessage } from '@/types'

export function LogsPanel({ className }: { className?: string }) {
  const logs = useMetricsStore((s) => s.logs)
  const logsWsStatus = useMetricsStore((s) => s.logsWsStatus)
  const logFilter = useMetricsStore((s) => s.logFilter)
  const appendLog = useMetricsStore((s) => s.appendLog)
  const prependLogs = useMetricsStore((s) => s.prependLogs)
  const setLogsWsStatus = useMetricsStore((s) => s.setLogsWsStatus)
  const setLogFilter = useMetricsStore((s) => s.setLogFilter)
  const clearLogs = useMetricsStore((s) => s.clearLogs)

  const [autoScroll, setAutoScroll] = useState(true)
  const [levelFilter, setLevelFilter] = useState<string>('ALL')
  const scrollRef = useRef<HTMLDivElement>(null)
  const wsRef = useRef<ReturnType<typeof createLogsWs> | null>(null)

  // Connect WebSocket on mount
  useEffect(() => {
    wsRef.current = createLogsWs(
      (msg: WsMessage) => {
        if (msg.type === 'log') appendLog(msg as WsLogEntry)
        if (msg.type === 'log_replay') {
          prependLogs([msg as WsLogEntry])
        }
      },
      setLogsWsStatus
    )
    wsRef.current.connect()
    return () => wsRef.current?.disconnect()
  }, [appendLog, prependLogs, setLogsWsStatus])

  // Auto-scroll to top (newest logs are prepended at top)
  useEffect(() => {
    if (autoScroll && scrollRef.current) {
      scrollRef.current.scrollTop = 0
    }
  }, [logs, autoScroll])

  const LEVELS = ['ALL', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']

  const filteredLogs = logs.filter((log) => {
    const levelMatch = levelFilter === 'ALL' || log.level === levelFilter
    const textMatch =
      !logFilter ||
      log.message.toLowerCase().includes(logFilter.toLowerCase()) ||
      log.logger.toLowerCase().includes(logFilter.toLowerCase())
    return levelMatch && textMatch
  })

  function downloadLogs() {
    const text = filteredLogs
      .map((l) => `[${l.timestamp}] ${l.level.padEnd(8)} ${l.logger}: ${l.message}`)
      .join('\n')
    const blob = new Blob([text], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `logs_${new Date().toISOString().slice(0, 19)}.txt`
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className={cn('flex flex-col rounded-lg border border-[hsl(var(--border))] bg-[hsl(var(--card))]', className)}>
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2.5 border-b border-[hsl(var(--border))]">
        <div className="flex items-center gap-2">
          <Terminal className="w-4 h-4 text-primary" />
          <span className="text-sm font-semibold text-foreground">Logs</span>
          <span className="text-xs text-muted-foreground">({filteredLogs.length})</span>
          {/* WS status */}
          <div
            className={cn(
              'flex items-center gap-1 text-[10px] px-1.5 py-0.5 rounded',
              logsWsStatus === 'connected'
                ? 'text-emerald-400 bg-emerald-950/40'
                : 'text-slate-400 bg-slate-800'
            )}
          >
            <Wifi className="w-2.5 h-2.5" />
            {logsWsStatus}
          </div>
        </div>

        <div className="flex items-center gap-2">
          {/* Auto-scroll toggle */}
          <button
            onClick={() => setAutoScroll((v) => !v)}
            className={cn(
              'flex items-center gap-1 text-xs px-2 py-1 rounded border transition-colors',
              autoScroll
                ? 'border-primary/50 text-primary bg-primary/10'
                : 'border-[hsl(var(--border))] text-muted-foreground hover:text-foreground'
            )}
            title="Toggle auto-scroll"
          >
            <ChevronDown className="w-3 h-3" />
            Auto
          </button>
          <button
            onClick={downloadLogs}
            className="p-1 text-muted-foreground hover:text-foreground transition-colors"
            title="Download logs"
          >
            <Download className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={clearLogs}
            className="p-1 text-muted-foreground hover:text-red-400 transition-colors"
            title="Clear logs"
          >
            <Trash2 className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* Filter bar */}
      <div className="flex items-center gap-2 px-4 py-2 border-b border-[hsl(var(--border))]">
        {/* Level tabs */}
        <div className="flex items-center gap-0.5">
          {LEVELS.map((level) => (
            <button
              key={level}
              onClick={() => setLevelFilter(level)}
              className={cn(
                'text-[10px] px-2 py-0.5 rounded font-mono font-medium transition-colors',
                levelFilter === level
                  ? 'bg-primary text-primary-foreground'
                  : 'text-muted-foreground hover:text-foreground hover:bg-accent'
              )}
            >
              {level}
            </button>
          ))}
        </div>

        {/* Text search */}
        <input
          type="text"
          value={logFilter}
          onChange={(e) => setLogFilter(e.target.value)}
          placeholder="Filter messages…"
          className="flex-1 bg-transparent border border-[hsl(var(--border))] rounded px-2 py-0.5 text-xs text-foreground placeholder-muted-foreground focus:outline-none focus:border-primary"
        />
      </div>

      {/* Log entries */}
      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto font-mono text-xs p-2 space-y-0.5 min-h-0"
        style={{ maxHeight: '320px' }}
      >
        {filteredLogs.length === 0 ? (
          <div className="flex items-center justify-center h-20 text-muted-foreground text-xs">
            {logsWsStatus === 'connected' ? 'Waiting for log events…' : 'Not connected'}
          </div>
        ) : (
          filteredLogs.map((log, i) => (
            <LogLine key={i} log={log} />
          ))
        )}
      </div>
    </div>
  )
}

function LogLine({ log }: { log: WsLogEntry }) {
  const [expanded, setExpanded] = useState(false)

  return (
    <div
      className="group flex items-start gap-2 px-2 py-1 rounded hover:bg-accent/30 cursor-pointer transition-colors"
      onClick={() => setExpanded((v) => !v)}
    >
      {/* Timestamp */}
      <span className="text-muted-foreground text-[10px] flex-shrink-0 mt-0.5 w-20">
        {log.timestamp.slice(11, 19)}
      </span>

      {/* Level badge */}
      <span
        className={cn(
          'text-[10px] font-bold flex-shrink-0 w-14',
          levelColor(log.level)
        )}
      >
        {log.level}
      </span>

      {/* Logger */}
      <span className="text-slate-500 text-[10px] flex-shrink-0 w-24 truncate">
        {log.logger.split('.').pop()}
      </span>

      {/* Message */}
      <span
        className={cn(
          'flex-1 text-foreground/90 break-all',
          !expanded && 'truncate'
        )}
      >
        {log.message}
      </span>

      {/* Exc text */}
      {expanded && log.exc_text && (
        <pre className="w-full mt-1 text-red-400 text-[10px] overflow-x-auto whitespace-pre-wrap">
          {log.exc_text}
        </pre>
      )}
    </div>
  )
}

import { useLocation } from 'react-router-dom'
import { Bell, RefreshCw, Wifi, WifiOff } from 'lucide-react'
import { cn } from '@/lib/utils'
import { useMetricsStore } from '@/store/useMetricsStore'
import { ModelSelector } from '@/components/ModelSelector'

const PAGE_TITLES: Record<string, string> = {
  '/': 'Dashboard',
  '/training': 'Training',
  '/live-compare': 'Live Compare',
  '/benchmark': 'Benchmark',
}

interface StatusPillProps {
  label: string
  status: 'connected' | 'disconnected' | 'connecting' | 'error'
}

function StatusPill({ label, status }: StatusPillProps) {
  return (
    <div
      className={cn(
        'flex items-center gap-1.5 text-xs px-2 py-1 rounded-full border',
        status === 'connected'
          ? 'bg-emerald-950/50 border-emerald-800 text-emerald-400'
          : status === 'connecting'
          ? 'bg-yellow-950/50 border-yellow-800 text-yellow-400'
          : 'bg-slate-800 border-slate-700 text-slate-400'
      )}
    >
      {status === 'connected' ? (
        <Wifi className="w-3 h-3" />
      ) : status === 'connecting' ? (
        <RefreshCw className="w-3 h-3 animate-spin" />
      ) : (
        <WifiOff className="w-3 h-3" />
      )}
      <span>{label}</span>
    </div>
  )
}

export function Header() {
  const location = useLocation()
  const title = PAGE_TITLES[location.pathname] ?? 'ML Dashboard'
  const metricsStatus = useMetricsStore((s) => s.metricsWsStatus)
  const logsStatus = useMetricsStore((s) => s.logsWsStatus)

  return (
    <header className="flex items-center justify-between h-14 px-6 border-b border-[hsl(var(--border))] bg-[hsl(var(--card))]">
      {/* Page title */}
      <h1 className="text-base font-semibold text-foreground">{title}</h1>

      {/* Right controls */}
      <div className="flex items-center gap-3">
        {/* WS connection status pills */}
        <div className="hidden md:flex items-center gap-2">
          <StatusPill label="Metrics" status={metricsStatus} />
          <StatusPill label="Logs" status={logsStatus} />
        </div>

        {/* Model selector in header */}
        <ModelSelector compact />

        {/* Notification bell placeholder */}
        <button
          className="p-1.5 rounded-md text-muted-foreground hover:text-foreground hover:bg-accent transition-colors"
          aria-label="Notifications"
        >
          <Bell className="w-4 h-4" />
        </button>
      </div>
    </header>
  )
}

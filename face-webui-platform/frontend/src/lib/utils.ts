import { type ClassValue, clsx } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs))
}

export function formatMs(ms: number): string {
  return ms < 1 ? '<1ms' : `${ms.toFixed(1)}ms`
}

export function formatPercent(value: number, decimals = 1): string {
  return `${value.toFixed(decimals)}%`
}

export function formatMb(mb: number): string {
  return mb >= 1024 ? `${(mb / 1024).toFixed(1)}GB` : `${mb.toFixed(0)}MB`
}

export function formatFps(fps: number): string {
  return `${fps.toFixed(1)} FPS`
}

export function statusColor(status: string): string {
  switch (status) {
    case 'running': return 'text-blue-400'
    case 'completed': return 'text-emerald-400'
    case 'failed': return 'text-red-400'
    case 'cancelled': return 'text-yellow-400'
    case 'pending': return 'text-slate-400'
    default: return 'text-slate-400'
  }
}

export function levelColor(level: string): string {
  switch (level.toUpperCase()) {
    case 'DEBUG': return 'text-slate-400'
    case 'INFO': return 'text-sky-400'
    case 'WARNING': return 'text-yellow-400'
    case 'ERROR': return 'text-red-400'
    case 'CRITICAL': return 'text-red-600 font-bold'
    default: return 'text-slate-300'
  }
}

import { NavLink } from 'react-router-dom'
import {
  LayoutDashboard,
  BrainCircuit,
  Swords,
  Gauge,
  Activity,
  ChevronRight,
} from 'lucide-react'
import { cn } from '@/lib/utils'
import { useModelStore } from '@/store/useModelStore'

interface NavItem {
  label: string
  to: string
  icon: React.ElementType
  badge?: string
}

const NAV_ITEMS: NavItem[] = [
  { label: 'Dashboard', to: '/', icon: LayoutDashboard },
  { label: 'Training', to: '/training', icon: BrainCircuit },
  { label: 'Live Compare', to: '/live-compare', icon: Swords },
  { label: 'Benchmark', to: '/benchmark', icon: Gauge },
]

export function Sidebar() {
  const activeModel = useModelStore((s) => s.activeModel)

  return (
    <aside className="flex flex-col h-full w-56 bg-[hsl(var(--card))] border-r border-[hsl(var(--border))]">
      {/* Logo */}
      <div className="flex items-center gap-2 px-4 h-14 border-b border-[hsl(var(--border))]">
        <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-primary">
          <Activity className="w-4 h-4 text-primary-foreground" />
        </div>
        <span className="text-sm font-semibold text-foreground tracking-tight">
          ML Dashboard
        </span>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-2 py-4 space-y-0.5 overflow-y-auto">
        <p className="px-3 mb-2 text-[10px] font-semibold tracking-widest uppercase text-muted-foreground">
          Navigation
        </p>
        {NAV_ITEMS.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.to === '/'}
            className={({ isActive }) =>
              cn(
                'group flex items-center gap-3 px-3 py-2 rounded-md text-sm transition-colors',
                isActive
                  ? 'bg-primary/10 text-primary font-medium'
                  : 'text-muted-foreground hover:bg-accent hover:text-foreground'
              )
            }
          >
            {({ isActive }) => (
              <>
                <item.icon
                  className={cn(
                    'w-4 h-4 flex-shrink-0',
                    isActive ? 'text-primary' : 'text-muted-foreground group-hover:text-foreground'
                  )}
                />
                <span className="flex-1">{item.label}</span>
                {isActive && (
                  <ChevronRight className="w-3 h-3 text-primary opacity-60" />
                )}
                {item.badge && (
                  <span className="text-[10px] bg-primary text-primary-foreground px-1.5 py-0.5 rounded-full">
                    {item.badge}
                  </span>
                )}
              </>
            )}
          </NavLink>
        ))}
      </nav>

      {/* Active model indicator */}
      <div className="px-4 py-3 border-t border-[hsl(var(--border))]">
        <p className="text-[10px] font-semibold tracking-widest uppercase text-muted-foreground mb-1">
          Active Model
        </p>
        {activeModel ? (
          <div className="flex items-center gap-2">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
              <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500" />
            </span>
            <span className="text-xs font-mono text-foreground truncate">{activeModel}</span>
          </div>
        ) : (
          <div className="flex items-center gap-2">
            <span className="inline-flex h-2 w-2 rounded-full bg-slate-600" />
            <span className="text-xs text-muted-foreground">None selected</span>
          </div>
        )}
      </div>
    </aside>
  )
}

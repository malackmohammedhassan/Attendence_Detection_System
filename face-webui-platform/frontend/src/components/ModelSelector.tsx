import { useEffect, useState } from 'react'
import { ChevronDown, Loader2, CheckCircle2, AlertCircle } from 'lucide-react'
import { cn } from '@/lib/utils'
import { modelsApi } from '@/services/api'
import { useModelStore } from '@/store/useModelStore'
import type { ModelInfo } from '@/types'

interface ModelSelectorProps {
  compact?: boolean
}

export function ModelSelector({ compact = false }: ModelSelectorProps) {
  const [open, setOpen] = useState(false)
  const [activating, setActivating] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const availableModels = useModelStore((s) => s.availableModels)
  const activeModel = useModelStore((s) => s.activeModel)
  const isLoadingModels = useModelStore((s) => s.isLoadingModels)
  const setAvailableModels = useModelStore((s) => s.setAvailableModels)
  const setActiveModel = useModelStore((s) => s.setActiveModel)
  const setIsLoadingModels = useModelStore((s) => s.setIsLoadingModels)

  async function loadModels() {
    setIsLoadingModels(true)
    setError(null)
    try {
      const [models, activeInfo] = await Promise.all([
        modelsApi.list(),
        modelsApi.getActive(),
      ])
      setAvailableModels(models)
      if (activeInfo) setActiveModel(activeInfo.name)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load models')
    } finally {
      setIsLoadingModels(false)
    }
  }

  // Load model list on mount
  useEffect(() => { loadModels() }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // Re-fetch every time the dropdown is opened so stale/missing data is refreshed
  useEffect(() => {
    if (open) loadModels()
  }, [open]) // eslint-disable-line react-hooks/exhaustive-deps

  async function handleActivate(model: ModelInfo) {
    if (model.name === activeModel || activating) return
    setActivating(model.name)
    setError(null)
    try {
      await modelsApi.activate(model.name)
      setActiveModel(model.name)
      setOpen(false)
      // Refresh list to get updated load status
      const updated = await modelsApi.list()
      setAvailableModels(updated)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to activate model')
    } finally {
      setActivating(null)
    }
  }


  return (
    <div className="relative">
      {/* Trigger button */}
      <button
        onClick={() => setOpen((o) => !o)}
        className={cn(
          'flex items-center gap-2 rounded-md border border-[hsl(var(--border))] bg-[hsl(var(--card))]',
          'text-sm text-foreground hover:bg-accent transition-colors',
          compact ? 'px-2.5 py-1.5' : 'px-3 py-2 w-full'
        )}
      >
        {isLoadingModels ? (
          <Loader2 className="w-3.5 h-3.5 animate-spin text-muted-foreground" />
        ) : activeModel ? (
          <CheckCircle2 className="w-3.5 h-3.5 text-emerald-500 flex-shrink-0" />
        ) : (
          <div className="w-3.5 h-3.5 rounded-full border-2 border-slate-600 flex-shrink-0" />
        )}
        <span className={cn('truncate', compact ? 'max-w-[120px]' : '')}>
          {isLoadingModels
            ? 'Loading models…'
            : activeModel ?? 'Select model'}
        </span>
        <ChevronDown
          className={cn(
            'w-3.5 h-3.5 text-muted-foreground transition-transform flex-shrink-0',
            open && 'rotate-180'
          )}
        />
      </button>

      {/* Dropdown */}
      {open && (
        <div className="absolute right-0 top-full mt-1 z-50 w-64 rounded-lg border border-[hsl(var(--border))] bg-[hsl(var(--card))] shadow-xl animate-slide-in">
          <div className="p-2">
            <p className="px-2 py-1 text-[10px] font-semibold tracking-widest uppercase text-muted-foreground">
              Available Models
            </p>
            {error && (
              <div className="flex items-center gap-1.5 mx-2 my-1 p-2 rounded bg-red-950/40 border border-red-900 text-xs text-red-400">
                <AlertCircle className="w-3.5 h-3.5 flex-shrink-0" />
                {error}
              </div>
            )}
            {availableModels.length === 0 && !isLoadingModels ? (
              <p className="px-2 py-3 text-xs text-muted-foreground text-center">
                No models registered
              </p>
            ) : (
              availableModels.map((model) => (
                <ModelOption
                  key={model.name}
                  model={model}
                  isActive={model.name === activeModel}
                  isActivating={activating === model.name}
                  onActivate={handleActivate}
                />
              ))
            )}
          </div>
        </div>
      )}

      {/* Backdrop */}
      {open && (
        <div
          className="fixed inset-0 z-40"
          onClick={() => setOpen(false)}
        />
      )}
    </div>
  )
}

interface ModelOptionProps {
  model: ModelInfo
  isActive: boolean
  isActivating: boolean
  onActivate: (model: ModelInfo) => void
}

function ModelOption({ model, isActive, isActivating, onActivate }: ModelOptionProps) {
  return (
    <button
      onClick={() => onActivate(model)}
      disabled={isActivating}
      className={cn(
        'w-full flex items-start gap-2 px-2 py-2 rounded-md text-left transition-colors',
        isActive
          ? 'bg-primary/10 text-primary'
          : 'hover:bg-accent text-foreground',
        isActivating && 'opacity-60 cursor-not-allowed'
      )}
    >
      <div className="mt-0.5 flex-shrink-0">
        {isActivating ? (
          <Loader2 className="w-3.5 h-3.5 animate-spin text-primary" />
        ) : isActive ? (
          <CheckCircle2 className="w-3.5 h-3.5 text-primary" />
        ) : model.loaded ? (
          <div className="w-3.5 h-3.5 rounded-full border-2 border-emerald-600" />
        ) : (
          <div className="w-3.5 h-3.5 rounded-full border-2 border-slate-600" />
        )}
      </div>
      <div className="min-w-0 flex-1">
        <p className="text-sm font-medium truncate">{model.name}</p>
        <p className="text-[11px] text-muted-foreground">
          {model.framework} •{' '}
          {model.loaded ? (
            <span className="text-emerald-500">Loaded</span>
          ) : (
            <span className="text-slate-500">Not loaded</span>
          )}
          {model.total_inferences > 0 && ` • ${model.total_inferences.toLocaleString()} runs`}
        </p>
      </div>
    </button>
  )
}

import { useCallback, useEffect, useRef } from 'react'
import { SlidersHorizontal, RotateCcw } from 'lucide-react'
import * as SliderPrimitive from '@radix-ui/react-slider'
import { cn } from '@/lib/utils'
import { useModelStore } from '@/store/useModelStore'
import type { InferenceSettings } from '@/types'

// ── Styled Radix Slider wrapper ──────────────────────────────────────────────

interface SliderProps {
  value: number
  min: number
  max: number
  step: number
  onChange: (value: number) => void
  disabled?: boolean
}

function Slider({ value, min, max, step, onChange, disabled }: SliderProps) {
  return (
    <SliderPrimitive.Root
      className={cn(
        'relative flex items-center select-none touch-none w-full h-5',
        disabled && 'opacity-40 cursor-not-allowed'
      )}
      value={[value]}
      min={min}
      max={max}
      step={step}
      onValueChange={([v]) => onChange(v)}
      disabled={disabled}
    >
      <SliderPrimitive.Track className="relative h-1.5 w-full grow overflow-hidden rounded-full bg-slate-700">
        <SliderPrimitive.Range className="absolute h-full bg-primary rounded-full" />
      </SliderPrimitive.Track>
      <SliderPrimitive.Thumb
        className={cn(
          'block h-4 w-4 rounded-full border-2 border-primary bg-background',
          'ring-offset-background transition-colors',
          'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2',
          'disabled:pointer-events-none',
          'shadow-md hover:border-primary/80'
        )}
      />
    </SliderPrimitive.Root>
  )
}

// ── Slider row ───────────────────────────────────────────────────────────────

interface SliderRowProps {
  label: string
  description: string
  value: number
  min: number
  max: number
  step: number
  displayValue: string
  onChange: (value: number) => void
  onCommit?: (value: number) => void
  disabled?: boolean
}

function SliderRow({
  label,
  description,
  value,
  min,
  max,
  step,
  displayValue,
  onChange,
  disabled,
}: SliderRowProps) {
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-medium text-foreground">{label}</p>
          <p className="text-[11px] text-muted-foreground">{description}</p>
        </div>
        <span className="text-sm font-mono font-semibold text-primary min-w-[48px] text-right">
          {displayValue}
        </span>
      </div>
      <Slider
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={onChange}
        disabled={disabled}
      />
      <div className="flex justify-between text-[10px] text-muted-foreground">
        <span>{min}</span>
        <span>{max}</span>
      </div>
    </div>
  )
}

// ── Main panel ───────────────────────────────────────────────────────────────

const DEFAULTS: InferenceSettings = {
  confidenceThreshold: 0.5,
  stride: 1,
  scale: 1.0,
}

interface SlidersPanelProps {
  /** Called when a value changes (after debounce) — use to push to backend */
  onSettingsChange?: (settings: InferenceSettings) => void
  disabled?: boolean
  className?: string
}

export function SlidersPanel({ onSettingsChange, disabled, className }: SlidersPanelProps) {
  const settings = useModelStore((s) => s.settings)
  const updateSettings = useModelStore((s) => s.updateSettings)

  // Debounce backend sync — only fire after 300ms of no change
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const handleChange = useCallback(
    (key: keyof InferenceSettings, value: number) => {
      updateSettings({ [key]: value })

      if (onSettingsChange) {
        if (debounceRef.current) clearTimeout(debounceRef.current)
        debounceRef.current = setTimeout(() => {
          onSettingsChange({ ...settings, [key]: value })
        }, 300)
      }
    },
    [settings, updateSettings, onSettingsChange]
  )

  // Cleanup debounce on unmount
  useEffect(() => {
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current)
    }
  }, [])

  function handleReset() {
    updateSettings(DEFAULTS)
    if (onSettingsChange) onSettingsChange(DEFAULTS)
  }

  return (
    <div className={cn('rounded-lg border border-[hsl(var(--border))] bg-[hsl(var(--card))] p-4', className)}>
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <SlidersHorizontal className="w-4 h-4 text-primary" />
          <span className="text-sm font-semibold text-foreground">Inference Settings</span>
        </div>
        <button
          onClick={handleReset}
          disabled={disabled}
          className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors disabled:opacity-40"
          title="Reset to defaults"
        >
          <RotateCcw className="w-3 h-3" />
          Reset
        </button>
      </div>

      {/* Sliders */}
      <div className="space-y-5">
        <SliderRow
          label="Confidence Threshold"
          description="Minimum detection confidence"
          value={settings.confidenceThreshold}
          min={0.1}
          max={0.99}
          step={0.01}
          displayValue={`${(settings.confidenceThreshold * 100).toFixed(0)}%`}
          onChange={(v) => handleChange('confidenceThreshold', v)}
          disabled={disabled}
        />

        <SliderRow
          label="Stride"
          description="Sliding window stride (px)"
          value={settings.stride}
          min={1}
          max={8}
          step={1}
          displayValue={`${settings.stride}px`}
          onChange={(v) => handleChange('stride', v)}
          disabled={disabled}
        />

        <SliderRow
          label="Scale Factor"
          description="Image pyramid scale"
          value={settings.scale}
          min={0.5}
          max={2.0}
          step={0.1}
          displayValue={`×${settings.scale.toFixed(1)}`}
          onChange={(v) => handleChange('scale', v)}
          disabled={disabled}
        />
      </div>
    </div>
  )
}

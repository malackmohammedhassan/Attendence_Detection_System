import { create } from 'zustand'
import { devtools } from 'zustand/middleware'
import type {
  ModelInfo,
  InferenceSettings,
  InferenceResult,
  ConnectionStatus,
} from '@/types'

// ─────────────────────────────────────────────────────────────────────────────
//  Model store — active model selection, settings, inference state
// ─────────────────────────────────────────────────────────────────────────────

interface ModelState {
  // Registry
  availableModels: ModelInfo[]
  activeModel: string | null
  isLoadingModels: boolean

  // Inference settings (synced to backend via API)
  settings: InferenceSettings

  // Latest inference result
  latestResult: InferenceResult | null
  isInferring: boolean
  inferenceError: string | null

  // WebSocket connection to inference stream
  streamStatus: ConnectionStatus

  // Actions
  setAvailableModels: (models: ModelInfo[]) => void
  setActiveModel: (name: string | null) => void
  setIsLoadingModels: (loading: boolean) => void
  updateSettings: (patch: Partial<InferenceSettings>) => void
  setLatestResult: (result: InferenceResult) => void
  setIsInferring: (inferring: boolean) => void
  setInferenceError: (error: string | null) => void
  setStreamStatus: (status: ConnectionStatus) => void
  resetInference: () => void
}

export const useModelStore = create<ModelState>()(
  devtools(
    (set) => ({
      // ── initial state ────────────────────────────────────────────────
      availableModels: [],
      activeModel: null,
      isLoadingModels: false,

      settings: {
        confidenceThreshold: 0.5,
        stride: 1,
        scale: 1.0,
      },

      latestResult: null,
      isInferring: false,
      inferenceError: null,
      streamStatus: 'disconnected',

      // ── actions ──────────────────────────────────────────────────────
      setAvailableModels: (models) => set({ availableModels: models }),

      setActiveModel: (name) => set({ activeModel: name }),

      setIsLoadingModels: (loading) => set({ isLoadingModels: loading }),

      updateSettings: (patch) =>
        set((state) => ({
          settings: { ...state.settings, ...patch },
        })),

      setLatestResult: (result) =>
        set({ latestResult: result, inferenceError: null }),

      setIsInferring: (inferring) => set({ isInferring: inferring }),

      setInferenceError: (error) =>
        set({ inferenceError: error, isInferring: false }),

      setStreamStatus: (status) => set({ streamStatus: status }),

      resetInference: () =>
        set({
          latestResult: null,
          isInferring: false,
          inferenceError: null,
          streamStatus: 'disconnected',
        }),
    }),
    { name: 'ModelStore' }
  )
)

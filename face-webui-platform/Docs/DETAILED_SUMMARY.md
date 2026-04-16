# ML Evaluation Dashboard — Detailed Summary

> **Status:** ✅ Fully built and running  
> **Backend:** `http://localhost:8000` · **Frontend:** `http://localhost:5173`  
> **Verified:** 31 API routes registered · 0 TypeScript errors · Python syntax clean

For IEEE manuscript-ready implementation and evidence workflow (including generated result charts from persisted DB runs), see [IEEE_IMPLEMENTATION_GUIDE.md](IEEE_IMPLEMENTATION_GUIDE.md).

For a concise system architecture and block diagram, see [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md).

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Technology Stack](#2-technology-stack)
3. [Directory Structure](#3-directory-structure)
4. [Backend — Architecture & Modules](#4-backend--architecture--modules)
5. [Frontend — Architecture & Modules](#5-frontend--architecture--modules)
6. [Full API Reference](#6-full-api-reference)
7. [WebSocket Channels](#7-websocket-channels)
8. [Feature Deep-Dive by Phase](#8-feature-deep-dive-by-phase)
9. [Data Flow Diagrams](#9-data-flow-diagrams)
10. [How to Run](#10-how-to-run)

---

## 1. Project Overview

The **ML Evaluation Dashboard** is a production-quality web application that provides a unified interface to:

- **Monitor** a running ML face-detection model in real-time (live webcam stream with bounding boxes)
- **Train** a Scratch CNN model with live epoch charts and raw log streaming
- **Benchmark** models with latency profiling and a full 200-frame evaluation that measures Precision, Recall, F1, False Positives, CPU usage, and Memory usage
- **Export** all results (benchmark CSV via pandas, JSON reports, training history)
- **Inspect** live system metrics (CPU %, Memory MB, FPS, inference latency) via WebSocket

It integrates directly with the ML engine project at `../realtime-face-detection-dl/`, running the real `train_scratch_cnn.py` training script and loading `.pth` model checkpoints.

---

## 2. Technology Stack

### Backend

| Concern             | Library/Version                        |
| ------------------- | -------------------------------------- |
| Web framework       | FastAPI 0.128                          |
| ASGI server         | Uvicorn                                |
| Data validation     | Pydantic v2                            |
| Async               | asyncio + ThreadPoolExecutor           |
| Training subprocess | subprocess.Popen (non-blocking stdout) |
| System metrics      | psutil                                 |
| Image processing    | numpy, OpenCV (cv2), Pillow            |
| WebSockets          | FastAPI native WebSocket               |
| Logging             | Python logging + custom LogStreamer    |
| Python version      | 3.11                                   |

### Frontend

| Concern          | Library/Version                            |
| ---------------- | ------------------------------------------ |
| UI framework     | React 18                                   |
| Build tool       | Vite 5                                     |
| Language         | TypeScript (strict mode)                   |
| Routing          | React Router v6                            |
| State management | Zustand                                    |
| HTTP client      | Axios                                      |
| WebSocket client | Custom `WsClient` class (native WebSocket) |
| Charts           | Recharts                                   |
| Styling          | Tailwind CSS + ShadCN/Radix-UI primitives  |
| Icons            | Lucide React                               |

---

## 3. Directory Structure

```
face-webui-platform/
├── DETAILED_SUMMARY.md          ← this file
├── SHORT_SUMMARY.md             ← quick reference
│
├── backend/
│   ├── requirements.txt
│   └── app/
│       ├── main.py              ← FastAPI app factory + lifespan
│       ├── config.py            ← Pydantic Settings (env vars, paths)
│       ├── websocket_manager.py ← Channel-based WS manager
│       ├── routes/
│       │   ├── benchmark.py     ← POST /benchmark, /benchmark/run, /compare
│       │   ├── export.py        ← GET /export/* (JSON + pandas CSV)
│       │   ├── inference.py     ← POST /inference/frame, /batch
│       │   ├── live.py          ← WS /ws/live (real-time webcam stream)
│       │   ├── metrics.py       ← GET /metrics/system, /inference, /training
│       │   └── train.py         ← POST /train, WS /train/ws/logs
│       ├── services/
│       │   ├── benchmark_engine.py  ← BenchmarkEngine (latency + full eval)
│       │   ├── inference_service.py ← InferenceService wrapper
│       │   ├── log_streamer.py      ← LogStreamer (bridges Python logging → WS)
│       │   ├── model_manager.py     ← ModelManager (load/activate/run models)
│       │   └── training_service.py  ← TrainingService (subprocess-based)
│       └── utils/
│           ├── metrics_collector.py  ← In-memory inference + training metrics
│           └── performance_tracker.py ← Async system metric sampler (1 Hz)
│
└── frontend/
    ├── vite.config.ts           ← Proxy /api/v1 + /ws → localhost:8000
    ├── tailwind.config.ts
    ├── tsconfig.json
    └── src/
        ├── App.tsx              ← BrowserRouter + all routes
        ├── main.tsx             ← React entry point
        ├── types/index.ts       ← All shared TypeScript interfaces
        ├── services/
        │   ├── api.ts           ← Axios API client (all endpoints)
        │   └── websocket.ts     ← WsClient class + WS factory functions
        ├── store/
        │   ├── useMetricsStore.ts  ← System metrics, train logs, live state
        │   └── useModelStore.ts    ← Available models, active model
        ├── hooks/
        │   └── useLiveStream.ts ← WebSocket-based live stream hook
        ├── pages/
        │   ├── Dashboard.tsx    ← System metrics charts + model info
        │   ├── Training.tsx     ← Training control (charts + live logs)
        │   ├── LiveCompare.tsx  ← Webcam feed with detection overlay
        │   └── Benchmark.tsx    ← Full evaluation + latency benchmarks
        └── components/
            ├── layout/
            │   ├── Layout.tsx   ← Top-level Layout wrapper
            │   ├── Header.tsx   ← App header
            │   └── Sidebar.tsx  ← Navigation sidebar
            ├── BenchmarkReport.tsx  ← Charts + metric cards for results
            ├── LogsPanel.tsx        ← Auto-scroll log panel (color-coded)
            ├── ModelSelector.tsx    ← Model dropdown with status indicators
            ├── PerformanceCharts.tsx ← CPU/Memory/FPS line charts
            ├── SlidersPanel.tsx     ← Confidence threshold + stride sliders
            └── WebcamViewer.tsx     ← Webcam capture + canvas overlay
```

---

## 4. Backend — Architecture & Modules

### `main.py` — App Factory

- FastAPI application with `lifespan` context manager for startup/shutdown
- Registers all 6 routers under `/api/v1` prefix, except `live.router` (mounted at root for `WS /ws/live`)
- Startup sequence: PerformanceTracker → LogStreamer → WebSocket heartbeat timer
- Shutdown sequence: cancels all async tasks, stops tracker and streamer
- Additional routes: `GET /health`, `GET /status`
- CORS configured for `localhost:5173`

### `config.py` — Settings

```python
class Settings(BaseSettings):
    ml_engine_root: Path           # points to ../realtime-face-detection-dl/
    models_dir: Path               # ml_engine_root / "models"
    data_dir: Path                 # ml_engine_root / "data"
    benchmark_warmup_runs: int = 5
    benchmark_measure_runs: int = 50
    log_level: str = "INFO"
    debug: bool = False
```

### `websocket_manager.py` — Channel-Based WebSocket Manager

- Maintains a dict of `channel_name → List[WebSocket]`
- `connect(ws, channel)` / `disconnect(ws, channel)` — lifecycle management
- `broadcast(channel, message_dict)` — JSON-serialise and send to all subscribers
- `broadcast_bytes(channel, data)` — binary frame broadcast
- Built-in channels: `"training"`, `"train-logs"`, `"metrics"`, `"benchmark"`, `"logs"`

### `services/model_manager.py` — ModelManager

- Registry of all loaded models (dict keyed by name)
- `load(name)` — loads `.pth` scratch CNN or MTCNN from `models_dir`
- `set_active(name)` / `get_active_name()`
- `run_active(frame)` → `(List[Dict], latency_ms)` — inference + timing
- `available_names()` — returns list of discovered model files

### `services/training_service.py` — TrainingService (Phase 4)

Completely subprocess-based — no fake training:

```python
# Executes:
python scripts/train_scratch_cnn.py \
  --epochs N --batch-size B --lr LR --optimizer OPT --wd WD
```

- `subprocess.Popen(stdout=PIPE, stderr=STDOUT, bufsize=1)` — line-buffered
- Background thread reads stdout line by line (non-blocking)
- `_EPOCH_RE` regex parses: `"Epoch [ 1/50] | Train Loss: X, Acc: Y | Val Loss: Z, Acc: W"`
- Per-job `deque(maxlen=2000)` ring buffer stores all stdout lines
- `cancel_active()` — SIGTERM → 5s wait → SIGKILL
- `set_log_callback(fn)` — fan-out to WebSocket channel

### `services/benchmark_engine.py` — BenchmarkEngine (Phase 5)

Two benchmark modes:

**Mode 1 — Latency-only** (original):

- `run(model_name, warmup=5, measure=50)` — synchronous from caller perspective
- Generates random synthetic frames (`np.random.default_rng`)
- `LatencyStats` object with mean/median/p50/p90/p95/p99/stdev/min/max

**Mode 2 — Full 200-frame evaluation** (new):

- `start_full(model_name, num_frames=200)` — fire-and-forget, returns `run_id`
- Returns immediately, background task via `asyncio.create_task`
- Loads REAL images from `data/val/` or `data/lfw_raw/`, falls back to synthetic
- Runs inference on each frame, measures:
  - Per-frame latency (ms) → builds LatencyStats
  - Detections vs ground truth (LFW-style: 1 face per real image, 0 for synthetic)
  - TP / FP / FN counting → Precision, Recall, F1
  - CPU % + memory RSS via `psutil.Process` (sampled every 10 frames)
- Updates `progress_pct` in real-time (frontend polls `/benchmark/results/{id}`)
- Final `BenchmarkResult` includes all 8 new fields

### `services/log_streamer.py` — LogStreamer

- Python `logging.Handler` that captures all log records at INFO+ level
- Serialises to `WsLogEntry` JSON and broadcasts on `"logs"` WS channel
- Attached at app startup via `logging.root.addHandler(streamer)`

### `services/inference_service.py` — InferenceService

- Wraps `model_manager` with additional preprocessing
- Handles base64 image decoding, resize, BGR conversion before inference
- Returns structured `InferenceResult` with bounding boxes

### `utils/metrics_collector.py` — MetricsCollector

- In-memory stores: `inference_records[]`, `training_epochs[]`
- `record_inference(model_name, latency, detections, ...)` — called after every inference
- `get_inference_stats(model_name)` → `InferenceStats` (mean/p50/p95/p99 latency, mean confidence, fps estimate)
- `get_training_history(model_name)` / `get_best_epoch(model_name)`
- Thread-safe with `threading.Lock`

### `utils/performance_tracker.py` — PerformanceTracker

- Async task sampling CPU, memory, process stats at 1 Hz
- Maintains a 60-entry rolling history (1 minute)
- Broadcasts `WsMetricsTick` on `"metrics"` WS channel every tick
- `snapshot()` → current `SystemSnapshot`
- `history_as_dicts(n)` → last-N samples as list

---

## 5. Frontend — Architecture & Modules

### State Management (Zustand)

**`useModelStore`**

```typescript
{ availableModels: ModelInfo[], activeModel: string | null,
  setAvailableModels, setActiveModel }
```

**`useMetricsStore`**

```typescript
{
  // System metrics (WS-driven)
  systemSnapshot: SystemSnapshot | null,
  cpuHistory: ChartPoint[], memHistory: ChartPoint[], fpsHistory: ChartPoint[],
  // Training
  trainingJob: TrainingJob | null, trainLogs: string[], trainLogsJobId: string | null,
  trainLogsWsStatus: ConnectionStatus,
  // Live stream
  liveResult: WsLiveResult | null, liveFps: number, liveCpu: number, liveMemMb: number,
}
```

### WebSocket Architecture

`websocket.ts` defines a reusable `WsClient` class:

```typescript
class WsClient {
  connect(url, onOpen, onMessage, onClose, onError): void;
  send(data): void;
  disconnect(): void;
}
```

Factory functions built on `WsClient`:

- `createMetricsWs()` → subscribes to `/ws/metrics`
- `createTrainingWs()` → subscribes to `/ws/training`
- `createTrainLogsWs()` → subscribes to `/ws/train/ws/logs`
- `createLiveStreamWs()` → subscribes to `/ws/live`
- `createLogsWs()` → subscribes to `/ws/logs`

### `hooks/useLiveStream.ts`

Custom hook that:

1. Opens camera via `navigator.mediaDevices.getUserMedia`
2. Connects to `/ws/live` WebSocket
3. Sends a `{type: "ready", model}` message to start
4. Reads video frames via `canvas.getContext("2d").drawImage`
5. Encodes each frame as JPEG blob and sends over WS
6. Receives `live_result` messages → renders bounding boxes on canvas overlay
7. Handles `metrics_tick` → updates CPU/memory/FPS in store
8. Cleans up on unmount (closes WS + camera)

### `components/WebcamViewer.tsx`

- Two `<canvas>` elements: capture canvas (hidden) + overlay canvas (visible)
- `requestAnimationFrame` loop for frame capture at 15fps (configurable)
- Bounding box rendering with confidence labels
- Shows real-time FPS / Latency / CPU / Memory badges

### `pages/Training.tsx`

- **Charts tab:** Epoch train/val loss + accuracy with Recharts `LineChart`
- **Logs tab:** `LogsPanel` component with auto-scroll, color-coded lines (ERROR=red, WARNING=yellow, default=green)
- **History tab:** Table of all epoch results
- Training runs via `POST /api/v1/train/start`
- Progress via WS `epoch` messages
- Logs via WS `train_log` / `train_log_batch` messages
- `Stop` button → `POST /api/v1/train/stop` + SIGTERM/SIGKILL chain
- Progress bar colored: running=blue, completed=green, failed=red, cancelled=yellow

### `pages/Benchmark.tsx`

Two tabs:

**Full Evaluation tab:**

- Model dropdown (from `useModelStore`)
- `num_frames` input (default 200)
- `Run Full Evaluation` button → `POST /api/v1/benchmark`
- Returns `run_id` immediately, then polls `GET /benchmark/results/{run_id}` every 1s
- Live progress bar with percentage
- 8 metric cards: FPS, Mean Latency, Precision, Recall, F1, False Positives, CPU avg %, Memory avg MB
- Latency percentile bar chart (P50/P95/P99 via Recharts `BarChart`)
- CPU/Memory bar chart
- Export JSON (`/export/benchmark/all`) + Export CSV (`/export/benchmark/all/csv`) buttons

**Latency / Compare tab:**

- Multi-model checkbox list
- Warmup + Measure frames inputs + frame size inputs
- Single model → `POST /benchmark/run` → `BenchmarkReport` with Latency stats
- Multi-model → `POST /benchmark/compare` → comparison bar chart + ranked cards

### `pages/Dashboard.tsx`

- Live system metrics tiles (CPU %, Memory MB, uptime)
- `PerformanceCharts` with 60-second rolling line chart for CPU, Memory, FPS
- Loaded models list with activation controls
- Total inference count + error stats from `MetricsSummary`

### `pages/LiveCompare.tsx`

- `WebcamViewer` for live detection
- `SlidersPanel` for confidence threshold + stride/scale adjustments
- `ModelSelector` to switch active model

---

## 6. Full API Reference

### System

| Method | Path      | Description                                                 |
| ------ | --------- | ----------------------------------------------------------- |
| GET    | `/health` | Lightweight health probe — returns `{status: "ok"}`         |
| GET    | `/status` | Full status: uptime, model, ws connections, system snapshot |

### Inference (`/api/v1/inference`)

| Method | Path                                | Description                                         |
| ------ | ----------------------------------- | --------------------------------------------------- |
| GET    | `/inference/models`                 | List all registered models with metadata            |
| GET    | `/inference/models/active`          | Get the currently active model                      |
| POST   | `/inference/models/{name}/activate` | Set a model as active                               |
| DELETE | `/inference/models/{name}`          | Unload a model                                      |
| POST   | `/inference/frame`                  | Run inference on a multipart/form-data image upload |
| POST   | `/inference/frame/base64`           | Run inference on a base64-encoded image             |
| POST   | `/inference/batch`                  | Run inference on multiple frames                    |

### Training (`/api/v1/train`)

| Method | Path                     | Description                                           |
| ------ | ------------------------ | ----------------------------------------------------- |
| POST   | `/train/start`           | Start a new training job (returns job_id immediately) |
| POST   | `/train/stop`            | Stop the active training job (SIGTERM → SIGKILL)      |
| GET    | `/train/status`          | Get status of the active training job                 |
| GET    | `/train/active`          | Get full active job details                           |
| GET    | `/train/{job_id}`        | Get a specific job by ID                              |
| POST   | `/train/{job_id}/cancel` | Cancel a specific job                                 |
| POST   | `/train`                 | Convenience alias for `/train/start`                  |
| WS     | `/api/v1/train/ws/logs`  | Stream training stdout/stderr log lines               |

### Benchmark (`/api/v1/benchmark`)

| Method | Path                              | Description                                                  |
| ------ | --------------------------------- | ------------------------------------------------------------ |
| POST   | `/benchmark`                      | **Start 200-frame full evaluation** → returns `run_id` (202) |
| POST   | `/benchmark/run`                  | Synchronous latency benchmark (blocks until done)            |
| POST   | `/benchmark/compare`              | Compare multiple models, returns ranked table                |
| GET    | `/benchmark/results`              | List all results (filter by `?model_name=`)                  |
| GET    | `/benchmark/results/{run_id}`     | Get specific result with `progress_pct`                      |
| GET    | `/benchmark/results/model/{name}` | All results for a model                                      |

### Export (`/api/v1/export`)

| Method | Path                                | Description                                          |
| ------ | ----------------------------------- | ---------------------------------------------------- |
| GET    | `/export/benchmark/all`             | All benchmark results as structured JSON download    |
| GET    | `/export/benchmark/all/csv`         | All benchmark results as flat CSV (pandas-based)     |
| GET    | `/export/benchmark/{run_id}`        | Single benchmark result as JSON                      |
| GET    | `/export/metrics/{model_name}`      | Inference metrics JSON download                      |
| GET    | `/export/metrics/{model_name}/csv`  | Inference metrics CSV download                       |
| GET    | `/export/training/{model_name}`     | Training epoch history JSON                          |
| GET    | `/export/training/{model_name}/csv` | Training epoch history CSV                           |
| GET    | `/export/report`                    | Full system report (all stats + metrics + benchmark) |

### Metrics (`/api/v1/metrics`)

| Method | Path                                      | Description                                 |
| ------ | ----------------------------------------- | ------------------------------------------- |
| GET    | `/metrics/system`                         | Current system snapshot                     |
| GET    | `/metrics/system/history`                 | Last 60s history                            |
| GET    | `/metrics/summary`                        | Full summary: system + inference + training |
| GET    | `/metrics/inference`                      | All model inference stats                   |
| GET    | `/metrics/inference/{model_name}`         | Single model inference stats                |
| GET    | `/metrics/inference/{model_name}/history` | Per-inference records                       |
| GET    | `/metrics/training/{model_name}`          | Training epoch history                      |
| GET    | `/metrics/training/{model_name}/best`     | Best epoch record                           |
| POST   | `/metrics/reset`                          | Reset all or specific model metrics         |

---

## 7. WebSocket Channels

| Endpoint                   | Channel      | Messages                                                    | Direction     |
| -------------------------- | ------------ | ----------------------------------------------------------- | ------------- |
| `WS /ws/live`              | N/A (direct) | `ready`, `live_result`, `metrics_tick`, `no_model`, `error` | Bidirectional |
| `WS /api/v1/train/ws/logs` | `train-logs` | `train_log_batch` (on connect), `train_log` (live)          | Server→Client |
| `WS /ws/metrics`           | `metrics`    | `metrics_tick` (1 Hz), `heartbeat` (30s)                    | Server→Client |
| `WS /ws/training`          | `training`   | `epoch`, `status_change`, `job_state`, `job_complete`       | Server→Client |
| `WS /ws/logs`              | `logs`       | `log`, `log_replay`                                         | Server→Client |

### `/ws/live` — Live Session Architecture

4 coordinated async tasks inside `LiveSession`:

1. **Frame reader** — receives JPEG binary frames from browser, puts into `asyncio.Queue(maxsize=2)` (drop-oldest)
2. **Frame processor** — dequeues frame, runs inference via `model_manager.run_active()`, broadcasts `live_result`
3. **Metrics emitter** — every 1 second, samples `psutil` and broadcasts `metrics_tick` with `{cpu, memory_mb, fps, timestamp}`
4. **Heartbeat** — every 30 seconds, sends `{type: "pong"}`

### `train_log` Message Schema

```jsonc
// Single line (live)
{ "type": "train_log", "job_id": "uuid", "line": "Epoch [ 1/50] ...", "ts": 1234567890.0 }

// Batch (sent once on connect to replay history)
{ "type": "train_log_batch", "job_id": "uuid", "lines": ["line1", "line2", ...] }
```

---

## 8. Feature Deep-Dive by Phase

### Phase 1 — Backend Foundation

Created 20 Python files from scratch:

- All 6 route modules
- All 5 service modules
- 2 utility modules
- `main.py`, `config.py`, `websocket_manager.py`

Key design decisions:

- Pydantic v2 for all request/response models
- `AsyncContextManager` lifespan pattern (not deprecated `on_event`)
- All services as module-level singletons (thread-safe where needed)
- Settings auto-discover ML engine root via relative path calculation

### Phase 2 — Frontend Foundation

Created full React 18 + Vite + TypeScript + Tailwind project:

- `vite.config.ts` with proxy for `/api/v1` and `/ws` → `localhost:8000`
- All 4 pages wired up in router
- `tailwind.config.ts` with CSS variables for theming
- `WsClient` class with auto-reconnect capability
- Zustand stores fully typed

### Phase 3 — Real-Time WebSocket Streaming

**The hardest engineering challenge:** Streaming live webcam frames bidirectionally over WebSocket without blocking the event loop.

Solution:

- Browser: `requestAnimationFrame` captures at 15 fps, encodes to JPEG (quality 60%), sends as binary over WS
- Backend: `asyncio.Queue(maxsize=2)` with drop-oldest — if processor is slow, old frames are discarded
- 4 async tasks coordinate via shared state (no thread locks needed, all asyncio)
- Canvas overlay renders bounding boxes by mapping normalized coords back to display pixels
- Memory-leak-free: cleanup runs in React `useEffect` teardown

### Phase 4 — Training Control

**Challenge:** Existing `training_service.py` was stub/fake. Required full subprocess re-implementation.

Real implementation:

```python
proc = subprocess.Popen(
    ["python", "scripts/train_scratch_cnn.py", "--epochs", ...],
    cwd=settings.ml_engine_root,
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    text=True, bufsize=1,  # line-buffered
)
```

- Background daemon thread reads lines without blocking FastAPI event loop
- Regex `r"Epoch\s*\[\s*(\d+)/(\d+)\].*Train Loss:\s*([\d.]+).*Acc:\s*([\d.]+).*Val Loss:\s*([\d.]+).*Acc:\s*([\d.]+)"` parses epoch lines
- `set_log_callback(fn)` allows train route to fan lines to WS channel in real-time
- Frontend `LogPanel` auto-scrolls to bottom using `scrollTop = scrollHeight`

### Phase 5 — Benchmark Engine

**New capability:** End-to-end evaluation with detection quality metrics.

The LFW-style GT heuristic:

- Real images (from dataset) → ground truth = 1 face per image
- Synthetic frames → ground truth = 0 faces
- For each frame: `TP = min(det, gt)`, `FP = max(0, det - gt)`, `FN = max(0, gt - det)`
- At the end: `Precision = TP / (TP + FP)`, `Recall = TP / (TP + FN)`, `F1 = 2*P*R / (P+R)`

This is not IoU-based (no ground truth boxes available), but gives a meaningful quality signal for face detection.

Progress polling flow:

```
Frontend POST /benchmark → run_id
  ↓ immediately
Frontend polls GET /benchmark/results/{run_id} every 1s
  ↓ result.progress_pct increases 0 → 100
  ↓ result.status changes PENDING → RUNNING → COMPLETED
Frontend renders progress bar + metric cards from result
```

---

## 9. Data Flow Diagrams

### Live Stream Flow

```
Browser Camera
    ↓ JPEG binary (15 fps)
WS /ws/live (FastAPI)
    ↓ asyncio.Queue(maxsize=2)
Frame Processor Task
    ↓ model_manager.run_active(frame)
    ↓ returns (detections, latency_ms)
WS send "live_result" JSON
    ↓
Browser canvas overlay (bounding boxes)

Metrics Task (1 Hz)
    ↓ psutil.cpu_percent(), .memory_info()
WS send "metrics_tick" JSON
    ↓
Zustand store → React re-render
```

### Training Flow

```
Frontend POST /train/start
    ↓ TrainingService.start_job(params)
    ↓ subprocess.Popen([python, train_scratch_cnn.py, ...])
    ↓ Background Thread reads proc.stdout line by line
    ↓ set_log_callback → broadcast on WS channel "train-logs"
    ↓ parse epoch line → broadcast on WS channel "training"
    ↓
Frontend WS /train/ws/logs → LogPanel displays raw lines
Frontend WS /training → epoch charts update
```

### Benchmark Full Eval Flow

```
Frontend POST /benchmark → run_id (202 Accepted)
    ↓
asyncio.create_task(_run_full_task)
    ↓ ThreadPoolExecutor (_sync_full_benchmark)
    ↓ Load images from data/val/ or synthetic fallback
    ↓ for each frame:
    │   model_manager.run_active(frame)
    │   TP/FP/FN counting
    │   psutil sampling (every 10 frames)
    │   result.progress_pct += 100/N
    ↓ build LatencyStats, compute P/R/F1
    ↓ result.status = COMPLETED
    ↓
Frontend polls GET /benchmark/results/{run_id} (1s interval)
    ↓ progress bar updates
    ↓ on status=completed → render 8 metric cards + charts
```

---

## 10. How to Run

### Prerequisites

- Python 3.11 with: `fastapi`, `uvicorn`, `pydantic`, `psutil`, `numpy`, `opencv-python-headless`, `Pillow`
- Node.js 18+ with npm

### Start Backend

```bash
cd face-webui-platform/backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Start Frontend

```bash
cd face-webui-platform/frontend
npm install   # first time only
npm run dev
```

### Access

- **Frontend:** http://localhost:5173
- **Backend API:** http://localhost:8000
- **API Docs (Swagger):** http://localhost:8000/docs
- **Health check:** http://localhost:8000/health

### Environment Variables (optional)

Create `backend/.env` to override defaults:

```env
ML_ENGINE_ROOT=../../realtime-face-detection-dl
LOG_LEVEL=INFO
DEBUG=false
BENCHMARK_WARMUP_RUNS=5
BENCHMARK_MEASURE_RUNS=50
```

---

_Built in 5 phases. All 31 API routes verified working. 0 TypeScript compilation errors._

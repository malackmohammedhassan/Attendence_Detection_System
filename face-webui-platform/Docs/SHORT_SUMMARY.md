# ML Evaluation Dashboard — Short Summary

> **Status:** ✅ Running | Backend `:8000` · Frontend `:5173` · 31 API routes · 0 TS errors

For IEEE manuscript preparation with reproducible figures/tables from real benchmark DB values, see [IEEE_IMPLEMENTATION_GUIDE.md](IEEE_IMPLEMENTATION_GUIDE.md).

---

## What It Is

A full-stack web dashboard for real-time monitoring, training, and benchmarking of ML face-detection models. Integrates directly with `../realtime-face-detection-dl/`.

---

## Tech Stack

| Layer        | Stack                                                                    |
| ------------ | ------------------------------------------------------------------------ |
| **Backend**  | Python 3.11 · FastAPI · Uvicorn · Pydantic v2 · psutil · numpy · OpenCV  |
| **Frontend** | React 18 · Vite · TypeScript · Zustand · Axios · Recharts · Tailwind CSS |
| **Realtime** | WebSocket (FastAPI native + custom WsClient class)                       |
| **Training** | `subprocess.Popen` running real `train_scratch_cnn.py`                   |

---

## Four Pages

| Page             | What It Does                                                                                        |
| ---------------- | --------------------------------------------------------------------------------------------------- |
| **Dashboard**    | Live CPU/Memory/FPS charts, loaded model list, inference stats                                      |
| **Training**     | Start/stop training · live epoch charts · color-coded raw log stream                                |
| **Live Compare** | Webcam feed with bounding box overlay · real-time latency, FPS, CPU badges                          |
| **Benchmark**    | Full 200-frame eval (P/R/F1/FP/CPU/Mem) · latency profiling · multi-model compare · CSV/JSON export |

---

## Key Features

- **Real-time live stream** — JPEG frames sent over WebSocket at 15 fps, bounding boxes drawn on canvas overlay
- **Real training** — actual `subprocess.Popen` of `train_scratch_cnn.py`, non-blocking stdout streaming to browser in real-time
- **Full benchmark eval** — 200 frames, real dataset images when available, measures Precision/Recall/F1/False Positives, CPU/Memory averages
- **Progress polling** — benchmark fires async, frontend polls `GET /benchmark/results/{run_id}` every 1s for live progress bar
- **Pandas CSV export** — all benchmark results exportable as flat CSV or structured JSON
- **System metrics** — `psutil`-based CPU/memory tracker broadcasts at 1 Hz to all connected clients

---

## Project Structure (key files)

```
face-webui-platform/
├── backend/app/
│   ├── main.py                 — FastAPI app + lifespan
│   ├── config.py               — Settings (paths, env vars)
│   ├── websocket_manager.py    — Channel-based WS broadcast
│   ├── routes/                 — benchmark, export, inference, live, metrics, train
│   ├── services/               — benchmark_engine, model_manager, training_service, ...
│   └── utils/                  — metrics_collector, performance_tracker
└── frontend/src/
    ├── pages/                  — Dashboard, Training, LiveCompare, Benchmark
    ├── components/             — WebcamViewer, BenchmarkReport, LogsPanel, ...
    ├── store/                  — useMetricsStore, useModelStore (Zustand)
    ├── services/               — api.ts (Axios), websocket.ts (WsClient)
    └── types/index.ts          — All shared TypeScript interfaces
```

---

## WebSocket Endpoints

| WS Endpoint                | Streams                                              |
| -------------------------- | ---------------------------------------------------- |
| `WS /ws/live`              | Webcam frames → detections + metrics (bidirectional) |
| `WS /api/v1/train/ws/logs` | Training stdout log lines                            |
| `WS /ws/metrics`           | System metrics tick (1 Hz)                           |
| `WS /ws/training`          | Epoch progress events                                |

---

## How to Start

```bash
# Backend
# Activate the venv
& d:\PROJECTS\Collage_Projects\SC_Project\realtime-face-detection-dl\.venv\Scripts\Activate.ps1

# Then start the backend
cd d:\PROJECTS\Collage_Projects\SC_Project\face-webui-platform\backend
python -m uvicorn app.main:app --port 8000 --reload

# Frontend
cd face-webui-platform/frontend
npm run dev
```

Open **http://localhost:5173** → API docs at **http://localhost:8000/docs**

---

_See [DETAILED_SUMMARY.md](DETAILED_SUMMARY.md) for full architecture, API reference, and phase-by-phase breakdown._

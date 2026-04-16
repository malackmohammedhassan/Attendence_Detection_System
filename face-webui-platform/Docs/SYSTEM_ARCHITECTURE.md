# System Architecture

## Overview

This project is a full-stack ML evaluation dashboard for face-detection workflows. The system is split into four main layers:

1. Frontend UI for monitoring, training, benchmarking, and live comparison.
2. FastAPI backend that exposes REST and WebSocket endpoints.
3. Service layer that handles inference, training, benchmarking, logging, and metrics.
4. ML engine integration that runs the real training and inference workload.

The design is centered on real-time communication. REST is used for one-shot actions such as starting jobs and fetching reports, while WebSocket channels stream live frames, metrics, logs, and training progress.

## System Architecture

### Frontend Layer

- React 18 application built with Vite and TypeScript.
- Pages for Dashboard, Training, Live Compare, and Benchmark.
- Zustand stores for shared state such as active model, system metrics, and training jobs.
- Axios for REST calls and native WebSocket clients for streaming updates.

### Backend Layer

- FastAPI application with lifecycle management in `main.py`.
- Route modules for inference, training, benchmark, metrics, export, internal, and live streaming.
- WebSocket manager for channel-based broadcasts.
- CORS and health/status endpoints for operational visibility.

### Service Layer

- `ModelManager` loads and activates models.
- `InferenceService` prepares frames and runs inference.
- `TrainingService` launches the external training script and streams logs.
- `BenchmarkEngine` runs latency and full-evaluation benchmarks.
- `MetricsCollector` and `PerformanceTracker` keep runtime and historical metrics.
- `LogStreamer` forwards application logs to connected clients.

### ML Engine Layer

- Real model training is executed through the external `train_scratch_cnn.py` script.
- Model checkpoints and datasets are read from the ML engine workspace.
- The backend treats the ML engine as the source of truth for training and inference execution.

## Block Diagram

```mermaid
flowchart LR
    U[User] --> B[Browser UI\nReact + Vite]

    B -->|REST| R[FastAPI Routes]
    B -->|WebSocket| W[WebSocket Channels]

    R --> I[Inference Service]
    R --> T[Training Service]
    R --> X[Benchmark Engine]
    R --> M[Metrics & Export APIs]

    W --> WM[WebSocket Manager]
    WM --> LS[Live Stream]
    WM --> TL[Training Logs]
    WM --> MT[Metrics Stream]
    WM --> LG[Application Logs]

    I --> MM[Model Manager]
    X --> MM
    T --> ME[External ML Engine\ntrain_scratch_cnn.py]
    MM --> ME

    ME --> D[(Models / Datasets)]
    X --> P[Performance Tracker]
    M --> C[Metrics Collector]
    LG --> O[Server Logs]

    LS --> B
    TL --> B
    MT --> B
    LG --> B
    X --> B
    I --> B
```

## Internal Block View

```mermaid
flowchart TB
    subgraph Frontend
        D1[Dashboard]
        D2[Training]
        D3[Live Compare]
        D4[Benchmark]
        S1[Zustand Stores]
        S2[API + WebSocket Clients]
    end

    subgraph Backend
        M1[main.py]
        M2[Route Modules]
        M3[Service Modules]
        M4[WebSocket Manager]
        M5[Metrics Utilities]
    end

    subgraph ML_Engine[External ML Engine]
        E1[Training Script]
        E2[Model Checkpoints]
        E3[Dataset Files]
    end

    D1 --> S1
    D2 --> S1
    D3 --> S2
    D4 --> S2

    S2 --> M1
    M1 --> M2
    M2 --> M3
    M2 --> M4
    M3 --> M5
    M3 --> E1
    M3 --> E2
    E1 --> E3
```

## Data Paths

- Browser to backend: HTTP requests for job control, model activation, and exports.
- Browser to backend: WebSocket frames for live webcam streaming and streamed results.
- Backend to ML engine: subprocess execution for training and model evaluation.
- Backend to frontend: live metrics, logs, training progress, and benchmark progress.

## Why This Structure Works

- It keeps the UI responsive because long-running work is pushed into background services.
- It separates real-time streams from request/response APIs.
- It isolates ML execution in the external engine so the dashboard stays thin and stable.
- It makes the system easy to extend with new models, metrics, and benchmark modes.

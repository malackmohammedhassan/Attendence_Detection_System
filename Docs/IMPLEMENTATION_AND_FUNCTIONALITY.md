# Implementation and Functionality Guide

This document consolidates how the two projects in this workspace are built, how they behave at runtime, and where the important implementation pieces live.

Projects covered:

- [attendance*system_soft_computing*](../attendance_system_soft_computing_)
- [face-webui-platform](../face-webui-platform)

For the face web UI project, the more specific deep-dive docs already exist under [face-webui-platform/Docs](../face-webui-platform/Docs). This file ties both systems together in one place.

---

## 1. At a Glance

### attendance*system_soft_computing*

A desktop-oriented face recognition attendance system that combines face detection, face embedding, biometric vault storage, live attendance marking, batch video processing, vault management, diagnostics, and summary generation.

### face-webui-platform

A full-stack web dashboard for model monitoring, live inference, training orchestration, benchmarking, export, and real-time metrics/log streaming.

The two projects solve different problems but share the same general pattern:

- detect or infer on face-related data
- compare results against stored model/vault data
- persist outputs into local files or database-backed reports
- present the results through a user interface

---

## 2. Project 1: attendance*system_soft_computing*

### 2.1 Purpose

This project implements a face-based attendance workflow. The system enrolls people into a biometric vault, detects faces from a camera or video stream, matches them against stored embeddings, and writes attendance records to CSV files. It also includes diagnostic and management tools for reviewing the vault and generated attendance data.

### 2.2 Main Entry Points

- [main.py](../attendance_system_soft_computing_/main.py) is the live kiosk application used for real-time attendance capture.
- [enrollmentupd.py](../attendance_system_soft_computing_/enrollmentupd.py) handles student enrollment and biometric vault creation.
- [process_videoupd.py](../attendance_system_soft_computing_/process_videoupd.py) processes video files in batch and generates attendance logs.
- [vault_manager.py](../attendance_system_soft_computing_/vault_manager.py) is the vault inspection and deletion utility.
- [generate_summary.py](../attendance_system_soft_computing_/generate_summary.py) aggregates attendance CSV files into a summary report.
- [diagnostic_view.py](../attendance_system_soft_computing_/diagnostic_view.py) is a debugging view for embedding and match inspection.

### 2.3 How the System Works

#### Enrollment flow

1. The enrollment script opens the camera and detects the face in the frame.
2. The face is cropped and converted into an embedding vector.
3. The new embedding is compared with existing vault entries to avoid duplicates.
4. If the person is accepted as new, the system stores the biometric vector and related metadata in the vault.
5. Captured reference images are stored under [data/known_faces](../attendance_system_soft_computing_/data/known_faces).

In practice, this creates a persistent biometric identity record that later attendance sessions can reuse.

#### Live attendance flow

1. The kiosk UI reads frames from a live camera.
2. Each frame is run through the face detector.
3. Detected faces are embedded and compared against the biometric vault.
4. When a match is found, the system marks the person as present and writes the result to [data/live_attendance.csv](../attendance_system_soft_computing_/data/live_attendance.csv).
5. The UI shows the recognized identity, status, and time-slot state.

This mode is designed for interactive attendance capture during a live session.

#### Batch video processing flow

1. A video file is opened and processed frame by frame.
2. The system detects faces in the video stream and skips redundant frames where appropriate for performance.
3. Each recognized face is matched against the vault.
4. Attendance rows are appended to a date-based CSV file in [data](../attendance_system_soft_computing_/data).
5. Duplicate marking is avoided so the same person is not repeatedly logged for the same video.

This mode is useful for CCTV-style or offline attendance review.

#### Vault management and diagnostics

- The vault manager allows viewing enrolled identities and removing records.
- The diagnostic view helps inspect whether the detector and embedding pipeline are separating identities correctly.
- The summary generator collects attendance CSV files and produces a consolidated report for review or submission.

### 2.4 Core Implementation Pieces

#### Face detection and recognition

The project combines a detector and an embedding model:

- the detector localizes faces in frames
- the recognition model converts each detected face into a vector representation
- similarity between vectors determines identity matches

This split is important because detection answers where the face is, while recognition answers who the face belongs to.

#### Biometric vault

The biometric vault is stored as a serialized file under [data/biometric_vault.pkl](../attendance_system_soft_computing_/data/biometric_vault.pkl). It acts as the system of record for enrolled identities. The enrollment script writes to it, and the live and batch processing scripts read from it.

#### Attendance outputs

Attendance is stored in CSV form instead of a database. That keeps the workflow simple and portable:

- live attendance writes to [data/live_attendance.csv](../attendance_system_soft_computing_/data/live_attendance.csv)
- batch processing creates date-stamped CSV reports in [data](../attendance_system_soft_computing_/data)
- the summary script aggregates the CSV files into a more complete report

### 2.5 Key Folders and Files

- [data](../attendance_system_soft_computing_/data) stores attendance logs, the biometric vault, and reference images.
- [models](../attendance_system_soft_computing_/models) stores the face detector weights.
- [examples](../attendance_system_soft_computing_/examples) contains sample media used by the project.
- [tests](../attendance_system_soft_computing_/tests) contains automated checks.
- [docs](../attendance_system_soft_computing_/docs) contains the packaged project documentation.

### 2.6 Runtime Characteristics

- The system is built around camera or video input rather than web APIs.
- Attendance persistence is file-based, which makes it easy to inspect and export.
- The project depends on computer vision and deep learning libraries for both detection and embedding extraction.
- Performance tuning is driven by frame skipping, face-size checks, and duplicate suppression.

### 2.7 Operational Notes

- Enrollment quality matters because the vault is only as good as the captured reference data.
- Matching thresholds and duplicate checks are the main controls that prevent false attendance records.
- The summary and diagnostics tools exist to make the pipeline easier to verify and audit.

---

## 3. Project 2: face-webui-platform

### 3.1 Purpose

This project is a browser-based control center for face-detection model workflows. It provides real-time inference monitoring, training orchestration, benchmarking, metrics dashboards, logs, and export tools through a FastAPI backend and React frontend.

### 3.2 Main Entry Points

- [backend/app/main.py](../face-webui-platform/backend/app/main.py) creates and configures the FastAPI application.
- [backend/app/routes](../face-webui-platform/backend/app/routes) contains REST and WebSocket route modules.
- [backend/app/services](../face-webui-platform/backend/app/services) contains the business logic for inference, training, benchmarking, logging, and model management.
- [frontend/src/main.tsx](../face-webui-platform/frontend/src/main.tsx) is the React entry point.
- [frontend/src/App.tsx](../face-webui-platform/frontend/src/App.tsx) wires up the application routes and layout.

### 3.3 How the System Works

#### Backend request flow

1. The frontend sends REST requests for one-shot actions such as starting training, activating a model, or requesting benchmark/export data.
2. The backend route layer validates the request and hands it to the relevant service.
3. The service layer performs the actual work, such as launching a subprocess, running inference, or collecting metrics.
4. Results are returned to the frontend immediately or streamed through WebSockets when the job is long-running.

This separation keeps the web UI responsive while the backend handles heavier work.

#### Live inference flow

1. The browser opens a WebSocket connection to the live inference endpoint.
2. Webcam frames are captured and sent to the server.
3. The server runs the frame through the active model.
4. The backend streams detections, latency, and related metrics back to the browser.
5. The UI overlays detections on the camera feed and updates charts or counters.

#### Training flow

1. The user submits a training request from the Training page.
2. The backend starts a training job and returns a job identifier.
3. A subprocess runs the external training script used by the ML engine.
4. Stdout and stderr are streamed back to the UI as logs and training progress.
5. The frontend renders epoch history, progress state, and final job status.

#### Benchmark flow

1. The user starts a latency or evaluation benchmark from the Benchmark page.
2. The backend runs the selected model through the benchmark engine.
3. The engine measures latency and, in the full evaluation mode, aggregates metrics such as precision, recall, F1, FPS, and memory usage.
4. Results are persisted and can be exported as JSON or CSV.
5. The frontend presents the benchmark report and comparison view.

### 3.4 Backend Architecture

#### Application bootstrap

The FastAPI app sets up:

- lifecycle management for startup and shutdown
- CORS for browser access
- route registration for inference, live streaming, training, metrics, export, benchmark, and internal health checks
- background services for performance tracking and log streaming

#### Service layer responsibilities

- model management loads and activates available models
- inference service prepares frames and executes inference
- training service runs the external training job and streams logs
- benchmark engine measures performance and evaluation quality
- metrics collector stores runtime and historical metrics
- performance tracker samples system statistics in the background
- log streamer forwards logs to connected clients
- websocket manager coordinates channel-based message delivery

This structure keeps the route layer thin and makes the runtime behavior easier to reason about.

### 3.5 Frontend Architecture

#### Pages

- [frontend/src/pages/Dashboard.tsx](../face-webui-platform/frontend/src/pages/Dashboard.tsx) shows system status and summary metrics.
- [frontend/src/pages/Training.tsx](../face-webui-platform/frontend/src/pages/Training.tsx) manages training jobs and live logs.
- [frontend/src/pages/LiveCompare.tsx](../face-webui-platform/frontend/src/pages/LiveCompare.tsx) presents the live webcam inference view.
- [frontend/src/pages/Benchmark.tsx](../face-webui-platform/frontend/src/pages/Benchmark.tsx) handles benchmarking and result review.

#### Shared UI structure

- [frontend/src/components/layout](../face-webui-platform/frontend/src/components/layout) contains the app shell, header, and sidebar.
- [frontend/src/components](../face-webui-platform/frontend/src/components) contains reusable panels for charts, logs, model selection, sliders, and webcam rendering.
- [frontend/src/store](../face-webui-platform/frontend/src/store) keeps shared app state synchronized across pages.
- [frontend/src/services](../face-webui-platform/frontend/src/services) provides the REST and WebSocket clients.

### 3.6 Key Backend Files

- [backend/app/config.py](../face-webui-platform/backend/app/config.py) centralizes settings and environment-driven paths.
- [backend/app/websocket_manager.py](../face-webui-platform/backend/app/websocket_manager.py) manages channel subscriptions and broadcast delivery.
- [backend/app/routes/train.py](../face-webui-platform/backend/app/routes/train.py) handles training jobs.
- [backend/app/routes/live.py](../face-webui-platform/backend/app/routes/live.py) streams live inference.
- [backend/app/routes/benchmark.py](../face-webui-platform/backend/app/routes/benchmark.py) handles evaluation runs and comparisons.
- [backend/app/routes/export.py](../face-webui-platform/backend/app/routes/export.py) generates downloadable reports.
- [backend/app/routes/metrics.py](../face-webui-platform/backend/app/routes/metrics.py) exposes live and historical metrics.
- [backend/app/routes/inference.py](../face-webui-platform/backend/app/routes/inference.py) runs single-frame and batch inference.

### 3.7 Data, Outputs, and Persistence

- [backend/results](../face-webui-platform/backend/results) stores benchmark run data and related artifacts.
- [backend/logs](../face-webui-platform/backend/logs) stores runtime logs.
- [backend/exports](../face-webui-platform/backend/exports) stores generated exports and figures.
- the frontend build output is produced in [frontend/dist](../face-webui-platform/frontend/dist)

The backend also depends on an external ML engine workspace for model assets and training execution.

### 3.8 Runtime Characteristics

- The frontend is reactive and state-driven, not server-rendered.
- WebSockets are used where continuous updates matter more than polling.
- Long-running operations are isolated behind background jobs or subprocesses.
- The application is structured to support ongoing evaluation and comparison of multiple model configurations.

---

## 4. Relationship Between the Two Projects

The projects are separate, but they follow similar engineering principles:

- both use face-related computer vision workflows
- both transform raw image/video input into structured outputs
- both persist results for later review
- both separate processing logic from presentation logic

The main difference is the deployment model:

- attendance*system_soft_computing* is a local/desktop workflow centered on camera input and CSV-based attendance records
- face-webui-platform is a web application centered on orchestration, monitoring, and reporting

If you want a single project narrative, the first system is the operational attendance engine, and the second system is the monitoring and experimentation dashboard.

---

## 5. Where to Read More

- [face-webui-platform/Docs/DETAILED_SUMMARY.md](../face-webui-platform/Docs/DETAILED_SUMMARY.md)
- [face-webui-platform/Docs/SYSTEM_ARCHITECTURE.md](../face-webui-platform/Docs/SYSTEM_ARCHITECTURE.md)
- [face-webui-platform/Docs/IEEE_IMPLEMENTATION_GUIDE.md](../face-webui-platform/Docs/IEEE_IMPLEMENTATION_GUIDE.md)
- [attendance*system_soft_computing*/README.md](../attendance_system_soft_computing_/README.md)
- [attendance*system_soft_computing*/vit_attendance guide.txt](../attendance_system_soft_computing_/vit_attendance%20guide.txt)

---

## 6. Short Summary

- attendance*system_soft_computing* implements attendance capture from live camera or video, stores identities in a biometric vault, and writes attendance to CSV files.
- face-webui-platform implements a FastAPI and React control center for training, live inference, benchmarking, metrics, and export.
- This document serves as the top-level explanation of how both systems work and how their main pieces fit together.

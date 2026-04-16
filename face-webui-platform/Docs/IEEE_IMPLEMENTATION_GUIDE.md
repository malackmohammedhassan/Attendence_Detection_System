# IEEE Paper Implementation and Evidence Guide

This guide is for writing an IEEE-style paper with reproducible, real project evidence from this repository.
No synthetic values are introduced in this document. All numeric values below come from:

- `backend/results/dashboard.db` table `benchmark_results`
- generated artifacts in `backend/exports/ieee/`

## 1. What Is Already Real in This Project

1. Training pipeline is real subprocess execution (`train_scratch_cnn.py`) through backend service code.
2. Benchmark pipeline stores real run results to SQLite (`backend/results/dashboard.db`).
3. Export routes already provide JSON/CSV/HTML benchmark exports.
4. Frontend pages are real and wired to backend APIs and WebSocket streams.

## 2. Components and Their Actual Implementation Roles

Use this mapping in the paper "Implementation" section.

1. API bootstrap and lifecycle: `backend/app/main.py`
2. Benchmark APIs: `backend/app/routes/benchmark.py`
3. Export APIs and HTML report: `backend/app/routes/export.py`
4. Benchmark execution engine and metric logic: `backend/app/services/benchmark_engine.py`
5. Model loading/inference bridge: `backend/app/services/model_manager.py`
6. Training subprocess orchestration: `backend/app/services/training_service.py`
7. WebSocket channel manager: `backend/app/websocket_manager.py`
8. Frontend pages: `frontend/src/pages/Dashboard.tsx`, `frontend/src/pages/Training.tsx`, `frontend/src/pages/LiveCompare.tsx`, `frontend/src/pages/Benchmark.tsx`
9. Frontend benchmark chart component: `frontend/src/components/BenchmarkReport.tsx`

## 3. Real Benchmark Snapshot (Current DB State)

Data extraction timestamp basis: latest rows in `benchmark_results` with `status='completed'`.

### 3.1 Latest full-evaluation runs (`is_full_eval=1`, `frames_evaluated=200`)

| run_id                               |  model_name | precision | recall |     f1 | false_positives | latency_mean_ms | latency_p95_ms | avg_fps | cpu_avg | memory_avg_mb |
| ------------------------------------ | ----------: | --------: | -----: | -----: | --------------: | --------------: | -------------: | ------: | ------: | ------------: |
| 606834a2-180d-4635-8a29-4caff51a0aa3 |       mtcnn |    0.0472 | 0.0495 | 0.0483 |             101 |         106.877 |        201.186 |    9.35 |   605.4 |        1201.7 |
| a20c2f0a-acfe-4cce-aad8-5206d0bdbf07 | scratch_cnn |    0.0000 | 0.0000 | 0.0000 |             661 |         366.908 |        580.049 |    2.72 |   691.8 |        1345.6 |
| e78b32ac-1727-4cc2-8522-c21bc70fc3e5 |       mtcnn |    0.0472 | 0.0495 | 0.0483 |             101 |          95.238 |        193.908 |   10.49 |   708.7 |        1387.7 |
| 72b104af-8760-4e42-a2e4-48439a2219f2 | scratch_cnn |    0.0000 | 0.0000 | 0.0000 |            2324 |         331.997 |        344.094 |    3.01 |   700.8 |        1506.4 |
| 3c49ebd9-70b3-43dc-8a2b-c032d1aa0284 | scratch_cnn |    0.0000 | 0.0000 | 0.0000 |             679 |         208.957 |        222.246 |    4.78 |   750.7 |        1038.7 |

### 3.2 Model-level aggregate (all completed runs)

| model_name  | runs | latency_mean_ms_avg | latency_p95_ms_avg | avg_fps_avg | precision_avg | recall_avg | f1_avg |
| ----------- | ---: | ------------------: | -----------------: | ----------: | ------------: | ---------: | -----: |
| scratch_cnn |    8 |             148.971 |            182.789 |      13.348 |        0.0000 |     0.0000 | 0.0000 |
| mtcnn       |    3 |              99.929 |            169.285 |      10.023 |        0.0472 |     0.0495 | 0.0483 |

## 4. Generate IEEE Results Figures and Tables (Reproducible)

### 4.1 One command

From `backend/`:

```bash
python scripts/generate_ieee_results.py
```

### 4.2 Generated outputs

1. `backend/exports/ieee/latest_completed_runs.csv`
2. `backend/exports/ieee/latest_full_eval_runs.csv`
3. `backend/exports/ieee/model_summary.csv`
4. `backend/exports/ieee/figure_latency_mean_ms.png`
5. `backend/exports/ieee/figure_latency_p95_ms.png`
6. `backend/exports/ieee/figure_f1_full_eval.png`
7. `backend/exports/ieee/figure_cpu_memory_full_eval.png`
8. `backend/exports/ieee/figure_fps_vs_latency_scatter.png`

### 4.3 Script source

`backend/scripts/generate_ieee_results.py`

The script reads only `benchmark_results` rows with `status='completed'`. No hard-coded metric values exist in the script.

## 5. How to Prepare System Architecture Figure

Primary architecture text/diagram source:

- `Docs/SYSTEM_ARCHITECTURE.md`

Instructions:

1. Open `Docs/SYSTEM_ARCHITECTURE.md` in VS Code preview.
2. Export or capture the Mermaid block diagram at full width.
3. Save as `Docs/figures/architecture_overview.png`.
4. In paper caption, cite source as: "Generated from repository architecture definition in SYSTEM_ARCHITECTURE.md".

Recommended caption:

"Figure X. End-to-end architecture of the face-webui-platform showing REST and WebSocket interaction between React frontend, FastAPI backend services, and ML engine components."

## 6. How to Prepare Web UI Figures (Real UI, No Mockups)

Capture these pages from the running application:

1. Dashboard page (`/`)
2. Training page (`/training`)
3. Live Compare page (`/live`)
4. Benchmark page (`/benchmark`)

Capture protocol:

1. Start backend and frontend.
2. Keep browser zoom at 100% and set a fixed viewport (for example 1920x1080).
3. Ensure visible timestamp and run identifiers where possible.
4. Save raw screenshots without editing overlays.
5. Store under `Docs/figures/` with names:
   - `ui_dashboard.png`
   - `ui_training.png`
   - `ui_live_compare.png`
   - `ui_benchmark.png`

## 7. How to Place Results in IEEE Paper

Use this structure in your Results section:

1. Experimental setup
2. Latency performance
3. Accuracy-related metrics (Precision/Recall/F1)
4. Resource usage (CPU/Memory)
5. Comparative discussion

Minimum evidence package to include as figures/tables:

1. Table from `latest_full_eval_runs.csv`
2. Bar chart `figure_latency_mean_ms.png`
3. Bar chart `figure_latency_p95_ms.png`
4. Bar chart `figure_f1_full_eval.png`
5. Resource chart `figure_cpu_memory_full_eval.png`
6. Scatter plot `figure_fps_vs_latency_scatter.png`

## 8. Integrity Checklist (Anti-Fabrication)

Before final paper submission:

1. Re-run `python scripts/generate_ieee_results.py`.
2. Confirm figure values match CSV values exactly.
3. Cross-check run IDs in paper tables against `benchmark_results` table.
4. Keep exported CSVs in appendix/supplementary material.
5. Do not report metrics for runs with `status != completed`.

## 9. Reproducibility Notes for Reviewers

You can add this statement in the paper:

"All quantitative results are generated directly from persisted benchmark logs in a project SQLite database (`dashboard.db`) using an open script (`generate_ieee_results.py`) that emits the exact CSV tables and PNG charts used in this manuscript."

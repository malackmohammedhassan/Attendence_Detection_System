# Run Guide: Face WebUI Platform

This guide is for the web dashboard module inside face-webui-platform.

## Project Path

face-webui-platform

## 1. Prerequisites

- Docker Desktop with Docker Compose
- Optional local run tools: Python 3.11+, Node.js 20+

## 2. Recommended Method: Docker Compose

Open terminal in face-webui-platform and run:

```bash
docker compose up --build
```

Access:

- Frontend UI: http://localhost:5173
- Backend API: http://localhost:8000
- API docs: http://localhost:8000/docs

Stop services:

```bash
docker compose down
```

## 3. Optional Environment Overrides

Set values before compose run (or via .env next to docker-compose.yml):

- ML_ENGINE_ROOT (default: ../realtime-face-detection-dl)
- FASTAPI_DEBUG
- BACKEND_PORT
- FRONTEND_PORT
- VITE_API_BASE_URL
- VITE_WS_BASE_URL

Example:

```bash
VITE_API_BASE_URL=http://localhost:8000 docker compose up --build
```

## 4. Local Development Method (Without Docker)

### Backend

Open terminal in face-webui-platform/backend:

```bash
pip install -r requirements.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend

Open terminal in face-webui-platform/frontend:

```bash
npm install
npm run dev
```

Frontend default dev URL:

- http://localhost:5173

## 5. Sanity Check Endpoints

- Health: http://localhost:8000/health
- Status: http://localhost:8000/status

## 6. Reference Docs

- Architecture and implementation evidence:
  - Docs/IEEE_IMPLEMENTATION_GUIDE.md
  - Docs/SYSTEM_ARCHITECTURE.md
  - Docs/DETAILED_SUMMARY.md

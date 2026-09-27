# AERIS: Adaptive Environmental Risk & Intelligence System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-009688.svg)](https://fastapi.tiangolo.com)
[![Next.js 14](https://img.shields.io/badge/Next.js-14.2-black.svg)](https://nextjs.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

AERIS is an advanced, research-oriented environmental intelligence platform engineered to predict daily Air Quality Index (AQI), model non-linear pollutant rate-of-change volatility, and provide transparent, interpretable forecasts for urban regions across India.

The platform unites 1D Convolutional Neural Networks (1D-CNN), Bidirectional Long Short-Term Memory networks (BiLSTM), temporal attention, and domain-informed fuzzy logic attention modulators alongside a typed REST API and an interactive web dashboard.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Technology Stack](#technology-stack)
- [Repository Structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Environment Configuration](#environment-configuration)
- [Local Backend Setup](#local-backend-setup)
- [Local Frontend Setup](#local-frontend-setup)
- [Docker Compose Setup](#docker-compose-setup)
- [Health Check & Verification](#health-check--verification)
- [Research & Architecture Specifications](#research--architecture-specifications)
- [Troubleshooting](#troubleshooting)

---

## Project Overview

* **Primary Objective:** Forecast next-day ($t+1$) continuous AQI from 30 historical consecutive days of multivariate pollutant measurements ($\text{PM}_{2.5}, \text{PM}_{10}, \text{NO}_2, \text{SO}_2, \text{CO}, \text{O}_3$, etc.).
* **Flagship Architecture:** CNN-BiLSTM with Fuzzy Logic-Modulated Temporal Attention (`cnn_bilstm_fuzzy_attn`).
* **Benchmarking Suite:** Evaluates 8 distinct model families under strict chronological 70/15/15 splits.
* **Explainability:** Multi-pollutant feature attribution, temporal attention heatmaps, and counterfactual "what-if" policy scenario simulations.

---

## Technology Stack

| Layer | Technologies |
| :--- | :--- |
| **Backend & ML** | Python 3.10+, FastAPI, Pydantic v2, Uvicorn, PyTorch 2.x, NumPy, Pandas, Scikit-learn, XGBoost, LightGBM |
| **Frontend & UI** | Next.js 14 (App Router), React 18, TypeScript, Tailwind CSS, shadcn/ui, Recharts, Lucide Icons |
| **Data & Storage** | Apache Parquet, Joblib, PyTorch Tensor checkpoints (`.pt`), JSON |
| **DevOps & QA** | Docker, Docker Compose, pytest, pytest-asyncio, Ruff, ESLint, Prettier |

---

## Repository Structure

```
AERIS/
├── frontend/                     # Next.js 14 Web Dashboard
│   ├── app/                      # App router pages & layouts
│   ├── components/               # UI components & chart widgets
│   ├── lib/                      # API client & utility functions
│   ├── types/                    # TypeScript interfaces
│   ├── package.json
│   ├── tsconfig.json
│   ├── next.config.mjs
│   └── Dockerfile
│
├── backend/                      # FastAPI REST API & ML Service
│   ├── app/
│   │   ├── api/v1/               # API endpoints & routers
│   │   ├── core/                 # App settings, logging, errors
│   │   ├── schemas/              # Pydantic v2 data models
│   │   ├── services/             # Business & forecasting logic
│   │   ├── data/                 # Ingestion & preprocessing utilities
│   │   └── main.py               # FastAPI application entrypoint
│   ├── tests/                    # Pytest test suite
│   ├── requirements.txt
│   ├── pyproject.toml
│   └── Dockerfile
│
├── data/                         # Historical & Processed Datasets
│   ├── raw/                      # Raw CPCB city_day.csv
│   └── processed/                # Parquet tables & train/val/test splits
│
├── artifacts/                    # Serialized Models & Evaluation Metrics
│   ├── scalers/                  # Fitted Robust/MinMax scalers
│   ├── checkpoints/              # PyTorch (.pt) & GBDT (.joblib) models
│   └── metrics/                  # Benchmark JSON outputs
│
├── docs/                         # Additional technical documentation
├── scripts/                      # Utility and automation scripts
├── docker-compose.yml            # Multi-container orchestration
├── .env.example                  # Environment template
├── .gitignore
├── README.md
│
├── PROJECT_SPEC.md               # Phase 0: Project Specification
├── ARCHITECTURE.md               # Phase 0: System Architecture Design
├── DEVELOPMENT_PLAN.md           # Phase 0: 16-Phase Implementation Plan
├── RESEARCH_BASELINE.md          # Phase 0: Research & Methodology Review
├── MODEL_SPECIFICATION.md        # Phase 0: 8-Model Deep Learning Design
├── API_SPECIFICATION.md          # Phase 0: OpenAPI 3.1 REST Contracts
└── DASHBOARD_SPECIFICATION.md    # Phase 0: UI/UX & Page Wireframes
```

---

## Prerequisites

* **Python:** Version 3.10, 3.11, 3.12, or 3.13
* **Node.js:** Version 18.x, 20.x, or 22.x (with `npm` 9+)
* **Docker & Docker Compose:** Optional for containerized deployment

---

## Environment Configuration

1. Copy `.env.example` to create your local `.env`:
   ```bash
   cp .env.example .env
   ```
2. Key environment variables:
   ```env
   ENVIRONMENT=development
   DEBUG=True
   BACKEND_HOST=0.0.0.0
   BACKEND_PORT=8000
   FRONTEND_PORT=3000
   NEXT_PUBLIC_API_URL=http://localhost:8000
   CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
   ```

---

## Local Backend Setup

1. Navigate to the backend directory or project root:
   ```bash
   pip install -r backend/requirements.txt
   ```
2. Run code formatting and lint checks:
   ```bash
   ruff check backend/
   ```
3. Run automated backend tests:
   ```bash
   pytest backend/
   ```
4. Start the FastAPI development server:
   ```bash
   uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
   ```
5. Access interactive API documentation:
   * Swagger UI: [http://localhost:8000/api/v1/docs](http://localhost:8000/api/v1/docs)
   * ReDoc: [http://localhost:8000/api/v1/redoc](http://localhost:8000/api/v1/redoc)

---

## Local Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```
2. Install npm dependencies:
   ```bash
   npm install
   ```
3. Run TypeScript validation and linting:
   ```bash
   npm run type-check
   npm run lint
   ```
4. Start the Next.js development server:
   ```bash
   npm run dev
   ```
5. Open [http://localhost:3000](http://localhost:3000) in your browser.

---

## Docker Compose Setup

To build and run the entire AERIS platform in isolated containers:

```bash
# Build and start all services in detached mode
docker compose up --build -d

# Check service logs
docker compose logs -f

# Verify service health
docker compose ps

# Stop all containers
docker compose down
```

---

## Health Check & Verification

Once services are running, verify the backend health check endpoint:

```bash
# Using curl
curl http://localhost:8000/api/v1/health

# Expected response:
# {
#   "status": "healthy",
#   "project": "AERIS",
#   "version": "1.0.0",
#   "environment": "development",
#   "uptime_seconds": 12.4,
#   "device": "cpu",
#   "models_loaded": [],
#   "dataset_version": "city_day_v1.0"
# }
```

---

## Research & Architecture Specifications

Detailed research formulations, mathematical models, and UI wireframes are documented in the root specifications:
* [PROJECT_SPEC.md](PROJECT_SPEC.md) — Problem formulation, scope, and requirements.
* [ARCHITECTURE.md](ARCHITECTURE.md) — Architectural diagrams, data pipelines, and serving workflows.
* [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md) — 16-phase milestone execution plan and decisions tracker.
* [RESEARCH_BASELINE.md](RESEARCH_BASELINE.md) — Historical CPCB benchmark methodology and anti-leakage rules.
* [MODEL_SPECIFICATION.md](MODEL_SPECIFICATION.md) — Mathematical formulations for all 8 model families and fuzzy attention.
* [API_SPECIFICATION.md](API_SPECIFICATION.md) — OpenAPI 3.1 schema definitions for all 10 endpoints.
* [DASHBOARD_SPECIFICATION.md](DASHBOARD_SPECIFICATION.md) — Comprehensive visual blueprints for all 9 dashboard views.

---

## Troubleshooting

* **Backend `ModuleNotFoundError: No module named 'backend'`:**
  Ensure you are running commands from the root directory or set `PYTHONPATH=.`.
* **CORS Blocked Requests in Browser:**
  Check `CORS_ORIGINS` in `.env` and verify your frontend is running on an authorized port (default: `http://localhost:3000`).
* **Next.js Port Conflict (Port 3000 already in use):**
  Specify an alternate port via `npm run dev -- -p 3001` or update `FRONTEND_PORT` in `.env`.

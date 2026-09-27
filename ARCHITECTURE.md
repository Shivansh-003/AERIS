# AERIS: System Architecture & Design Specification
**Adaptive Environmental Risk & Intelligence System**
*System Architecture & Data Pipeline Specification*

---

## 1. High-Level System Architecture

AERIS is structured as a modular, three-tier architecture comprising an **Analytical Frontend (Next.js/React)**, a **High-Performance API Service (FastAPI/Python)**, and a **Machine Learning Core (PyTorch/Scikit-learn/XGBoost)** backed by a structured file and database storage layer.

```mermaid
flowchart TD
    subgraph UI ["Client Tier — Next.js 14 / React Dashboard"]
        Dashboard["AERIS Web Dashboard\n(Overview, Map, Forecast, Model Lab, XAI, Fuzzy, Simulator)"]
        State["Client State & TanStack Query"]
        MapEngine["MapLibre GL / Leaflet Engine"]
    end

    subgraph API ["Application Tier — FastAPI Backend (/api/v1)"]
        Router["API Gateway / Routers\n(Health, Predict, Forecast, Explain, Simulate, Metrics)"]
        Middleware["CORS, Rate Limiting, Pydantic Validation"]
        ModelRegistry["Model Artifact Registry & In-Memory Cache"]
        FuzzyEngine["Fuzzy Logic & Attention Engine"]
        XAIEngine["Explainability & Attribution Engine"]
    end

    subgraph MLCore ["Machine Learning & Data Processing Engine"]
        DataPipeline["Data Ingestion & Chronological Window Generator"]
        Preproc["City-Aware Scalers & Imputation Modules"]
        ModelSuite["PyTorch DL Models & GBDT Baselines\n(CNN-BiLSTM-Fuzzy, Attention, BiLSTM, XGBoost, etc.)"]
    end

    subgraph Storage ["Storage & Artifact Tier"]
        RawData["Raw & Processed Data (CSV / Parquet)"]
        ModelWeights["Model Artifacts (.pt, .joblib, .json)"]
        AppDB["Application Store (SQLite / JSON Metadata)"]
    end

    Dashboard <--> |HTTP / JSON REST API| Router
    Router --> Middleware
    Middleware --> ModelRegistry
    Middleware --> FuzzyEngine
    Middleware --> XAIEngine
    ModelRegistry --> ModelSuite
    FuzzyEngine --> ModelSuite
    XAIEngine --> ModelSuite
    MLCore --> Storage
    ModelRegistry <--> Storage
```

---

## 2. Technology Stack Evaluation & Rationale

| Layer | Selected Technology | Alternative Considered | Selection Rationale |
| :--- | :--- | :--- | :--- |
| **Frontend Framework** | **Next.js 14 (App Router) + TypeScript** | Vite + React SPA | Server-side rendering, file-based routing, robust production build tooling, and native TypeScript support. |
| **Styling & Components** | **Tailwind CSS + shadcn/ui** | Material UI / AntD | Lightweight, highly customizable, headless accessible primitives with dark-mode aesthetic support. |
| **Visualization** | **Recharts + Lucide Icons** | Chart.js / D3.js | Declarative React-native SVG chart rendering for complex multi-line, area, and attention heatmaps. |
| **Geospatial Mapping** | **MapLibre GL JS / Leaflet** | Google Maps API | Open-source, high-performance vector rendering for Indian city-level geospatial coordinates without API cost/keys. |
| **Backend Framework** | **FastAPI + Pydantic v2** | Flask / Django | Asynchronous ASGI performance, automatic OpenAPI documentation, strict request/response data validation. |
| **ML Framework** | **PyTorch 2.x** | TensorFlow / Keras | Dynamic computational graphs, seamless custom attention and fuzzy modulation implementations. |
| **Tabular & GBDT** | **XGBoost + LightGBM + Scikit-Learn** | CatBoost | Industry-standard high-performance gradient boosting baselines for time-series tabular comparison. |
| **Data Storage** | **Parquet + SQLite + JSON** | PostgreSQL / Kafka | Low-overhead, zero-maintenance, file-based storage perfectly sized for multi-city historical daily records. |
| **Containerization** | **Docker + Docker Compose** | Kubernetes | Simple, reproducible, portable local and cloud deployment without distributed orchestration overhead. |

---

## 3. Data Ingestion Architecture

The ingestion pipeline handles multi-year, multi-pollutant tabular records from Central Pollution Control Board (CPCB) stations across India.

```mermaid
flowchart LR
    CSV["city_day.csv\n(Raw Observations)"] --> Validator["Schema Validation Engine\n(Pandas / Pydantic)"]
    Validator --> NullCheck["Missingness & Anomaly Audit"]
    NullCheck --> CityPartition["City-Wise Partitioning"]
    CityPartition --> ParquetStore["Partitioned Parquet Lake\n(data/processed/city_day.parquet)"]
```

### Ingestion Specifications
1. **Schema Validation:** Checks for mandatory columns (`City`, `Date`, pollutant columns, `AQI`).
2. **Type Enforcement:** Converts `Date` to `datetime64[ns]` and all pollutant concentrations to `float32`.
3. **Partitioning:** Separates data into discrete city-wise time-series containers to prevent cross-city contamination during subsequent temporal operations.

---

## 4. Data Preprocessing & Feature Engineering Pipeline

Preprocessing enforces strict chronological isolation to prevent future-to-past data leakage.

```mermaid
flowchart TD
    Raw["City-Partitioned Data"] --> Split["Chronological Split (70% Train, 15% Val, 15% Test)"]
    
    subgraph Preproc ["Preprocessing (Fitted ONLY on Train)"]
        Split --> FitScaler["Fit Robust/MinMax Scalers on Train"]
        FitScaler --> Transform["Transform Train, Val, Test Splits"]
        Transform --> Impute["City-Aware Temporal Imputation\n(Linear Interpolation + Fwd/Bwd Fill)"]
    end

    subgraph FeatureEng ["Feature Engineering Engine"]
        Impute --> ROC["Compute Rate-of-Change (ΔX_t = X_t - X_{t-1})"]
        ROC --> Rolling["Compute 7d/14d Rolling Means & Volatility"]
        Rolling --> Cyclic["Cyclic Temporal Encodings (sin/cos Month, Day of Year)"]
    end

    subgraph Windowing ["Window Generation Engine"]
        Cyclic --> SlidingWindow["Sliding Window Extraction\n(Lookback L=30 days, Step=1)"]
        SlidingWindow --> TensorPack["Pack into Tensors\nInput: (B, 30, D), Target: (B, 1)"]
    end
```

---

## 5. Model Training Architecture

The training architecture supports modular execution for all 8 model families with early stopping, learning rate scheduling, gradient clipping, and metric logging.

```mermaid
flowchart TD
    DataModule["PyTorch Dataset & DataLoader\n(TrainLoader, ValLoader, TestLoader)"] --> Trainer["PyTorch Training Loop"]
    
    subgraph ModelPool ["Model Architecture Registry"]
        M1["XGBoost / LightGBM"]
        M2["LSTM / BiLSTM"]
        M3["Transformer Encoder"]
        M4["CNN-BiLSTM"]
        M5["CNN-BiLSTM + Temporal Attention"]
        M6["CNN-BiLSTM + Fuzzy Attention"]
    end

    ModelPool --> Trainer
    Trainer --> Loss["Loss Engine (Huber / Smooth L1 Loss)"]
    Loss --> Optimizer["AdamW Optimizer + CosineAnnealingLR"]
    Optimizer --> Checkpointer["Checkpoint Manager\n(Save Best Val RMSE Model)"]
    Checkpointer --> MetricsLogger["Metrics Logger (JSON / TensorBoard)"]
```

---

## 6. Fuzzy Attention Architecture

The novel **Fuzzy Temporal Attention** engine introduces domain-informed environmental volatility modeling directly into the neural attention computation.

```mermaid
flowchart TD
    subgraph Input ["Temporal Feature Sequence"]
        X["Input Window X_t ∈ R^{30 × D}"]
    end

    subgraph CNN ["Spatial Feature Extraction"]
        X --> Conv1D["1D Convolution (Kernel=3, Filters=64, ReLU)"]
        Conv1D --> MaxPool["Optional MaxPool1D / Dropout"]
    end

    subgraph BiLSTM ["Bidirectional Recurrence"]
        MaxPool --> BLSTM["BiLSTM Layers (Hidden=64 per direction)"]
        BLSTM --> H["Hidden Representations H ∈ R^{30 × 128}"]
    end

    subgraph FuzzyModule ["Fuzzy Modulation Engine"]
        X --> ExtractROC["Extract Pollutant Rates of Change (ΔPM2.5, ΔPM10, ΔNO2)"]
        ExtractROC --> FuzzyMF["Gaussian / Trapezoidal Membership Functions\n(Rising, Stable, Rapidly Surging)"]
        FuzzyMF --> FuzzyAgg["Fuzzy Rule Aggregation → Volatility Weight γ_t ∈ [0.5, 2.0]"]
    end

    subgraph AttentionEngine ["Fuzzy-Modulated Attention"]
        H --> TempScore["Raw Temporal Attention Score e_t = v^T tanh(W H_t + b)"]
        TempScore & FuzzyAgg --> ModulateScore["Modulated Score ẽ_t = e_t · γ_t"]
        ModulateScore --> Softmax["Softmax Normalization → α̃_t"]
        Softmax & H --> ContextVector["Context Vector c = ∑ α̃_t H_t"]
    end

    subgraph Head ["Prediction Head"]
        ContextVector --> Dense1["Dense(64) + LayerNorm + ReLU"]
        Dense1 --> Dropout["Dropout(0.2)"]
        Dropout --> Out["Linear(1) → Predicted AQI_{t+1}"]
    end
```

---

## 7. Model Inference & Serving Architecture

Inference is optimized for fast, deterministic execution within the FastAPI backend service.

```mermaid
sequenceDiagram
    autonumber
    actor User as Client / Dashboard
    participant API as FastAPI Router (/api/v1/predict)
    participant Cache as In-Memory Model Cache
    participant Pre as Preprocessing Engine
    participant Model as PyTorch Inference Engine
    participant Fuzzy as Fuzzy Attention Module
    participant XAI as Explainability Engine

    User->>API: POST /api/v1/predict (City, 30-day sequence or date range, model_id)
    API->>API: Validate Request Payload (Pydantic)
    API->>Cache: Fetch Cached Model & Scaler (Singleton)
    API->>Pre: Impute & Scale 30-day Input Window
    Pre-->>API: Scaled Tensor (1, 30, D)
    API->>Model: Forward Pass
    Model->>Fuzzy: Compute Fuzzy Modulated Attention
    Fuzzy-->>Model: Context Vector & Attention Weights (α̃_t)
    Model-->>API: Predicted AQI (Scalar) & Raw Embeddings
    opt Explainability Requested
        API->>XAI: Compute Feature Attributions & Attention Map
        XAI-->>API: Attribution JSON
    end
    API-->>User: 200 OK Response (Predicted AQI, NAQI Bucket, Confidence Bounds, Attention Weights)
```

---

## 8. Backend Architecture (FastAPI)

The backend follows clean layered architecture principles:

* **Routers (`/api/v1/`):** Route handling, request validation, and HTTP response formatting.
* **Services:** Business logic for forecasting, what-if simulations, fuzzy analysis, and metric retrieval.
* **Core Engine:** Preprocessing transformations, PyTorch model loaders, and mathematical fuzzy logic modules.
* **Schemas:** Pydantic models enforcing strict input and output contracts.

```
backend/
├── app/
│   ├── api/
│   │   ├── v1/
│   │   │   ├── endpoints/
│   │   │   │   ├── health.py
│   │   │   │   ├── cities.py
│   │   │   │   ├── predict.py
│   │   │   │   ├── forecast.py
│   │   │   │   ├── models.py
│   │   │   │   ├── metrics.py
│   │   │   │   ├── explain.py
│   │   │   │   ├── fuzzy.py
│   │   │   │   └── simulate.py
│   │   │   └── api_router.py
│   ├── core/
│   │   ├── config.py
│   │   └── errors.py
│   ├── models/
│   │   ├── model_registry.py
│   │   └── pytorch_models.py
│   ├── schemas/
│   │   ├── predict_schema.py
│   │   ├── forecast_schema.py
│   │   ├── explain_schema.py
│   │   ├── fuzzy_schema.py
│   │   └── metrics_schema.py
│   ├── services/
│   │   ├── forecast_service.py
│   │   ├── fuzzy_service.py
│   │   ├── explain_service.py
│   │   └── simulation_service.py
│   └── main.py
```

---

## 9. Frontend Architecture (Next.js 14 App Router)

The dashboard provides an intuitive, data-dense interface designed for analytical clarity and rapid navigation.

```
frontend/
├── src/
│   ├── app/
│   │   ├── layout.tsx
│   │   ├── page.tsx (Redirect to /overview)
│   │   ├── overview/page.tsx
│   │   ├── map/page.tsx
│   │   ├── forecast/page.tsx
│   │   ├── model-lab/page.tsx
│   │   ├── explainability/page.tsx
│   │   ├── fuzzy/page.tsx
│   │   ├── simulator/page.tsx
│   │   ├── data-explorer/page.tsx
│   │   └── system/page.tsx
│   ├── components/
│   │   ├── common/ (Header, Sidebar, CitySelector, DatePicker, NAQIBadge)
│   │   ├── charts/ (AQITrendChart, AttentionHeatmap, PollutantBarChart, LossCurve)
│   │   ├── map/ (IndiaAirQualityMap, MapTooltip, LayerControl)
│   │   └── ui/ (shadcn primitives: Button, Card, Dialog, Slider, Tabs, etc.)
│   ├── lib/
│   │   ├── api.ts (Typed fetch client)
│   │   ├── constants.ts (NAQI ranges, color mappings)
│   │   └── utils.ts
│   └── types/
│       ├── api.d.ts
│       └── models.d.ts
```

---

## 10. Storage Architecture

```
AERIS Storage Structure:
├── data/
│   ├── raw/
│   │   └── city_day.csv              # Historical CPCB daily dataset
│   ├── processed/
│   │   ├── city_day_cleaned.parquet  # Imputed, validated dataset
│   │   ├── train_windows.npy         # Sliding sequence inputs (N, 30, D)
│   │   ├── val_windows.npy
│   │   ├── test_windows.npy
│   │   └── metadata.json             # Feature names, scaler params, city lists
├── artifacts/
│   ├── scalers/
│   │   └── robust_scaler.joblib      # Fitted feature scalers
│   ├── checkpoints/
│   │   ├── xgboost_baseline.joblib
│   │   ├── lightgbm_baseline.joblib
│   │   ├── lstm.pt
│   │   ├── bilstm.pt
│   │   ├── transformer.pt
│   │   ├── cnn_bilstm.pt
│   │   ├── cnn_bilstm_temp_attn.pt
│   │   └── cnn_bilstm_fuzzy_attn.pt  # Flagship research model
│   └── metrics/
│       ├── benchmark_summary.json    # Consolidated test metrics
│       └── ablation_results.json
```

---

## 11. Deployment Architecture

A single-host Docker Compose orchestrates the frontend and backend services in an isolated virtual bridge network.

```mermaid
flowchart TD
    Client[Web Browser / API Client] -->|Port 3000| NGINX_FE[Next.js Frontend Container]
    Client -->|Port 8000| API_BE[FastAPI Backend Container]
    
    subgraph DockerHost ["Docker Compose Bridge Network"]
        API_BE --> VolumeArtifacts["/app/artifacts (Shared Checkpoint Volume)"]
        API_BE --> VolumeData["/app/data (Processed Data Volume)"]
        NGINX_FE -.->|Internal Proxy /api/v1| API_BE
    end
```

---

## 12. Component Responsibilities & Inter-Component Communication

1. **Preprocessing & Sequence Generator:** Reads raw CSV, handles imputation, applies scaling, and produces unified `(B, 30, D)` arrays.
2. **Model Registry:** Loads PyTorch weights and Scikit-learn/LightGBM pipelines on backend startup; caches models in memory as singletons.
3. **Fuzzy Logic Service:** Computes rate-of-change metrics, evaluates Gaussian/Trapezoidal fuzzy sets, and passes weights to model forward passes.
4. **FastAPI Endpoints:** Receives JSON requests, coordinates feature extraction and model inference, formats responses with NAQI categorization.
5. **Next.js Client:** Queries API via typed client with TanStack Query caching, rendering interactive Recharts and MapLibre visualizations.

---

## 13. Error Handling Strategy & System Resilience

* **Schema Validation Errors:** FastAPI automatically intercepts malformed payloads and returns RFC 7807 compliant `422 Unprocessable Entity` responses.
* **Missing Feature Imputation:** If a single day within a 30-day lookback is missing individual pollutant readings during user inference, the backend applies local spline/linear interpolation rather than failing.
* **Out-of-Distribution Inputs:** What-if simulator clamps pollutant adjustments within physical bounds (e.g., non-negative concentrations, reasonable maximum limits) and returns informative warnings.
* **Model Inference Fallback:** If a requested advanced model fails to execute, the system returns a descriptive `500 Internal Server Error` with diagnostic stack traces logged locally, without crashing the ASGI worker.

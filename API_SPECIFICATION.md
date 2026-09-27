# AERIS: REST API Specification (OpenAPI 3.1)
**Adaptive Environmental Risk & Intelligence System**
*Document Version: 1.0.0 — Phase 0 API Contract*
*Date: September 2026*

---

## 1. Overview & Architectural Principles

The AERIS API provides high-performance, asynchronous REST endpoints for model inference, time-series forecasting, explainability attribution, fuzzy intelligence analysis, and counterfactual simulation.

* **Base URL:** `/api/v1`
* **Protocol:** HTTP/1.1 and HTTP/2 over TLS
* **Data Format:** JSON (`application/json`)
* **Validation Engine:** Pydantic v2 with strict schema enforcement
* **Error Standard:** RFC 7807 Problem Details for HTTP APIs
* **Lifecycle:** Model weights and preprocessors are loaded during ASGI startup into a singleton in-memory `ModelRegistry`.

> [!NOTE] Implementation Status
> This document specifies the future API contract to be implemented in **Phase 13 (FastAPI Backend)**. No endpoints are currently running during Phase 0.

---

## 2. Global Error Handling & Problem Details Schema

All `4xx` and `5xx` errors follow the RFC 7807 standard:

```json
{
  "type": "https://aeris.ai/errors/validation-error",
  "title": "Unprocessable Entity",
  "status": 422,
  "detail": "Input sequence must contain exactly 30 consecutive daily records.",
  "instance": "/api/v1/predict",
  "timestamp": "2026-09-27T18:00:00Z",
  "invalid_params": [
    {
      "name": "sequence",
      "reason": "Length 28 is less than required 30"
    }
  ]
}
```

---

## 3. Endpoints Specification

---

### 1. System Health Check
* **Endpoint:** `GET /api/v1/health`
* **Purpose:** Returns service health, uptime, loaded model count, and GPU/CPU device availability.
* **Requires Trained Model:** No.
* **Data Source Dependency:** Internal runtime telemetry.
* **Response Schema (200 OK):**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "environment": "production",
  "uptime_seconds": 14205.2,
  "device": "cuda:0",
  "models_loaded": [
    "xgboost_baseline",
    "lightgbm_baseline",
    "lstm",
    "bilstm",
    "transformer",
    "cnn_bilstm",
    "cnn_bilstm_temp_attn",
    "cnn_bilstm_fuzzy_attn"
  ],
  "dataset_version": "city_day_v1.0"
}
```

---

### 2. Supported Cities List
* **Endpoint:** `GET /api/v1/cities`
* **Purpose:** Returns the list of Indian cities available in the historical dataset, along with geographic coordinates and date ranges.
* **Requires Trained Model:** No.
* **Data Source Dependency:** Processed dataset metadata.
* **Response Schema (200 OK):**
```json
{
  "total_cities": 26,
  "cities": [
    {
      "city_id": "delhi",
      "name": "Delhi",
      "state": "Delhi",
      "latitude": 28.6139,
      "longitude": 77.2090,
      "data_start": "2015-01-01",
      "data_end": "2020-07-01",
      "total_records": 2009,
      "completeness_score": 0.94
    },
    {
      "city_id": "bengaluru",
      "name": "Bengaluru",
      "state": "Karnataka",
      "latitude": 12.9716,
      "longitude": 77.5946,
      "data_start": "2015-01-01",
      "data_end": "2020-07-01",
      "total_records": 2009,
      "completeness_score": 0.88
    }
  ]
}
```

---

### 3. Current / Latest Recorded Air Quality
* **Endpoint:** `GET /api/v1/current?city={city_id}`
* **Purpose:** Retrieves the most recent recorded multi-pollutant observation and AQI for a selected city from the historical repository (or simulated latest buffer).
* **Requires Trained Model:** No.
* **Data Source Dependency:** Partitioned historical dataset.
* **Response Schema (200 OK):**
```json
{
  "city": "Delhi",
  "date": "2020-07-01",
  "is_simulated_live": false,
  "aqi": 182.0,
  "aqi_bucket": "Moderate",
  "pollutants": {
    "pm25": 54.01,
    "pm10": 108.32,
    "no2": 26.21,
    "so2": 11.45,
    "co": 0.85,
    "o3": 38.12
  }
}
```

---

### 4. Custom Sequence Predictor
* **Endpoint:** `POST /api/v1/predict`
* **Purpose:** Accepts an explicit 30-day multivariate sequence and returns single-day-ahead predicted AQI, NAQI bucket, and confidence bounds.
* **Requires Trained Model:** Yes.
* **Data Source Dependency:** User-provided payload.
* **Request Schema:**
```json
{
  "model_id": "cnn_bilstm_fuzzy_attn",
  "city": "Delhi",
  "sequence": [
    {
      "date": "2020-06-02",
      "pm25": 48.2,
      "pm10": 98.1,
      "no2": 22.4,
      "so2": 9.8,
      "co": 0.72,
      "o3": 34.1
    }
  ]
}
```
*Note: `sequence` array must contain exactly 30 daily elements.*
* **Response Schema (200 OK):**
```json
{
  "model_id": "cnn_bilstm_fuzzy_attn",
  "city": "Delhi",
  "forecast_date": "2020-07-02",
  "predicted_aqi": 189.4,
  "aqi_bucket": "Moderate",
  "color_hex": "#F59E0B",
  "confidence_interval": {
    "lower_95": 172.1,
    "upper_95": 206.8
  },
  "inference_latency_ms": 14.8
}
```

---

### 5. City Forecast by Date Horizon
* **Endpoint:** `GET /api/v1/forecast?city={city_id}&date={date}&model_id={model_id}`
* **Purpose:** Automatically pulls the 30-day lookback sequence ending on `{date}` for `{city}` and returns the $t+1$ forecast.
* **Requires Trained Model:** Yes.
* **Data Source Dependency:** Processed dataset + Model registry.
* **Response Schema (200 OK):**
```json
{
  "city": "Delhi",
  "reference_date": "2020-06-30",
  "forecast_date": "2020-07-01",
  "model_id": "cnn_bilstm_fuzzy_attn",
  "actual_aqi": 182.0,
  "predicted_aqi": 178.6,
  "error": -3.4,
  "aqi_bucket": "Moderate",
  "category_match": true
}
```

---

### 6. Model Registry & Metadata
* **Endpoint:** `GET /api/v1/models`
* **Purpose:** Lists all 8 supported model families, their architecture types, parameter counts, and active status.
* **Requires Trained Model:** Yes.
* **Response Schema (200 OK):**
```json
{
  "models": [
    {
      "model_id": "xgboost_baseline",
      "display_name": "XGBoost Tabular Baseline",
      "family": "Traditional ML",
      "is_flagship": false,
      "trainable_parameters": 0
    },
    {
      "model_id": "cnn_bilstm_fuzzy_attn",
      "display_name": "CNN-BiLSTM with Fuzzy Attention",
      "family": "Hybrid Deep Learning",
      "is_flagship": true,
      "trainable_parameters": 148560
    }
  ]
}
```

---

### 7. Evaluation Metrics & Benchmarks
* **Endpoint:** `GET /api/v1/metrics?model_id={optional_model_id}`
* **Purpose:** Returns comprehensive test-set benchmark metrics across regression (RMSE, MAE, $R^2$) and categorical classification (Accuracy, Macro F1, Severe Bucket Recall).
* **Requires Trained Model:** Yes (reads evaluated benchmark artifacts).
* **Response Schema (200 OK):**
```json
{
  "benchmark_summary": [
    {
      "model_id": "cnn_bilstm_fuzzy_attn",
      "rmse": 31.42,
      "mae": 19.85,
      "r2_score": 0.884,
      "mape_percent": 14.2,
      "naqi_accuracy": 0.826,
      "macro_f1": 0.804,
      "severe_recall": 0.891
    },
    {
      "model_id": "xgboost_baseline",
      "rmse": 48.15,
      "mae": 31.20,
      "r2_score": 0.742,
      "mape_percent": 22.8,
      "naqi_accuracy": 0.710,
      "macro_f1": 0.672,
      "severe_recall": 0.695
    }
  ]
}
```

---

### 8. Explainability & Attribution
* **Endpoint:** `POST /api/v1/explain`
* **Purpose:** Returns temporal attention distributions over the 30-day sequence and multi-pollutant feature attribution scores.
* **Requires Trained Model:** Yes (Temporal/Fuzzy Attention models).
* **Request Schema:**
```json
{
  "model_id": "cnn_bilstm_fuzzy_attn",
  "city": "Delhi",
  "target_date": "2020-07-01"
}
```
* **Response Schema (200 OK):**
```json
{
  "model_id": "cnn_bilstm_fuzzy_attn",
  "target_date": "2020-07-01",
  "predicted_aqi": 189.4,
  "temporal_attention_weights": [
    {"day_offset": -29, "date": "2020-06-02", "raw_attention": 0.012, "fuzzy_modulated_attention": 0.009},
    {"day_offset": -1, "date": "2020-06-30", "raw_attention": 0.085, "fuzzy_modulated_attention": 0.142}
  ],
  "pollutant_attributions": {
    "pm25": 0.42,
    "pm10": 0.28,
    "no2": 0.14,
    "co": 0.07,
    "so2": 0.05,
    "o3": 0.04
  }
}
```

---

### 9. Fuzzy Intelligence Diagnostics
* **Endpoint:** `POST /api/v1/fuzzy/analyze`
* **Purpose:** Inspects pollutant rate-of-change ($\text{ROC}$), Gaussian/Trapezoidal fuzzy set activations, and modulation multipliers for a sequence.
* **Requires Trained Model:** No (executes pure fuzzy logic engine).
* **Request Schema:**
```json
{
  "city": "Delhi",
  "lookback_days": 30
}
```
* **Response Schema (200 OK):**
```json
{
  "analysis_date": "2020-07-01",
  "roc_series": [
    {
      "day": 29,
      "date": "2020-06-30",
      "pm25_roc": 0.41,
      "composite_roc": 0.38,
      "memberships": {
        "stable": 0.02,
        "moderate_rise": 0.34,
        "severe_surge": 0.81
      },
      "volatility_multiplier_gamma": 1.72
    }
  ]
}
```

---

### 10. Counterfactual What-If Simulator
* **Endpoint:** `POST /api/v1/simulate`
* **Purpose:** Evaluates how hypothetical policy interventions (e.g., $-25\%$ $\text{PM}_{2.5}$, $-10\%$ $\text{NO}_2$) alter the next-day forecast relative to baseline.
* **Requires Trained Model:** Yes.
* **Request Schema:**
```json
{
  "city": "Delhi",
  "reference_date": "2020-06-30",
  "model_id": "cnn_bilstm_fuzzy_attn",
  "perturbations": {
    "pm25_percent": -30.0,
    "pm10_percent": -20.0,
    "no2_percent": -15.0,
    "so2_percent": 0.0,
    "co_percent": -10.0,
    "o3_percent": 0.0
  }
}
```
* **Response Schema (200 OK):**
```json
{
  "city": "Delhi",
  "reference_date": "2020-06-30",
  "baseline_predicted_aqi": 218.5,
  "baseline_bucket": "Poor",
  "simulated_predicted_aqi": 164.2,
  "simulated_bucket": "Moderate",
  "aqi_delta": -54.3,
  "category_improved": true,
  "disclaimer": "Model-based hypothetical scenario; not an empirical physical dispersion forecast."
}
```

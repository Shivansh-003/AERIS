# AERIS: Project Specification
**Adaptive Environmental Risk & Intelligence System**
*Document Version: 1.0.0 — Phase 0 Specification*
*Date: September 2026*

---

## 1. Project Name & Description

**Project Name:** AERIS  
**Full Name:** Adaptive Environmental Risk & Intelligence System  
**Category:** AI-Powered Environmental Intelligence, Spatio-Temporal Time-Series Forecasting, Hybrid Deep Learning, Fuzzy Logic, and Explainable AI (XAI).

AERIS is an advanced, research-grounded environmental intelligence platform engineered to predict daily Air Quality Index (AQI) values, analyze complex atmospheric pollutant dynamics, and provide transparent, interpretable forecasts for urban regions across India.

By uniting convolutional feature extraction (1D-CNN), bidirectional temporal sequence modeling (BiLSTM), temporal attention, and domain-informed fuzzy logic attention modulators, AERIS models severe non-linear pollution spikes, rate-of-change volatility, and seasonal atmospheric transitions. Beyond predictive modeling, AERIS bridges research and decision support through a high-performance REST API and an interactive web dashboard featuring explainability attribution, fuzzy membership diagnostics, and counterfactual "what-if" pollutant simulations.

---

## 2. Executive Summary

Urban air pollution is among the most pressing public health and environmental challenges in India, characterized by extreme seasonal variations, abrupt meteorological shifts, and complex multi-pollutant interactions (e.g., particulate matter $\text{PM}_{2.5}, \text{PM}_{10}$, nitrogen oxides $\text{NO}_x$, ozone $\text{O}_3$). Standard machine learning models and black-box deep learning architectures often struggle with two fundamental issues:
1. **Inability to dynamically weigh abrupt pollutant surges and rate-of-change shocks** against long-term baseline trends.
2. **Lack of interpretability**, which impedes trust and actionable deployment by researchers and policy analysts.

AERIS solves this by implementing a rigorous, hybrid deep-learning paradigm that combines:
* Local temporal feature extraction via 1D Convolutional Neural Networks.
* Bidirectional temporal context capture via Bidirectional Long Short-Term Memory networks (BiLSTM).
* A novel **Fuzzy Attention Mechanism** that modulates temporal attention scores using pollutant rate-of-change ($\text{ROC}$) dynamics and expert-guided fuzzy membership functions.
* Comprehensive model comparison across 8 model families (XGBoost, LightGBM, LSTM, BiLSTM, Transformer, CNN-BiLSTM, CNN-BiLSTM + Temporal Attention, and CNN-BiLSTM + Fuzzy Attention).
* Explainable AI (XAI) modules delivering temporal attention weights, pollutant feature attributions, and what-if scenario simulations.
* An enterprise-grade software architecture featuring a Python/FastAPI backend and a Next.js/TypeScript analytical dashboard.

---

## 3. Problem Statement

Air pollution dynamics in Indian metropolitan and industrial zones exhibit high non-linearity, sharp seasonality (e.g., winter thermal inversions, post-monsoon crop residue burning), and volatile daily fluctuations. Existing air quality forecasting approaches suffer from several core deficiencies:
* **Shallow Feature Representation:** Standard time-series models (e.g., ARIMA, linear state-space models) fail to capture multi-pollutant cross-correlations and non-linear interactions.
* **Temporal Lag and Spike Attenuation:** Standard Recurrent Neural Networks (RNNs/LSTMs) often smooth out extreme peak pollution events, under-predicting the "Severe" and "Very Poor" hazard categories.
* **Opacity and Black-Box Nature:** Deep neural networks do not inherently provide domain-meaningful explanations of why a particular day's forecast spiked.
* **Sensitivity to Missing and Noisy Sensor Data:** Real-world continuous ambient air quality monitoring stations (CAAQMS) encounter frequent sensor downtime, drift, and transmission drops.

AERIS addresses these issues by delivering a structured, robust, and explainable deep learning pipeline that explicitly models pollutant volatility through fuzzy logic and provides granular insight into predictive confidence and temporal attribution.

---

## 4. Motivation and Real-World Relevance

Accurate single-day-ahead AQI forecasts are critical for:
* **Proactive Public Health Advisory:** Alerting vulnerable populations (children, elderly, asthmatics) before hazardous exposure events occur.
* **Municipal Decision-Making:** Assisting urban local bodies and environmental control boards in triggering staged anti-pollution measures (e.g., traffic restrictions, industrial emission throttling, construction pauses).
* **Research and Environmental Analytics:** Empowering environmental scientists with transparent tools to evaluate how specific pollutant rate-of-change patterns influence composite air quality indices.

---

## 5. Research Objectives

1. **Develop a Multi-Pollutant Hybrid Time-Series Architecture:** Construct a robust CNN-BiLSTM backbone capable of extracting both localized multi-pollutant signatures and long-range bidirectional temporal dependencies from historical multi-station observations.
2. **Design and Evaluate a Novel Fuzzy Temporal Attention Mechanism:** Formulate a mathematical framework where pollutant rates of change ($\text{ROC}$) are mapped to fuzzy linguistic sets (e.g., *Rapidly Rising*, *Stable*, *Declining*) to dynamically modulate temporal attention weights.
3. **Conduct Rigorous Empirical Benchmarking:** Compare the proposed fuzzy-attention hybrid model against 7 established baseline model families under identical chronological evaluation splits.
4. **Deliver Granular Explainability and Scenario Modeling:** Extract and visualize temporal attention distributions, feature-level attribution weights, and counterfactual response curves for simulated pollutant reductions.
5. **Demonstrate End-to-End System Reliability:** Integrate research models into a production-ready, typed REST API and an accessible, high-performance web dashboard.

---

## 6. Research Questions

* **RQ1 (Fuzzy Attention Efficacy):** Does modulating temporal attention with pollutant rate-of-change fuzzy membership functions yield statistically significant reductions in prediction error (RMSE, MAE) and improved categorical accuracy (Macro F1 on National AQI buckets) compared to unmodulated temporal attention and standard BiLSTM baselines?
* **RQ2 (Hybrid Spatio-Temporal Extraction):** To what extent does pre-pending 1D Convolutional layers before a BiLSTM improve the extraction of localized cross-pollutant correlations compared to pure recurrence or self-attention (Transformer) architectures?
* **RQ3 (Peak Pollution Sensitivity):** Does the fuzzy attention mechanism improve forecast accuracy specifically during hazardous peak-pollution episodes ($\text{AQI} > 300$, "Very Poor" / "Severe") where conventional regression models typically suffer from regression-to-the-mean smoothing?
* **RQ4 (Interpretability & Domain Alignment):** Do the temporal attention weights modulated by fuzzy logic correlate with human-expert domain knowledge regarding pollutant accumulation and meteorological persistence?

---

## 7. Proposed Contributions

1. **Unified Multi-Model Research Framework:** A reproducible benchmarking pipeline supporting 8 distinct model families under strictly leak-free chronological splitting.
2. **Mathematical Formulation of Fuzzy Temporal Attention:** An explicit mathematical engine translating first-order pollutant derivatives into fuzzy risk scalars that rescale softmax attention distributions.
3. **Ablation & Comparative Study:** Comprehensive ablation isolating the individual contributions of 1D-CNN feature extraction, BiLSTM recurrence, standard temporal attention, and fuzzy rate-of-change modulation.
4. **Interactive Environmental Intelligence Platform:** A full-stack, modular architecture unifying PyTorch model serving, FastAPI endpoints, and Next.js data visualizations for public and academic utility.

---

## 8. Scope of the Project

The AERIS project encompasses:
* Historical dataset ingestion, validation, and multi-city chronological partitioning using India's national daily air quality dataset (`city_day.csv`).
* Multi-pollutant preprocessing, missing value handling, feature scaling, and rolling volatility extraction.
* Implementation, hyperparameter optimization, and evaluation of 8 machine learning and deep learning model families.
* Sequence generation using a 30-day historical sliding lookback window for single-day-ahead ($t+1$) continuous AQI prediction.
* Model interpretability mechanisms (temporal attention extraction, pollutant attribution, what-if scenario simulator).
* High-performance, schema-validated REST API (FastAPI) and responsive analytics dashboard (Next.js/TypeScript).
* Full containerization and deployment specification using Docker and Docker Compose.

---

## 9. Explicit Out-of-Scope Features

To maintain scientific integrity and realistic engineering boundaries, the following are explicitly out of scope for the current system:
* **No Live Automated Public Health Warnings:** The system is an analytical and forecasting decision-support system. It will not dispatch automated emergency health alerts to citizens without institutional or regulatory human-in-the-loop validation.
* **No Real-Time IoT Hardware Ingestion:** Direct hardware firmware integration, LoRaWAN gateways, or direct CAAQMS sensor telemetry streaming are out of scope; data ingestion operates via structured tabular records and simulated live buffers.
* **No Sub-Hourly / Dispersion Physics Modeling:** Numerical meteorological dispersion modeling (e.g., WRF-Chem, AERMOD, Gaussian plume simulations) and micro-scale 15-minute sensor forecasting are out of scope. The temporal resolution is fixed at daily increments.
* **No Distributed Cluster Orchestration:** High-overhead distributed architectures (e.g., Apache Kafka, Apache Spark, Kubernetes clusters) are excluded in favor of a clean, maintainable single-node containerized deployment.

---

## 10. Intended Users

1. **Environmental & Climate Researchers:** Evaluating hybrid time-series architectures, attention mechanisms, and multi-pollutant interactions.
2. **Urban Planners & Environmental Policy Analysts:** Assessing city-level historical air quality trends, seasonal patterns, and simulated policy intervention impacts via what-if scenarios.
3. **Data Science & ML Engineers:** Leveraging the modular PyTorch and FastAPI codebase for reproducible benchmarking and time-series model deployment.
4. **Informed Citizens & Public Observers:** Exploring historical trends, model forecasts, and pollutant risk breakdowns through an intuitive, accessible web dashboard.

---

## 11. Dataset Description

The initial experimental foundation is derived from the official historical Indian air quality dataset (`city_day.csv`), compiled from Central Pollution Control Board (CPCB) continuous monitoring stations across Indian cities (2015–2020).

* **Temporal Granularity:** Daily observations.
* **Geographical Coverage:** Multiple major Indian metropolitan and industrial cities (e.g., Delhi, Bengaluru, Hyderabad, Chennai, Kolkata, Ahmedabad, Mumbai, Lucknow, Patna, Jaipur, etc.).
* **Observation Structure:** City-day aggregated records indexed by `City` and `Date`.

---

## 12. Input Features

The dataset comprises 12 primary ambient air pollutant measurements and composite indices:

| Feature Name | Description | Standard Unit | Role |
| :--- | :--- | :--- | :--- |
| `City` | Name of the urban area / city | Categorical | Grouping / Partitioning Identifier |
| `Date` | Observation date (`YYYY-MM-DD`) | Temporal | Chronological Index |
| `PM2.5` | Fine particulate matter ($\le 2.5\,\mu\text{m}$) | $\mu\text{g}/\text{m}^3$ | Continuous Feature |
| `PM10` | Coarse particulate matter ($\le 10\,\mu\text{m}$) | $\mu\text{g}/\text{m}^3$ | Continuous Feature |
| `NO` | Nitric oxide | $\mu\text{g}/\text{m}^3$ | Continuous Feature |
| `NO2` | Nitrogen dioxide | $\mu\text{g}/\text{m}^3$ | Continuous Feature |
| `NOx` | Total nitrogen oxides | $\text{ppb}$ / $\mu\text{g}/\text{m}^3$ | Continuous Feature |
| `NH3` | Ammonia | $\mu\text{g}/\text{m}^3$ | Continuous Feature |
| `CO` | Carbon monoxide | $\text{mg}/\text{m}^3$ | Continuous Feature |
| `SO2` | Sulfur dioxide | $\mu\text{g}/\text{m}^3$ | Continuous Feature |
| `O3` | Ozone | $\mu\text{g}/\text{m}^3$ | Continuous Feature |
| `Benzene` | Volatile organic compound (Benzene) | $\mu\text{g}/\text{m}^3$ | Continuous Feature |
| `Toluene` | Volatile organic compound (Toluene) | $\mu\text{g}/\text{m}^3$ | Continuous Feature |
| `Xylene` | Volatile organic compound (Xylene) | $\mu\text{g}/\text{m}^3$ | Continuous Feature |
| `AQI` | Composite Air Quality Index | Dimensionless (0–500+) | Historical Feature & Target |
| `AQI_Bucket` | Indian National AQI Category | Categorical | Ground-truth bucket label |

*Engineered Features (Generated during Preprocessing):*
* First-order rate-of-change ($\text{ROC}$): $\Delta X_t = X_t - X_{t-1}$ for major pollutants.
* 7-day and 14-day rolling means and standard deviations.
* Cyclic temporal encodings: Day of year, month of year ($\sin/\cos$ transformations).

---

## 13. Target Variable

* **Primary Target:** Next-day continuous Air Quality Index ($\text{AQI}_{t+1} \in \mathbb{R}^+$).
* **Target Characteristics:** Continuous, non-negative scalar exhibiting strong right-skewness and extreme upper-tail variance during severe episodic events.

---

## 14. Prediction Horizon

* **Horizon:** Single-step ahead ($t+1$ day).
* **Lookback Window:** $T = 30$ consecutive historical days $[t-29, t]$.
* **Sampling Rate:** $1\text{ sample} = 1\text{ calendar day}$.

---

## 15. Sequence-Generation Strategy

* **City-Isolated Sliding Windows:** For each individual city, observations are sorted chronologically. A sliding window of length $L = 30$ days is extracted to predict day $t+1$.
* **Boundary Integrity:** Windows are generated strictly within individual city boundaries; sequences never cross city boundaries.
* **Split Boundary Isolation:** Windows are formed within their respective data splits (Train, Validation, Test) to guarantee that no window spans across the partition boundaries, eliminating temporal lookahead bias.

---

## 16. AQI Category Handling

While the regression models predict a continuous AQI value $\widehat{\text{AQI}}_{t+1}$, predictions are mapped to the standard Indian National Air Quality Index (NAQI) categories for categorical evaluation and visual risk communication:

| NAQI Category | AQI Range | Color Code | Indicative Health Impact |
| :--- | :--- | :--- | :--- |
| **Good** | 0 – 50 | `#10B981` (Green) | Minimal impact |
| **Satisfactory** | 51 – 100 | `#84CC16` (Light Green) | Minor breathing discomfort to sensitive people |
| **Moderate** | 101 – 200 | `#F59E0B` (Amber) | Breathing discomfort to people with lungs, asthma, and heart diseases |
| **Poor** | 201 – 300 | `#F97316` (Orange) | Breathing discomfort to most people on prolonged exposure |
| **Very Poor** | 301 – 400 | `#EF4444` (Red) | Respiratory illness on prolonged exposure |
| **Severe** | 401 – 500+ | `#7F1D1D` (Dark Red / Maroon) | Affects healthy people and seriously impacts those with existing diseases |

---

## 17. Functional Requirements

* **FR-1 (Data Ingestion & Validation):** Ingest CSV/Parquet tabular data, validate schema integrity, enforce correct types, and report missingness distributions.
* **FR-2 (City-Aware Preprocessing):** Execute chronological splitting, per-city interpolation (linear + boundary preservation), feature scaling (fitted strictly on training splits), and rolling feature extraction.
* **FR-3 (Multi-Model Training & Evaluation):** Train, evaluate, and benchmark 8 model families using identical test folds, logging regression (RMSE, MAE, $R^2$, MAPE) and classification metrics (Accuracy, Macro F1).
* **FR-4 (Fuzzy Attention Computation):** Compute pollutant rate of change, evaluate Gaussian and Trapezoidal membership functions, aggregate rule activations, and modulate temporal attention weights.
* **FR-5 (Explainability Extraction):** Provide endpoints and visualizations for temporal attention heatmaps, feature attribution scores, and model uncertainty bounds.
* **FR-6 (What-If Simulation):** Allow users to perturb historical pollutant inputs (e.g., $-20\%$ $\text{PM}_{2.5}$) and receive updated model predictions with delta comparisons.
* **FR-7 (REST API Serving):** Deliver high-performance REST endpoints under `/api/v1` with Pydantic schema validation, error handling, and model caching.
* **FR-8 (Interactive Web Dashboard):** Provide 9 dedicated analytical views built with Next.js, Tailwind CSS, and Recharts.

---

## 18. Non-Functional Requirements

* **NFR-1 (Inference Latency):** Single-window prediction latency $< 200\,\text{ms}$ on standard CPU hardware.
* **NFR-2 (Reproducibility):** Deterministic random seeds across NumPy, Scikit-learn, and PyTorch; fully scripted pipeline from raw data to evaluation tables.
* **NFR-3 (Modularity & Extensibility):** Clean separation of concerns between data pipelines, model definitions, API routes, and UI components.
* **NFR-4 (Robustness & Graceful Degradation):** Informative HTTP error responses for missing features, invalid dates, or out-of-distribution values.
* **NFR-5 (UI Accessibility & Responsiveness):** High-contrast, WCAG-compliant environmental dark theme with full desktop and tablet responsiveness.

---

## 19. Technical Constraints

* **Platform:** Windows / Linux / macOS compatible.
* **Runtime:** Python 3.10+ (Backend / ML) and Node.js 18+ / 20+ (Frontend).
* **Deep Learning Framework:** PyTorch 2.x.
* **Frontend Framework:** Next.js (App Router), TypeScript, Tailwind CSS.
* **Hardware Footprint:** Optimized to train and execute on commodity workstation hardware (Single NVIDIA GPU or multi-core CPU).

---

## 20. Assumptions and Limitations

* **Station Aggregation:** Station-level sensor measurements are aggregated to city-level daily averages, smoothing micro-climate variance within large cities.
* **Historical Gaps:** Certain cities and time periods have prolonged missing data intervals (particularly for volatile organics like Benzene, Toluene, Xylene), requiring robust imputation or feature masking.
* **Model-Based Simulation Caveat:** Counterfactual what-if simulations reflect statistical model associations and do not constitute physical fluid dispersion simulations.
* **Non-Stationarity:** Extreme external events (e.g., COVID-19 lockdowns in 2020) present structural regime shifts that must be accounted for during temporal validation.

---

## 21. Expected Deliverables

1. Complete Phase 0 Specification & Architecture suite (7 markdown documents).
2. Clean, modular Python package for preprocessing, sequence generation, modeling, and evaluation.
3. Model artifact registry containing trained weights, scalers, and metric summaries.
4. FastAPI backend service with fully documented OpenAPI `/api/v1` routes.
5. Next.js environmental intelligence web dashboard.
6. Docker and Docker Compose configuration for one-command containerized execution.
7. Comprehensive research report detailing experimental findings, ablation studies, and attention visualizations.

---

## 22. Success Criteria

* **Methodological Rigor:** Zero temporal leakage across train/val/test splits; 100% reproducible baseline and hybrid models.
* **Predictive Performance:** The proposed CNN-BiLSTM + Fuzzy Attention architecture demonstrates competitive or superior RMSE, MAE, and Macro F1 scores compared to classical ML baselines (XGBoost, LightGBM) and standard LSTM/BiLSTM baselines.
* **Explainability Utility:** Clear, interpretable temporal attention distributions that highlight critical pollutant accumulation periods prior to AQI spike events.
* **Production Readiness:** Functional API and dashboard enabling real-time exploratory analysis, model comparison, and what-if simulation with zero runtime crashes.

# AERIS: Dashboard & User Interface Design Specification
**Adaptive Environmental Risk & Intelligence System**
*Document Version: 1.0.0 — Phase 0 UI/UX Specification*
*Date: September 2026*

---

## 1. Visual Direction & Design System

AERIS is designed as a **mission-critical environmental intelligence platform** for researchers, environmental analysts, and municipal decision-makers. It emphasizes high information density, visual clarity, data integrity, and WCAG-compliant contrast over decorative animations or frivolous styling.

### 1.1 Color Palette
* **Background Primary:** Dark Charcoal / Deep Navy (`#0B1120`)
* **Card & Surface Background:** Slate Navy (`#1E293B`)
* **Surface Border:** Subtle Slate Border (`#334155`)
* **Text Primary:** Pure Crisp White (`#F8FAFC`)
* **Text Secondary:** Slate Gray (`#94A3B8`)
* **Brand / Accent:** Electric Cyan (`#06B6D4`)

### 1.2 Indian National AQI (NAQI) Standard Palette
All AQI visualizations strictly adhere to the standardized Indian NAQI color tiers:
* **Good (0 – 50):** Emerald Green (`#10B981`)
* **Satisfactory (51 – 100):** Lime Green (`#84CC16`)
* **Moderate (101 – 200):** Amber / Warm Yellow (`#F59E0B`)
* **Poor (201 – 300):** Deep Orange (`#F97316`)
* **Very Poor (301 – 400):** Crimson Red (`#EF4444`)
* **Severe (401 – 500+):** Maroon / Dark Burgundy (`#7F1D1D`)

### 1.3 Typography & Components
* **Font Family:** Inter (`sans-serif`) for numerical readability and UI hierarchy; JetBrains Mono for metrics and tensor shapes.
* **Component Library:** Headless `shadcn/ui` primitives built on Tailwind CSS.
* **Chart Library:** `Recharts` for high-performance SVG time-series, bar attributions, and scatter plots.
* **Geospatial Engine:** `MapLibre GL JS` / `Leaflet` with a customized dark cartographic basemap.

---

## 2. Global Shell & Navigation Architecture

The global shell persists across all views and provides synchronized context switching:

```
+-----------------------------------------------------------------------------------------------+
| AERIS LOGO  |  [City Selector: Delhi v]  [Date: 2020-06-30]  [Model: CNN-BiLSTM-Fuzzy v] | Status |
+-------------+---------------------------------------------------------------------------------+
| [Sidebar]   |                                                                                 |
| 1. Overview |                                                                                 |
| 2. Map      |                                                                                 |
| 3. Forecast |                                MAIN VIEW AREA                                   |
| 4. ModelLab |                     (Routed dynamically via Next.js App Router)                 |
| 5. Explain  |                                                                                 |
| 6. Fuzzy    |                                                                                 |
| 7. Simulate |                                                                                 |
| 8. Data     |                                                                                 |
| 9. System   |                                                                                 |
+-------------+---------------------------------------------------------------------------------+
```

### Global Controls
1. **City Selector (Dropdown):** Switches active urban center across all views.
2. **Reference Date Picker:** Selects the anchor historical day $t$ (which looks back 30 days and predicts $t+1$).
3. **Model Selector:** Toggles active inference model (defaults to Flagship: `CNN-BiLSTM + Fuzzy Attention`).
4. **Telemetry Badge:** Live indicator of API connectivity and latency.

---

## 3. Detailed Page Specifications

---

### Page 1: Overview
* **Route:** `/overview`
* **Purpose:** Executive summary of regional air quality, next-day forecast, and multi-pollutant levels.
* **Layout & Grid:** 4-column metric card grid on top $\to$ 2-column split (Forecast hero + Pollutant breakdown) $\to$ Full-width 30-day historical trend chart.
* **Core Components:**
  * `AQIHeroCard`: Displays predicted AQI for $t+1$ with prominent NAQI badge, risk classification, and confidence interval.
  * `PollutantGrid`: 6 micro-cards showing current levels of $\text{PM}_{2.5}, \text{PM}_{10}, \text{NO}_2, \text{SO}_2, \text{CO}, \text{O}_3$.
  * `HistoricalTrendChart`: Area chart of past 30 days AQI overlaid with the $t+1$ prediction point.
* **Required API Data:** `GET /api/v1/current`, `GET /api/v1/forecast`.
* **State Handling:**
  * *Loading:* Skeleton cards with subtle pulse animations.
  * *Empty:* Banner indicating no data for the selected date range.
  * *Error:* Red alert banner with retry button.

---

### Page 2: Air Quality Map
* **Route:** `/map`
* **Purpose:** Spatial overview of air quality and hazard tiers across all monitored Indian cities.
* **Layout:** Full-viewport interactive map with a floating left-hand city drawer and a bottom timeline playback scrubber.
* **Core Components:**
  * `IndiaAirQualityMap`: Vector basemap rendering colored pulse markers at city coordinates sized by AQI severity.
  * `CityDetailDrawer`: Slide-over panel presenting the selected city's 30-day mini sparkline and top pollutant driver.
  * `TimelineScrubber`: Slider enabling historical date scrubbing across the dataset timeline.
* **Required API Data:** `GET /api/v1/cities`, `GET /api/v1/forecast`.
* **Interactions:** Clicking any city marker updates the global city context and opens the detail drawer.

---

### Page 3: Forecast
* **Route:** `/forecast`
* **Purpose:** Granular inspection of the single-day-ahead forecast with multi-model comparisons and historical ground truth.
* **Layout:** Top parameter control bar $\to$ Main Forecast Comparison Line Chart $\to$ Residual Error Distribution Table.
* **Core Components:**
  * `MultiModelForecastChart`: Line chart plotting ground-truth AQI alongside predictions from selected models (e.g., Flagship vs XGBoost vs LSTM).
  * `PredictionIntervalBand`: Shaded 95% confidence interval ribbon surrounding the flagship forecast.
  * `ForecastSummaryTable`: Tabular comparison of predicted value, absolute error, and NAQI category concordance.
* **Required API Data:** `GET /api/v1/forecast`, `GET /api/v1/models`.

---

### Page 4: Model Lab & Benchmark Studio
* **Route:** `/model-lab`
* **Purpose:** Research studio for comparing all 8 model families, evaluating test-set metrics, and inspecting loss curves.
* **Layout:** Metric summary cards $\to$ Model Comparison Bar Charts (RMSE, MAE, Macro F1) $\to$ Actual vs Predicted Scatter Plot $\to$ Full Benchmark Table.
* **Core Components:**
  * `BenchmarkMetricCards`: Best-in-class RMSE, MAE, and F1 indicators.
  * `ActualVsPredictedScatter`: 45-degree parity scatter plot color-coded by NAQI bucket.
  * `AblationMatrixTable`: Interactive table highlighting the incremental metric gains of Conv1D, BiLSTM, Attention, and Fuzzy ROC.
* **Required API Data:** `GET /api/v1/metrics`.

---

### Page 5: Explainability & Attribution
* **Route:** `/explainability`
* **Purpose:** Transparent inspection of neural attention weights and multi-pollutant feature attributions.
* **Layout:** Top summary $\to$ 30-Day Temporal Attention Heatmap $\to$ Feature Importance Bar Chart $\to$ Methodological notes.
* **Core Components:**
  * `TemporalAttentionBarChart`: Bar chart showing normalized attention weight $\tilde{\alpha}_t$ for each of the 30 preceding days. High-attention days are highlighted with event annotations.
  * `PollutantAttributionChart`: Horizontal bar chart displaying the relative contribution of each pollutant to the forecast.
  * `InterpretabilityDisclaimer`: Transparent note outlining attribution assumptions.
* **Required API Data:** `POST /api/v1/explain`.

---

### Page 6: Fuzzy Intelligence Diagnostics
* **Route:** `/fuzzy`
* **Purpose:** Deep-dive into the mathematical fuzzy logic engine and rate-of-change volatility dynamics.
* **Layout:** 3-column layout: Fuzzy Membership Curves $\to$ Rate of Change ($\text{ROC}$) Timeline $\to$ Volatility Multiplier ($\gamma_t$) Distribution.
* **Core Components:**
  * `MembershipCurvePlot`: Interactive SVG plotting Gaussian and Trapezoidal curves for *Stable*, *Moderate Rise*, and *Severe Surge*.
  * `ROCTrendChart`: Dual-axis line chart comparing raw pollutant concentrations against calculated $\Delta X_t$.
  * `AttentionModulationVisualizer`: Side-by-side comparison showing raw temporal attention $e_t$ vs fuzzy-modulated attention $\tilde{e}_t$.
* **Required API Data:** `POST /api/v1/fuzzy/analyze`.

---

### Page 7: What-If Scenario Simulator
* **Route:** `/simulator`
* **Purpose:** Interactive sandbox allowing researchers and planners to test simulated emission reduction scenarios.
* **Layout:** Left panel (Interactive Pollutant Sliders: $-50\%$ to $+50\%$) $\to$ Right panel (Real-time Baseline vs Simulated Forecast Comparison Card & Impact Waterfall).
* **Core Components:**
  * `PerturbationSliderPanel`: Granular sliders for $\text{PM}_{2.5}, \text{PM}_{10}, \text{NO}_2, \text{SO}_2, \text{CO}, \text{O}_3$ with preset buttons (e.g., "Odd-Even Traffic Policy: -20% PM / -15% NO2", "Industrial Weekend Curfew").
  * `SimulationDeltaCard`: Visual comparison of baseline AQI vs scenario AQI with category change indicators.
  * `SimulationCaveatBanner`: Explicit warning clarifying that results represent statistical model sensitivity rather than numerical dispersion physics.
* **Required API Data:** `POST /api/v1/simulate`.

---

### Page 8: Data Explorer & Quality Profiler
* **Route:** `/data-explorer`
* **Purpose:** Historical data inspection, missingness visualization, and distribution analysis.
* **Layout:** Dataset summary banner $\to$ City Missingness Heatmap $\to$ Pollutant Distribution Histograms $\to$ Raw Data Table with CSV export.
* **Core Components:**
  * `MissingnessMatrix`: Heatmap grid illustrating sensor availability across cities and time.
  * `PollutantDistributionPlot`: Box-plots and histograms displaying pollutant skewness and outliers.
  * `DataTable`: Paginated, sortable view of historical observations.
* **Required API Data:** `GET /api/v1/cities`, processed Parquet summary metadata.

---

### Page 9: System, Health & Metadata
* **Route:** `/system`
* **Purpose:** Operational transparency, runtime diagnostics, model versioning, and environment telemetry.
* **Layout:** Service Health Grid $\to$ Loaded Model Registry Table $\to$ Hardware Telemetry & Inference Latency Monitor.
* **Core Components:**
  * `HealthStatusCard`: API uptime, memory usage, CUDA/CPU execution target.
  * `ModelRegistryTable`: Active checkpoint filenames, parameter counts, and creation timestamps.
  * `LatencyChart`: Rolling p50/p95/p99 inference latency monitor.
* **Required API Data:** `GET /api/v1/health`, `GET /api/v1/models`.

---

## 4. Responsive Behavior & UI States

* **Breakpoints:**
  * Desktop ($> 1280\text{px}$): Full multi-column analytical layout.
  * Tablet ($768\text{px} - 1279\text{px}$): Collapsible sidebar; stacked 2-column grids.
  * Mobile ($< 768\text{px}$): Single-column scrollable cards with bottom navigation drawer.
* **Loading Experience:** Content-matched skeleton loaders preventing layout shifts (CLS).
* **Error Resilience:** Individual card-level error boundaries allowing healthy dashboard widgets to render even if a specific endpoint encounters a timeout.

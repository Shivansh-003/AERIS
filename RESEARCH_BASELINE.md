# AERIS: Research Baseline & Methodological Review
**Adaptive Environmental Risk & Intelligence System**
*Document Version: 1.0.0 — Phase 0 Research Foundation*
*Date: September 2026*

---

## 1. Original Problem Formulation & Historical Context

The original research effort focused on predicting urban Air Quality Index (AQI) across Indian cities using deep learning and hybrid architectures. In previous preliminary exploration, hybrid models combining spatial convolution and recurrent bidirectional layers were identified as promising approaches for multi-pollutant time-series. Furthermore, incorporating a fuzzy logic modulation layer over temporal attention was conceptualized to address sharp rate-of-change ($\text{ROC}$) dynamics during seasonal pollution episodes (such as post-monsoon smog events in Northern India).

Because the original source code and intermediate artifacts are no longer available in the workspace, this document serves as the **formal research baseline specification**. It captures the foundational hypotheses, documents known methodological challenges, establishes strict anti-leakage protocols, and outlines the empirical validation criteria required to verify all research claims.

---

## 2. Dataset Description

* **Source Data:** Historical daily air quality dataset (`city_day.csv`), compiled from Central Pollution Control Board (CPCB) continuous ambient monitoring stations across major Indian cities (2015–2020).
* **Granularity:** Daily aggregated readings.
* **Geographical Scope:** Multi-city coverage across diverse climatic zones (e.g., Delhi, Ahmedabad, Bengaluru, Chennai, Hyderabad, Kolkata, Lucknow, Mumbai, Patna, Visakhapatnam).
* **Observed Features:**
  * Particulate Matter: $\text{PM}_{2.5}, \text{PM}_{10}$
  * Nitrogen Compounds: $\text{NO}, \text{NO}_2, \text{NO}_x, \text{NH}_3$
  * Carbon & Sulfur: $\text{CO}, \text{SO}_2$
  * Photochemical Oxidants & Volatiles: $\text{O}_3, \text{Benzene}, \text{Toluene}, \text{Xylene}$
  * Target & Derived Labels: $\text{AQI}, \text{AQI\_Bucket}$

---

## 3. Original Prediction Task & Horizon

* **Task Formulation:** Single-step ahead univariate AQI regression from multivariate historical pollutant observations.
* **Input Representation:** A 30-consecutive-day sliding window of multivariate pollutant features $\mathbf{X}_t = [\mathbf{x}_{t-29}, \mathbf{x}_{t-28}, \dots, \mathbf{x}_t] \in \mathbb{R}^{30 \times D}$.
* **Target Output:** Continuous scalar AQI on the subsequent day $\widehat{Y}_{t+1} = \widehat{\text{AQI}}_{t+1} \in \mathbb{R}^+$.
* **Horizon:** Strictly 1 calendar day ($t+1$). It is **not** an hourly or multi-step forecast in this baseline configuration.

---

## 4. Chronological Splitting Strategy

To ensure zero future-to-past data leakage:
* **Partition Ratios:** 70% Training, 15% Validation, 15% Testing.
* **City-Level Isolation:** The split is executed chronologically within each individual city's timeline:
  * $\text{Train: } [T_0, T_{70\%}]$
  * $\text{Validation: } [T_{70\%} + 1, T_{85\%}]$
  * $\text{Test: } [T_{85\%} + 1, T_{100\%}]$
* **Window Generation Across Boundaries:** Windows are generated strictly within each partition. No 30-day window is permitted to cross partition boundaries (e.g., a window cannot start in Train and end in Validation).

---

## 5. Original Preprocessing Methodology & Limitations

In earlier experimental iterations, preprocessing typically involved:
* Standard linear interpolation across missing values.
* Global normalization across the entire dataset prior to splitting.

> [!WARNING] Methodological Pitfall: Global Scaling Leakage
> Global normalization fits scalers across future (test) data, introducing subtle optimistic bias. In the AERIS research design, scalers must be **fitted strictly on the Train partition** and applied without refitting to the Validation and Test partitions.

---

## 6. Model Families in the Research Baseline

The baseline suite evaluates 8 distinct model families categorized into three paradigms:

### A. Traditional Machine Learning Baselines
1. **XGBoost Regressor:** Extreme Gradient Boosting trained on flattened 30-day windows and engineered lag statistics.
2. **LightGBM Regressor:** Light Gradient Boosting Machine optimized for speed and tabular accuracy.

### B. Standard Sequential Deep Learning Baselines
3. **LSTM (Long Short-Term Memory):** Standard unidirectional recurrent network.
4. **BiLSTM (Bidirectional LSTM):** Two-way recurrent processing to capture both forward trajectory and backward context within the 30-day window.
5. **Transformer Encoder:** Multi-head self-attention architecture with sinusoidal temporal positional encodings.

### C. Hybrid & Attention Architectures
6. **CNN-BiLSTM:** 1D Convolutional layers extracting local cross-pollutant correlations, followed by BiLSTM recurrent sequence encoding.
7. **CNN-BiLSTM with Temporal Attention:** Incorporates a Bahdanau-style attention mechanism over BiLSTM hidden states to dynamically focus on critical historical days.
8. **CNN-BiLSTM with Fuzzy Attention (Flagship Research Model):** Modulates temporal attention weights using pollutant rate-of-change ($\text{ROC}$) dynamics passed through domain-specific fuzzy membership functions.

---

## 7. Mathematical Formulation of Fuzzy Temporal Attention

The core research hypothesis asserts that sharp day-to-day rate-of-change in critical pollutants ($\text{PM}_{2.5}, \text{PM}_{10}, \text{NO}_2$) signals sudden atmospheric stability changes (e.g., stagnation, inversion, wildfire/burning plumes) that standard attention mechanisms may under-weight if absolute concentrations are not yet at peak levels.

### Step 1: Pollutant Rate of Change (ROC)
For each time step $t \in [1, 30]$ and key pollutant $p \in \{\text{PM}_{2.5}, \text{PM}_{10}, \text{NO}_2\}$:
$$\Delta X_{t, p} = \frac{X_{t, p} - X_{t-1, p}}{X_{t-1, p} + \epsilon}$$

### Step 2: Fuzzy Membership Evaluation
The rate of change $\Delta X_{t, p}$ is evaluated against three fuzzy linguistic sets:
1. **Stable / Slow Change ($\mu_{\text{stable}}$):** Gaussian membership centered at 0:
   $$\mu_{\text{stable}}(\Delta X) = \exp\left( -\frac{(\Delta X)^2}{2\sigma_{\text{stable}}^2} \right)$$
2. **Moderate Rise ($\mu_{\text{mod}}$):** Trapezoidal membership over $[0.05, 0.15, 0.30, 0.45]$.
3. **Surge / Rapid Spike ($\mu_{\text{surge}}$):** Sigmoidal / Upper Trapezoidal membership activating for $\Delta X > 0.30$.

### Step 3: Fuzzy Rule Aggregation
The individual membership activations are aggregated via fuzzy max-product composition to produce a temporal volatility modulation factor $\gamma_t \in [0.5, 2.0]$:
$$\gamma_t = 1.0 + w_1 \cdot \mu_{\text{mod}}(\Delta X_t) + w_2 \cdot \mu_{\text{surge}}(\Delta X_t) - w_3 \cdot \mu_{\text{stable}}(\Delta X_t)$$

### Step 4: Modulated Temporal Attention
Given standard BiLSTM hidden states $\mathbf{H} = [\mathbf{h}_1, \dots, \mathbf{h}_{30}] \in \mathbb{R}^{30 \times 2d_h}$:
1. Unmodulated energy score:
   $$e_t = \mathbf{v}^T \tanh(\mathbf{W}_h \mathbf{h}_t + \mathbf{b}_h)$$
2. Fuzzy-modulated energy score:
   $$\tilde{e}_t = e_t \cdot \gamma_t$$
3. Normalized attention weights:
   $$\tilde{\alpha}_t = \frac{\exp(\tilde{e}_t)}{\sum_{k=1}^{30} \exp(\tilde{e}_k)}$$
4. Dynamic Context Vector:
   $$\mathbf{c} = \sum_{t=1}^{30} \tilde{\alpha}_t \mathbf{h}_t$$

---

## 8. Original Evaluation Metrics & Categorical Mapping

To ensure comprehensive evaluation across both continuous regression and discrete public health hazard tiers:

### Continuous Regression Metrics
* **Root Mean Squared Error (RMSE):** Measures penalty on large forecast errors.
  $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (Y_i - \widehat{Y}_i)^2}$$
* **Mean Absolute Error (MAE):** Robust measure of average prediction deviation.
  $$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |Y_i - \widehat{Y}_i|$$
* **Coefficient of Determination ($R^2$):** Measures variance explained by the model.
* **Mean Absolute Percentage Error (MAPE):** Relative percentage error.

### Discrete Categorical Metrics
Continuous predictions are mapped into Indian National Air Quality Index (NAQI) bins:
* Good (0–50), Satisfactory (51–100), Moderate (101–200), Poor (201–300), Very Poor (301–400), Severe (401–500+).
* **Metrics:** Overall Accuracy, Macro Precision, Macro Recall, Macro F1-Score, and specifically **Severe Bucket Recall** to measure sensitivity to hazardous smog episodes.

---

## 9. Historical Reported Results & Scientific Baseline

In historical preliminary studies on Indian city air quality subsets, hybrid architectures have generally reported the following performance trends (serving as reference targets to be validated):

| Model Family | Historical Reference RMSE | Historical Reference MAE | Notes / Reference Status |
| :--- | :--- | :--- | :--- |
| **XGBoost Baseline** | $\approx 45.0 - 55.0$ | $\approx 28.0 - 35.0$ | Historical tabular reference |
| **LSTM Baseline** | $\approx 42.0 - 48.0$ | $\approx 26.0 - 32.0$ | Historical sequential reference |
| **BiLSTM Baseline** | $\approx 38.0 - 44.0$ | $\approx 24.0 - 29.0$ | Historical bidirectional reference |
| **CNN-BiLSTM Hybrid** | $\approx 34.0 - 40.0$ | $\approx 21.0 - 26.0$ | Joint spatio-temporal baseline |
| **CNN-BiLSTM + Temp Attn** | $\approx 31.0 - 36.0$ | $\approx 19.0 - 23.0$ | Attention baseline |
| **CNN-BiLSTM + Fuzzy Attn** | $\approx 28.0 - 33.0$ | $\approx 17.0 - 21.0$ | Proposed research target |

> [!IMPORTANT] Scientific Disclaimer
> The figures above are historical reference ranges from past exploratory work. They are **not** claimed as completed benchmarks in this codebase. Full empirical results will be generated de novo under strict leak-free protocols during Phase 11.

---

## 10. Methodological Limitations & Proposed Improvements

| Component | Original Approach | Proposed Improvement | Verification Required |
| :--- | :--- | :--- | :--- |
| **Data Partitioning** | Random or unisolated city slicing | Strict chronological split (70/15/15) strictly per city; zero window crossing | Verify no sample in test set precedes train timestamp or shares window across boundary |
| **Feature Scaling** | Global scaling across all data | Scalers fitted strictly on Train split; serialized and applied to Val/Test | Assert `scaler.fit` is called solely on $X_{\text{train}}$ |
| **Missing Imputation** | Naive global mean/zero replacement | City-aware local linear interpolation ($\le 5\text{ days}$) + forward/backward padding | Audit reconstruction error on artificially masked segments |
| **Imbalance on Peaks** | Standard MSE loss smoothing out Severe peaks | Huber Loss / Smooth L1 Loss with optional upper-tail category penalty | Compare Severe bucket recall ($AQI > 400$) across loss variants |
| **Fuzzy Membership** | Fixed arbitrary manual thresholds | Parameterized Gaussian & Trapezoidal sets calibrated against empirical ROC percentiles | Validate membership curves on empirical training distribution |
| **Explainability** | Black-box output with no attention inspection | Temporal attention weight extraction + SHAP / Integrated Gradients feature attributions | Inspect attention heatmaps during recorded high-pollution events |
| **Uncertainty** | Single point estimate | Monte Carlo Dropout / Ensemble intervals for forecast confidence bounds | Evaluate prediction interval coverage probability (PICP) |

---

## 11. Hypotheses Requiring Empirical Validation

1. **Hypothesis 1 (Fuzzy ROC Benefit):** Modulating attention with rate-of-change fuzzy scores will decrease RMSE by at least 5–8% relative to unmodulated temporal attention during episodic winter pollution surges.
2. **Hypothesis 2 (Convolutional Advantage):** 1D convolutions over multi-pollutant feature vectors will improve feature extraction of co-emitted pollutants ($\text{NO}_x$ and $\text{CO}$; $\text{PM}_{2.5}$ and $\text{PM}_{10}$) relative to pure recurrent models.
3. **Hypothesis 3 (Categorical Reliability):** The proposed hybrid architecture will increase Macro F1 score on extreme NAQI categories ("Very Poor" and "Severe") compared to gradient boosting baselines.

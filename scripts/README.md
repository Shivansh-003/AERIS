# AERIS Automation & Utility Scripts

This directory contains standalone execution and development utility scripts:

* `ingest_city_day.py`: Ingests, validates, profiles, and converts raw historical air quality data (`city_day.csv`) into structured Parquet and metadata profile JSON (Data Acquisition & Validation).
* `run_eda.py`: Computes statistical distributions, missingness topology, IQR outliers, correlations, and generates publication-grade figures in `outputs/eda/figures/` and `outputs/eda/eda_report.json`.
* `run_preprocessing.py`: Executes leak-free chronological partitioning, city-aware forward-fill, training median imputation, and StandardScaler fitting strictly on the training partition (Data Preparation).
* `build_notebook.py`: Generates the structured `notebooks/01_exploratory_data_analysis.ipynb` Jupyter notebook.
* Data validation and inspection utilities.
* Model export and quantization utilities.

## Script Usage

### Ingestion & Validation Pipeline
```bash
python scripts/ingest_city_day.py [--raw-path PATH] [--output-parquet PATH] [--metadata-path PATH]
```

### Exploratory Data Analysis Pipeline
```bash
python scripts/run_eda.py [--parquet-path PATH] [--report-output PATH] [--figures-output PATH]
```

### Preprocessing & Splitting Pipeline
```bash
python scripts/run_preprocessing.py [--input-path PATH] [--output-dir PATH] [--artifacts-dir PATH]
```

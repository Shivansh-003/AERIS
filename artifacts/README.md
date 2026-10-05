# AERIS Model & Preprocessing Artifacts

This directory stores serialized models, scalers, and evaluation metrics produced across modeling and preprocessing workflows.

## Subdirectories (Generated During Training)

* `scalers/`: Fitted `RobustScaler` / `MinMaxScaler` objects (`.joblib`).
* `checkpoints/`: Model state dicts and pipeline weights (`.pt`, `.joblib`).
* `metrics/`: Benchmark evaluation summaries and ablation outputs (`.json`).

# AERIS Data Directory

This directory stores historical and processed air quality datasets.

## Directory Structure

* `raw/`: Place the primary historical dataset here (`city_day.csv`).
* `processed/`: Intermediate cleaned tables, partitioned Parquet files, and sequence tensors generated during data processing phases.

## Note
Raw and processed dataset records are ignored by version control to prevent repository bloat and maintain reproducibility via ingestion pipelines.

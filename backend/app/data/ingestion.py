"""Data ingestion, validation, profiling, and Parquet export pipeline for AERIS."""

import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from backend.app.data.schema import (
    BUCKET_COLUMN,
    CITY_COLUMN,
    DATE_COLUMN,
    EXPECTED_AQI_BUCKETS,
    NUMERIC_COLUMNS,
    POLLUTANT_COLUMNS,
    REQUIRED_COLUMNS,
    TARGET_COLUMN,
    ValidationReport,
)

logger = logging.getLogger("aeris.data.ingestion")


def load_raw_csv(file_path: Path | str) -> Tuple[pd.DataFrame, int]:
    """Load raw CSV dataset without type conversions.

    Args:
        file_path: Path to the raw CSV file.

    Returns:
        Tuple containing raw DataFrame and file size in bytes.

    Raises:
        FileNotFoundError: If the input file does not exist.
        ValueError: If the file is empty.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Raw dataset file not found at: {path}")

    file_size_bytes = path.stat().st_size
    if file_size_bytes == 0:
        raise ValueError(f"Raw dataset file is empty: {path}")

    # Read all columns as raw strings initially to perform strict validation
    df_raw = pd.read_csv(path, dtype=str, keep_default_na=True)
    return df_raw, file_size_bytes


def validate_and_normalize(
    df_raw: pd.DataFrame,
) -> Tuple[pd.DataFrame, ValidationReport]:
    """Validate schema, detect anomalies/duplicates, and safely normalize types.

    Args:
        df_raw: Raw DataFrame loaded directly from CSV.

    Returns:
        Tuple of normalized DataFrame and ValidationReport.
    """
    report = ValidationReport(
        row_count=len(df_raw),
        column_count=len(df_raw.columns),
    )

    # 1. Column presence check
    raw_columns = list(df_raw.columns)
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in raw_columns]
    unexpected_cols = [col for col in raw_columns if col not in REQUIRED_COLUMNS]

    report.missing_columns = missing_cols
    report.unexpected_columns = unexpected_cols

    if missing_cols:
        report.add_error(f"Missing required columns in dataset: {missing_cols}")
        return df_raw.copy(), report

    # 2. Duplicate detection on raw content
    report.duplicate_rows_count = int(df_raw.duplicated().sum())
    if report.duplicate_rows_count > 0:
        report.add_warning(
            f"Detected {report.duplicate_rows_count} completely duplicated rows."
        )

    report.duplicate_city_date_count = int(
        df_raw.duplicated(subset=[CITY_COLUMN, DATE_COLUMN]).sum()
    )
    if report.duplicate_city_date_count > 0:
        report.add_error(
            f"Detected {report.duplicate_city_date_count} duplicate "
            f"({CITY_COLUMN}, {DATE_COLUMN}) records."
        )

    # 3. Create normalized working copy
    df_clean = pd.DataFrame(index=df_raw.index)

    # 4. City column normalization & validation
    df_clean[CITY_COLUMN] = df_raw[CITY_COLUMN].astype("string").str.strip()
    empty_cities = int(
        df_clean[CITY_COLUMN].isna().sum() | (df_clean[CITY_COLUMN] == "").sum()
    )
    if empty_cities > 0:
        report.add_error(
            f"Found {empty_cities} records with missing or empty City values."
        )

    # 5. Date column normalization & validation
    parsed_dates = pd.to_datetime(
        df_raw[DATE_COLUMN], format="%Y-%m-%d", errors="coerce"
    )
    invalid_dates_mask = df_raw[DATE_COLUMN].notna() & parsed_dates.isna()
    report.invalid_dates_count = int(invalid_dates_mask.sum())

    if report.invalid_dates_count > 0:
        sample_invalid = df_raw.loc[invalid_dates_mask, DATE_COLUMN].head(5).tolist()
        report.add_error(
            f"Found {report.invalid_dates_count} invalid Date values "
            f"(expected YYYY-MM-DD). Samples: {sample_invalid}"
        )

    df_clean[DATE_COLUMN] = parsed_dates

    # 6. Numeric column normalization & validation (Pollutants + AQI)
    for col in NUMERIC_COLUMNS:
        raw_series = df_raw[col]
        numeric_series = pd.to_numeric(raw_series, errors="coerce")

        # Non-null raw values that failed numeric conversion
        invalid_numeric_mask = raw_series.notna() & numeric_series.isna()
        invalid_count = int(invalid_numeric_mask.sum())
        if invalid_count > 0:
            report.invalid_numeric_values[col] = invalid_count
            report.add_error(
                f"Column '{col}' has {invalid_count} unparseable non-numeric values."
            )

        # Check for negative values (physical concentrations & AQI must be >= 0)
        negative_mask = numeric_series < 0
        negative_count = int(negative_mask.sum())
        if negative_count > 0:
            report.negative_values_count[col] = negative_count
            report.add_warning(
                f"Column '{col}' contains {negative_count} negative values."
            )

        df_clean[col] = numeric_series.astype("float64")

    # 7. AQI_Bucket validation & normalization
    df_clean[BUCKET_COLUMN] = df_raw[BUCKET_COLUMN].astype("string").str.strip()
    valid_bucket_mask = df_clean[BUCKET_COLUMN].isna() | df_clean[BUCKET_COLUMN].isin(
        EXPECTED_AQI_BUCKETS
    )
    invalid_buckets_count = int((~valid_bucket_mask).sum())
    report.invalid_buckets_count = invalid_buckets_count

    if invalid_buckets_count > 0:
        unexpected_buckets = (
            df_clean.loc[~valid_bucket_mask, BUCKET_COLUMN].unique().tolist()
        )
        report.add_warning(
            f"Found {invalid_buckets_count} records with unexpected "
            f"AQI_Bucket values: {unexpected_buckets}"
        )

    # 8. Deterministic ordering: Sort by City and Date
    df_clean = df_clean.sort_values(
        by=[CITY_COLUMN, DATE_COLUMN], ascending=[True, True]
    ).reset_index(drop=True)

    return df_clean, report


def generate_dataset_profile(
    df: pd.DataFrame,
    source_file: str,
    report: ValidationReport,
    file_size_bytes: Optional[int] = None,
) -> Dict[str, Any]:
    """Compile comprehensive dataset metadata and statistical profile.

    Args:
        df: Validated and normalized DataFrame.
        source_file: Identifier or path of the raw source file.
        report: Validation report with anomaly diagnostics.
        file_size_bytes: Optional raw file size in bytes.

    Returns:
        Dictionary representation of dataset profile.
    """
    total_rows = len(df)
    total_cols = len(df.columns)

    # Date range metrics
    valid_dates = df[DATE_COLUMN].dropna()
    if not valid_dates.empty:
        min_date = valid_dates.min().strftime("%Y-%m-%d")
        max_date = valid_dates.max().strftime("%Y-%m-%d")
        span_days = int((valid_dates.max() - valid_dates.min()).days) + 1
    else:
        min_date = None
        max_date = None
        span_days = 0

    # City list
    unique_cities = sorted([str(c) for c in df[CITY_COLUMN].dropna().unique().tolist()])

    # Missing value counts and percentages
    missing_counts: Dict[str, int] = {}
    missing_percentages: Dict[str, float] = {}
    for col in df.columns:
        cnt = int(df[col].isna().sum())
        missing_counts[col] = cnt
        missing_percentages[col] = (
            round((cnt / total_rows) * 100.0, 2) if total_rows > 0 else 0.0
        )

    # AQI availability summary
    valid_aqi = int(df[TARGET_COLUMN].notna().sum())
    missing_aqi = int(df[TARGET_COLUMN].isna().sum())
    aqi_avail_pct = (
        round((valid_aqi / total_rows) * 100.0, 2) if total_rows > 0 else 0.0
    )

    aqi_availability = {
        "total_records": total_rows,
        "valid_count": valid_aqi,
        "missing_count": missing_aqi,
        "availability_pct": aqi_avail_pct,
    }

    # Pollutant availability summary
    pollutant_availability: Dict[str, Dict[str, Any]] = {}
    for pol in POLLUTANT_COLUMNS:
        valid_cnt = int(df[pol].notna().sum())
        missing_cnt = int(df[pol].isna().sum())
        pct = round((valid_cnt / total_rows) * 100.0, 2) if total_rows > 0 else 0.0
        pollutant_availability[pol] = {
            "valid_count": valid_cnt,
            "missing_count": missing_cnt,
            "availability_pct": pct,
        }

    # Numeric summaries (mean, std, min, percentiles, max)
    numeric_summary: Dict[str, Dict[str, Optional[float]]] = {}
    for col in NUMERIC_COLUMNS:
        series = df[col].dropna()
        if not series.empty:
            desc = series.describe(percentiles=[0.25, 0.50, 0.75])
            std_val = float(desc["std"]) if not np.isnan(desc["std"]) else 0.0
            numeric_summary[col] = {
                "count": int(desc["count"]),
                "mean": round(float(desc["mean"]), 4),
                "std": round(std_val, 4),
                "min": round(float(desc["min"]), 4),
                "p25": round(float(desc["25%"]), 4),
                "p50": round(float(desc["50%"]), 4),
                "p75": round(float(desc["75%"]), 4),
                "max": round(float(desc["max"]), 4),
            }
        else:
            numeric_summary[col] = {
                "count": 0,
                "mean": None,
                "std": None,
                "min": None,
                "p25": None,
                "p50": None,
                "p75": None,
                "max": None,
            }

    # AQI Bucket distribution
    bucket_counts_series = df[BUCKET_COLUMN].value_counts(dropna=False)
    aqi_bucket_distribution: Dict[str, int] = {}
    for bucket_name, count in bucket_counts_series.items():
        key = str(bucket_name) if pd.notna(bucket_name) else "Missing"
        aqi_bucket_distribution[key] = int(count)

    # City-level profile breakdown
    city_profiles: Dict[str, Dict[str, Any]] = {}
    for city_name, group in df.groupby(CITY_COLUMN):
        city_str = str(city_name)
        c_dates = group[DATE_COLUMN].dropna()
        c_rows = len(group)
        c_valid_aqi = int(group[TARGET_COLUMN].notna().sum())
        c_missing_aqi = int(group[TARGET_COLUMN].isna().sum())
        c_aqi_pct = round((c_valid_aqi / c_rows) * 100.0, 2) if c_rows > 0 else 0.0

        min_d = c_dates.min().strftime("%Y-%m-%d") if not c_dates.empty else None
        max_d = c_dates.max().strftime("%Y-%m-%d") if not c_dates.empty else None

        city_profiles[city_str] = {
            "row_count": c_rows,
            "min_date": min_d,
            "max_date": max_d,
            "valid_aqi_count": c_valid_aqi,
            "missing_aqi_count": c_missing_aqi,
            "aqi_availability_pct": c_aqi_pct,
        }

    # Data types dictionary
    dtypes_dict = {col: str(df[col].dtype) for col in df.columns}

    profile: Dict[str, Any] = {
        "metadata_version": "1.0.0",
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source": {
            "file_name": Path(source_file).name,
            "file_path": str(source_file).replace("\\", "/"),
            "file_size_bytes": file_size_bytes,
        },
        "dataset_shape": {
            "row_count": total_rows,
            "column_count": total_cols,
        },
        "columns": list(df.columns),
        "data_types": dtypes_dict,
        "geography": {
            "city_count": len(unique_cities),
            "cities": unique_cities,
        },
        "temporal_span": {
            "min_date": min_date,
            "max_date": max_date,
            "total_days_span": span_days,
        },
        "integrity_audit": {
            "is_valid": report.is_valid,
            "duplicate_rows": report.duplicate_rows_count,
            "duplicate_city_date_pairs": report.duplicate_city_date_count,
            "invalid_dates": report.invalid_dates_count,
            "invalid_numeric_values": report.invalid_numeric_values,
            "negative_numeric_values": report.negative_values_count,
            "invalid_aqi_buckets": report.invalid_buckets_count,
            "errors": report.errors,
            "warnings": report.warnings,
        },
        "missing_values": {
            "counts": missing_counts,
            "percentages": missing_percentages,
        },
        "target_availability": aqi_availability,
        "pollutant_availability": pollutant_availability,
        "numeric_summary": numeric_summary,
        "aqi_bucket_distribution": aqi_bucket_distribution,
        "city_profiles": city_profiles,
    }

    return profile


def save_dataset_profile(profile: Dict[str, Any], output_path: Path | str) -> Path:
    """Save dataset profile dictionary as structured JSON.

    Args:
        profile: Profile dictionary.
        output_path: Destination JSON file path.

    Returns:
        Resolved Path to the saved JSON file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)
    logger.info(f"Dataset profile saved to {path}")
    return path


def save_processed_parquet(df: pd.DataFrame, output_path: Path | str) -> Path:
    """Save validated DataFrame to Parquet format deterministically.

    Args:
        df: Validated DataFrame.
        output_path: Destination Parquet file path.

    Returns:
        Resolved Path to the saved Parquet file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(
        path,
        engine="pyarrow",
        compression="snappy",
        index=False,
    )
    logger.info(f"Processed dataset saved to {path} ({len(df)} rows)")
    return path


def run_ingestion_pipeline(
    raw_path: Path | str = "data/raw/city_day.csv",
    output_parquet: Path | str = "data/processed/cleaned_city_day.parquet",
    metadata_path: Path | str = "data/metadata/dataset_profile.json",
) -> Dict[str, Any]:
    """Execute end-to-end data ingestion, validation, profiling, and export.

    Args:
        raw_path: Path to raw input CSV.
        output_parquet: Target path for output Parquet file.
        metadata_path: Target path for metadata profile JSON.

    Returns:
        Generated dataset profile dictionary.

    Raises:
        ValueError: If validation errors are encountered.
    """
    logger.info(f"Starting AERIS data ingestion pipeline for {raw_path}")

    # 1. Load raw dataset
    df_raw, file_size_bytes = load_raw_csv(raw_path)
    logger.info(
        f"Loaded raw dataset: {len(df_raw)} rows, "
        f"{len(df_raw.columns)} columns, {file_size_bytes} bytes"
    )

    # 2. Validate and normalize schema
    df_clean, report = validate_and_normalize(df_raw)

    # 3. Generate comprehensive profile
    profile = generate_dataset_profile(
        df=df_clean,
        source_file=str(raw_path),
        report=report,
        file_size_bytes=file_size_bytes,
    )

    # 4. Save metadata profile regardless of validation state (to record audit)
    save_dataset_profile(profile, metadata_path)

    # 5. Check validation outcome
    if not report.is_valid:
        error_msg = (
            f"Data validation failed with {len(report.errors)} errors: {report.errors}"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    # 6. Save processed Parquet file
    save_processed_parquet(df_clean, output_parquet)
    logger.info("AERIS data ingestion pipeline completed successfully.")

    return profile

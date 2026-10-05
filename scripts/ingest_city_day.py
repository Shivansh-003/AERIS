#!/usr/bin/env python3
"""AERIS Data Ingestion & Validation CLI Script (Data Acquisition & Validation).

Ingests raw historical air quality data, executes schema validation and anomaly
auditing, generates metadata dataset profiling, and produces a cleaned Parquet file.
"""

import argparse
import sys
from pathlib import Path

# Ensure root repository directory is in sys.path when script is executed directly
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.core.logging import setup_logging  # noqa: E402
from backend.app.data.ingestion import (  # noqa: E402
    generate_dataset_profile,
    load_raw_csv,
    save_dataset_profile,
    save_processed_parquet,
    validate_and_normalize,
)


def print_banner() -> None:
    """Print the AERIS CLI banner."""
    print("=" * 80)
    print("  AERIS: Adaptive Environmental Risk & Intelligence System")
    print("  Data Acquisition & Schema Validation Pipeline")
    print("=" * 80)


def print_audit_summary(profile: dict) -> None:
    """Print structured audit results to stdout."""
    print("\n--- DATASET INGESTION SUMMARY ---")
    source = profile["source"]
    shape = profile["dataset_shape"]
    geo = profile["geography"]
    temporal = profile["temporal_span"]
    audit = profile["integrity_audit"]
    aqi_avail = profile["target_availability"]

    print(
        f"Source File       : {source['file_path']} "
        f"({source['file_size_bytes']:,} bytes)"
    )
    print(f"Total Rows        : {shape['row_count']:,}")
    print(f"Total Columns     : {shape['column_count']}")
    print(f"Unique Cities     : {geo['city_count']}")
    print(
        f"Date Range        : {temporal['min_date']} to {temporal['max_date']} "
        f"({temporal['total_days_span']} days)"
    )
    print(f"Duplicate Rows    : {audit['duplicate_rows']}")
    print(f"Duplicate (City, Date): {audit['duplicate_city_date_pairs']}")
    print(f"Invalid Dates     : {audit['invalid_dates']}")
    print(f"Validation Status : {'PASSED' if audit['is_valid'] else 'FAILED'}")

    print("\n--- TARGET (AQI) AVAILABILITY ---")
    print(
        f"Valid AQI Records : {aqi_avail['valid_count']:,} / "
        f"{aqi_avail['total_records']:,} ({aqi_avail['availability_pct']}%)"
    )
    print(f"Missing AQI       : {aqi_avail['missing_count']:,}")

    print("\n--- POLLUTANT AVAILABILITY ---")
    print(
        f"{'Pollutant':<12} | {'Valid Rows':<12} | {'Missing Rows':<12} | "
        f"{'Availability %':<15}"
    )
    print("-" * 58)
    for pol, stats in profile["pollutant_availability"].items():
        print(
            f"{pol:<12} | {stats['valid_count']:<12,d} | "
            f"{stats['missing_count']:<12,d} | {stats['availability_pct']:<15.2f}%"
        )

    print("\n--- AQI BUCKET DISTRIBUTION ---")
    for bucket, count in profile["aqi_bucket_distribution"].items():
        pct = (count / shape["row_count"]) * 100.0 if shape["row_count"] > 0 else 0.0
        print(f"  * {bucket:<15}: {count:>6,d} ({pct:>5.2f}%)")

    if audit["warnings"]:
        print("\n--- VALIDATION WARNINGS ---")
        for warning in audit["warnings"]:
            print(f"  [WARN] {warning}")

    if audit["errors"]:
        print("\n--- VALIDATION ERRORS ---")
        for error in audit["errors"]:
            print(f"  [ERROR] {error}")


def main() -> int:
    """CLI entrypoint for data ingestion."""
    parser = argparse.ArgumentParser(
        description="AERIS Air Quality Data Ingestion and Validation CLI"
    )
    parser.add_argument(
        "--raw-path",
        type=str,
        default="data/raw/city_day.csv",
        help="Path to raw CSV dataset (default: data/raw/city_day.csv)",
    )
    parser.add_argument(
        "--output-parquet",
        type=str,
        default="data/processed/cleaned_city_day.parquet",
        help=(
            "Path for cleaned Parquet dataset "
            "(default: data/processed/cleaned_city_day.parquet)"
        ),
    )
    parser.add_argument(
        "--metadata-path",
        type=str,
        default="data/metadata/dataset_profile.json",
        help=(
            "Path for generated metadata JSON "
            "(default: data/metadata/dataset_profile.json)"
        ),
    )

    args = parser.parse_args()
    logger = setup_logging()
    print_banner()

    raw_path = Path(args.raw_path)
    output_parquet = Path(args.output_parquet)
    metadata_path = Path(args.metadata_path)

    try:
        logger.info(f"Ingesting raw dataset from: {raw_path}")
        df_raw, file_size_bytes = load_raw_csv(raw_path)

        logger.info("Executing schema validation and type normalization...")
        df_clean, report = validate_and_normalize(df_raw)

        logger.info("Generating dataset profiling metadata...")
        profile = generate_dataset_profile(
            df=df_clean,
            source_file=str(raw_path),
            report=report,
            file_size_bytes=file_size_bytes,
        )

        # Save metadata profile
        save_dataset_profile(profile, metadata_path)
        logger.info(f"Metadata profile written to {metadata_path}")

        print_audit_summary(profile)

        if not report.is_valid:
            logger.error(
                "Dataset validation failed with critical errors. "
                "Parquet output aborted."
            )
            return 1

        # Save cleaned Parquet
        save_processed_parquet(df_clean, output_parquet)
        logger.info(f"Cleaned dataset written to {output_parquet}")
        print("\n[SUCCESS] Data ingestion and validation completed successfully.\n")
        return 0

    except Exception as exc:
        logger.exception(f"Ingestion pipeline failed: {exc}")
        print(f"\n[FATAL ERROR] Ingestion aborted: {exc}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())

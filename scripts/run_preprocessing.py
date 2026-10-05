#!/usr/bin/env python3
"""AERIS Data Preprocessing & Cleaning CLI Script (Data Preparation).

Executes leak-free chronological partitioning, city-aware forward fill,
training median imputation, and StandardScaler fitting strictly on the
training partition.
"""

import argparse
import sys
from pathlib import Path

# Ensure root repository directory is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.core.logging import setup_logging  # noqa: E402
from backend.app.data.preprocessing import (  # noqa: E402
    run_preprocessing_pipeline,
)
from backend.app.data.preprocessing_config import (  # noqa: E402
    PreprocessingConfig,
)


def print_preprocessing_summary(partitions: dict, artifacts, metadata: dict) -> None:
    """Print high-level summary of preprocessing results to stdout."""
    split_meta = metadata["split_summary"]
    val_meta = metadata["validation_summary"]
    pre_miss = metadata["missing_counts_before_preprocessing"]
    post_miss = metadata["missing_counts_after_preprocessing"]

    print("\n" + "=" * 80)
    print("  AERIS: Data Preparation — Cleaning & Preprocessing Pipeline")
    print("=" * 80)
    print(f"Source Dataset     : {metadata['source_dataset']}")
    print(
        f"Total Observations : {val_meta['total_rows']:,} across "
        f"{val_meta['unique_cities']} cities"
    )
    print(f"Date Span          : {val_meta['date_min']} to {val_meta['date_max']}")

    train_dates = (
        f"({split_meta['overall_dates']['train']['start']} to "
        f"{split_meta['overall_dates']['train']['end']})"
    )
    val_dates = (
        f"({split_meta['overall_dates']['val']['start']} to "
        f"{split_meta['overall_dates']['val']['end']})"
    )
    test_dates = (
        f"({split_meta['overall_dates']['test']['start']} to "
        f"{split_meta['overall_dates']['test']['end']})"
    )

    print("\n--- CHRONOLOGICAL PARTITIONS ---")
    print(
        f"Train Partition    : {split_meta['row_counts']['train']:,} rows {train_dates}"
    )
    print(f"Val Partition      : {split_meta['row_counts']['val']:,} rows {val_dates}")
    print(
        f"Test Partition     : {split_meta['row_counts']['test']:,} rows {test_dates}"
    )

    print("\n--- IMPUTATION & LEAK-FREE SCALING ---")
    print(
        f"Scaler Type        : {artifacts.config.scaler_type} "
        f"(fitted strictly on Train)"
    )
    print(
        f"Scaled Features ({len(artifacts.feature_names)}): {artifacts.feature_names}"
    )
    print(f"Target Feature     : {artifacts.target_name} (Standardized)")

    print("\n--- MISSING VALUE RESOLUTION ---")
    header = (
        f"{'Feature':<12} | {'Pre-Impute Missing':<18} | {'Post-Train':<10} | "
        f"{'Post-Val':<9} | {'Post-Test':<9}"
    )
    print(header)
    print("-" * 68)
    for col in artifacts.feature_order:
        pre_cnt = pre_miss.get(col, 0)
        tr_cnt = post_miss["train"].get(col, 0)
        va_cnt = post_miss["val"].get(col, 0)
        te_cnt = post_miss["test"].get(col, 0)
        print(f"{col:<12} | {pre_cnt:<18,d} | {tr_cnt:<10} | {va_cnt:<9} | {te_cnt:<9}")

    tgt_meta = metadata.get("target_validity_summary", {})
    if tgt_meta:
        col_name = tgt_meta.get("target_validity_column", "aqi_target_valid")
        tot_obs = tgt_meta.get("total_observed_targets", 0)
        tot_rows = val_meta["total_rows"]
        tot_pct = (tot_obs / tot_rows * 100.0) if tot_rows > 0 else 0.0
        tr_obs = tgt_meta["train"]["observed"]
        tr_rows = split_meta["row_counts"]["train"]
        tr_pct = tgt_meta["train"]["observed_pct"]
        va_obs = tgt_meta["val"]["observed"]
        va_rows = split_meta["row_counts"]["val"]
        va_pct = tgt_meta["val"]["observed_pct"]
        te_obs = tgt_meta["test"]["observed"]
        te_rows = split_meta["row_counts"]["test"]
        te_pct = tgt_meta["test"]["observed_pct"]

        print("\n--- TARGET AQI GROUND-TRUTH INTEGRITY ---")
        print(f"Target Validity Mask  : '{col_name}' (derived before imputation)")
        print(f"Total Valid Targets   : {tot_obs:,} / {tot_rows:,} ({tot_pct:.2f}%)")
        print(f"Train Valid Targets   : {tr_obs:,} / {tr_rows:,} ({tr_pct:.2f}%)")
        print(f"Val Valid Targets     : {va_obs:,} / {va_rows:,} ({va_pct:.2f}%)")
        print(f"Test Valid Targets    : {te_obs:,} / {te_rows:,} ({te_pct:.2f}%)")
        print(
            "Note: Imputed AQI is available ONLY as historical input feature, "
            "NOT as target label."
        )

    print("\n--- ARTIFACTS CREATED ---")
    for key, path in metadata["saved_files"].items():
        print(f"  * {key:<15}: {path}")
    print(f"  * Scaler Artifact: {artifacts.config.artifacts_dir}/scaler.joblib")
    print(
        f"  * Metadata Record: "
        f"{artifacts.config.artifacts_dir}/preprocessing_metadata.json"
    )


def main() -> int:
    """CLI entrypoint for data preprocessing."""
    parser = argparse.ArgumentParser(
        description="AERIS Data Preparation Preprocessing & Splitting CLI"
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default="data/processed/cleaned_city_day.parquet",
        help="Path to validated cleaned Parquet file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Directory to save train/val/test Parquet files",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="data/artifacts/preprocessing",
        help="Directory to save fitted scaler and artifacts",
    )
    parser.add_argument(
        "--metadata-dir",
        type=str,
        default="data/metadata",
        help="Directory to backup preprocessing metadata",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Proportion for chronological train partition (default: 0.70)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Proportion for chronological val partition (default: 0.15)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Proportion for chronological test partition (default: 0.15)",
    )

    args = parser.parse_args()
    logger = setup_logging()

    config = PreprocessingConfig(
        input_parquet_path=args.input_path,
        output_processed_dir=args.output_dir,
        artifacts_dir=args.artifacts_dir,
        metadata_dir=args.metadata_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    try:
        partitions, artifacts, metadata = run_preprocessing_pipeline(config)
        print_preprocessing_summary(partitions, artifacts, metadata)
        print("\n[SUCCESS] Data preprocessing pipeline completed successfully.\n")
        return 0
    except Exception as exc:
        logger.exception(f"Preprocessing pipeline failed: {exc}")
        print(f"\n[FATAL ERROR] Preprocessing aborted: {exc}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())

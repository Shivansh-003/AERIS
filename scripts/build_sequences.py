#!/usr/bin/env python3
"""AERIS 30-Day Sequence Generation CLI Script (Milestone 5).

Executes backward-only feature engineering (calendar, lags, rolling stats,
rate-of-change, volatility), fits scalers strictly on training data, and packages
leakage-free 30-day lookback supervised sequence tensors.
"""

import argparse
import sys
from pathlib import Path

# Ensure root repository directory is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.core.logging import setup_logging  # noqa: E402
from backend.app.data.feature_config import FeatureConfig  # noqa: E402
from backend.app.data.sequence_config import SequenceConfig  # noqa: E402
from backend.app.data.sequences import (  # noqa: E402
    build_sequences_pipeline,
)


def print_sequence_summary(datasets: dict, artifacts, metadata: dict) -> None:
    """Print structured summary of generated sequence datasets."""
    shapes = metadata["tensor_shapes"]
    filtering = metadata["target_filtering_summary"]
    categories = metadata["feature_categories"]

    print("\n" + "=" * 80)
    print("  AERIS: Feature Engineering & 30-Day Sequence Generation")
    print("=" * 80)
    print(f"Lookback Window (T)    : {metadata['lookback_window']} days")
    print(f"Forecast Horizon (H)   : {metadata['forecast_horizon']} day (t+1)")
    print(f"Boundary Policy        : {metadata['boundary_mode']}")
    print(f"Features per Timestep  : {metadata['feature_count_per_timestep']} channels")

    print("\n--- FEATURE CATEGORIES BREAKDOWN ---")
    for cat_name, feat_list in categories.items():
        print(f"  * {cat_name.title():<15}: {len(feat_list):>2d} features")

    print("\n--- GENERATED SUPERVISED SEQUENCES ---")
    header_seq = (
        f"{'Partition':<10} | {'Input Tensor X':<22} | "
        f"{'Target Vector y':<16} | {'Target Validity':<15}"
    )
    print(header_seq)
    print("-" * 70)
    for split in ["train", "val", "test"]:
        x_s = shapes[split]["X"]
        x_shape_str = f"({x_s[0]:,}, {x_s[1]}, {x_s[2]})"
        y_shape_str = f"({shapes[split]['y'][0]:,},)"
        valid_cnt = f"{filtering[split]['valid_sequences']:,} valid"
        print(
            f"{split.upper():<10} | {x_shape_str:<22} | "
            f"{y_shape_str:<16} | {valid_cnt:<15}"
        )

    print("\n--- TARGET VALIDITY FILTERING ---")
    header = (
        f"{'Partition':<12} | {'Candidate Windows':<18} | {'Skipped Imputed':<18} | "
        f"{'Kept Ground-Truth':<18}"
    )
    print(header)
    print("-" * 72)
    for split in ["train", "val", "test"]:
        cand = filtering[split]["candidates"]
        skip = filtering[split]["skipped_missing_target"]
        kept = filtering[split]["valid_sequences"]
        print(f"{split.upper():<12} | {cand:<18,d} | {skip:<18,d} | {kept:<18,d}")

    print("\n--- ARTIFACTS & TENSOR DATASETS CREATED ---")
    for key, path in metadata["saved_files"].items():
        print(f"  * Dataset [{key}]: {path}")
    for key, path in metadata.get("saved_artifacts", {}).items():
        print(f"  * Artifact [{key}]: {path}")


def main() -> int:
    """CLI entrypoint for sequence generation."""
    parser = argparse.ArgumentParser(
        description="AERIS Feature Engineering & Sequence Generation CLI"
    )
    parser.add_argument(
        "--train-parquet",
        type=str,
        default="data/processed/train_city_day.parquet",
        help="Path to preprocessed train Parquet file",
    )
    parser.add_argument(
        "--val-parquet",
        type=str,
        default="data/processed/val_city_day.parquet",
        help="Path to preprocessed val Parquet file",
    )
    parser.add_argument(
        "--test-parquet",
        type=str,
        default="data/processed/test_city_day.parquet",
        help="Path to preprocessed test Parquet file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed/sequences",
        help="Directory to save generated sequence datasets",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="data/artifacts/sequences",
        help="Directory to save sequence artifacts and scalers",
    )
    parser.add_argument(
        "--lookback-window",
        type=int,
        default=30,
        help="Lookback sequence length T in days (default: 30)",
    )
    parser.add_argument(
        "--boundary-mode",
        type=str,
        default="continuous_history",
        choices=["continuous_history", "isolated_partition"],
        help="Partition boundary handling mode (default: continuous_history)",
    )

    args = parser.parse_args()
    logger = setup_logging()

    seq_config = SequenceConfig(
        input_train_parquet=args.train_parquet,
        input_val_parquet=args.val_parquet,
        input_test_parquet=args.test_parquet,
        output_sequences_dir=args.output_dir,
        artifacts_dir=args.artifacts_dir,
        lookback_window=args.lookback_window,
        boundary_mode=args.boundary_mode,
        save_format="both",
    )

    feat_config = FeatureConfig()

    try:
        datasets, artifacts, metadata = build_sequences_pipeline(
            seq_config=seq_config, feat_config=feat_config
        )
        print_sequence_summary(datasets, artifacts, metadata)
        print(
            "\n[SUCCESS] Feature engineering and sequence generation "
            "completed successfully.\n"
        )
        return 0
    except Exception as exc:
        logger.exception(f"Sequence generation pipeline failed: {exc}")
        print(f"\n[FATAL ERROR] Sequence generation aborted: {exc}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())

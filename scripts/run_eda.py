#!/usr/bin/env python3
"""AERIS Exploratory Data Analysis (EDA) CLI Script.

Executes comprehensive exploratory data analysis, computes descriptive and
correlation statistics, generates research-grade figures, and exports
outputs/eda/eda_report.json.
"""

import argparse
import sys
from pathlib import Path

# Ensure root repository directory is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.core.logging import setup_logging  # noqa: E402
from backend.app.data.eda import run_eda_pipeline  # noqa: E402


def print_eda_summary(report: dict) -> None:
    """Print high-level findings to stdout."""
    overview = report["dataset_overview"]
    missing = report["missing_value_analysis"]
    aqi_stat = report["aqi_distribution_summary"]
    corrs = report["correlation_analysis"]["aqi_correlations"]

    print("\n" + "=" * 80)
    print("  AERIS: Exploratory Data Analysis & Quality Audit Summary")
    print("=" * 80)
    print(
        f"Total Observations : {overview['row_count']:,} across "
        f"{overview['city_count']} cities"
    )
    print(
        f"Date Span          : {overview['date_range']['min_date']} to "
        f"{overview['date_range']['max_date']}"
    )
    print(
        f"AQI Availability   : {aqi_stat['valid_count']:,} valid, "
        f"{missing['aqi_missing_count']:,} missing "
        f"({missing['aqi_missing_percentage']}%)"
    )
    print(
        f"AQI Distribution   : Mean = {aqi_stat['mean']}, "
        f"Median = {aqi_stat['median']}, Std = {aqi_stat['std']}, "
        f"IQR = {aqi_stat['iqr']}"
    )

    print("\n--- KEY POLLUTANT CORRELATIONS WITH AQI ---")
    print(
        f"{'Pollutant':<12} | {'Pearson r':<12} | {'Spearman rho':<12} | "
        f"{'Pairwise N':<12}"
    )
    print("-" * 56)
    for pol in [
        "PM2.5",
        "PM10",
        "CO",
        "NO2",
        "NOx",
        "SO2",
        "O3",
        "NH3",
        "Toluene",
        "Xylene",
        "Benzene",
        "NO",
    ]:
        stat = corrs.get(pol, {})
        print(
            f"{pol:<12} | {stat.get('pearson_r', 'N/A'):<12} | "
            f"{stat.get('spearman_rho', 'N/A'):<12} | "
            f"{stat.get('pairwise_observations', 0):<12,d}"
        )


def main() -> int:
    """CLI entrypoint for EDA execution."""
    parser = argparse.ArgumentParser(description="AERIS Exploratory Data Analysis CLI")
    parser.add_argument(
        "--parquet-path",
        type=str,
        default="data/processed/cleaned_city_day.parquet",
        help="Path to cleaned Parquet dataset",
    )
    parser.add_argument(
        "--report-output",
        type=str,
        default="outputs/eda/eda_report.json",
        help="Path for generated EDA JSON report",
    )
    parser.add_argument(
        "--figures-output",
        type=str,
        default="outputs/eda/figures",
        help="Directory to save generated figures",
    )

    args = parser.parse_args()
    logger = setup_logging()

    try:
        logger.info(f"Running EDA on dataset: {args.parquet_path}")
        report, saved_figs = run_eda_pipeline(
            parquet_path=args.parquet_path,
            report_output=args.report_output,
            figures_output=args.figures_output,
        )
        logger.info(f"EDA report successfully saved to {args.report_output}")
        logger.info(f"Generated {len(saved_figs)} figures in {args.figures_output}")

        print_eda_summary(report)
        print(
            f"\n[SUCCESS] EDA pipeline completed. "
            f"Figures saved to: {args.figures_output}\n"
        )
        return 0
    except Exception as exc:
        logger.exception(f"EDA pipeline execution failed: {exc}")
        print(f"\n[FATAL ERROR] EDA execution failed: {exc}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())

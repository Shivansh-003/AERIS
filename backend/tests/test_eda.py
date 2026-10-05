"""Unit and integration tests for exploratory data analysis (EDA)."""

import json
from pathlib import Path

import pandas as pd
import pytest
from backend.app.data.eda import (
    POLLUTANT_COLUMNS,
    compute_aqi_summary,
    compute_city_statistics,
    compute_correlations,
    compute_dataset_overview,
    compute_duplicate_analysis,
    compute_missing_analysis,
    compute_outlier_analysis,
    compute_pollutant_summaries,
    compute_temporal_statistics,
    plot_all_figures,
    run_eda_pipeline,
)


@pytest.fixture
def processed_df() -> pd.DataFrame:
    """Fixture to load validated cleaned parquet dataset."""
    parquet_path = Path("data/processed/cleaned_city_day.parquet")
    if not parquet_path.exists():
        pytest.skip(
            "Processed dataset not found at data/processed/cleaned_city_day.parquet"
        )
    return pd.read_parquet(parquet_path)


def test_compute_dataset_overview(processed_df: pd.DataFrame):
    """Test overview calculation matches actual dataset dimensions."""
    overview = compute_dataset_overview(processed_df)

    assert overview["row_count"] == 29531
    assert overview["column_count"] == 16
    assert overview["city_count"] == 26
    assert overview["date_range"]["min_date"] == "2015-01-01"
    assert overview["date_range"]["max_date"] == "2020-07-01"
    assert overview["date_range"]["total_days_span"] == 2009
    assert len(overview["observations_per_city"]) == 26
    assert overview["observations_per_city"]["Delhi"] == 2009


def test_compute_missing_analysis(processed_df: pd.DataFrame):
    """Test missingness calculations for AQI, pollutants, and cities."""
    missing = compute_missing_analysis(processed_df)

    assert missing["aqi_missing_count"] == 4681
    assert missing["aqi_missing_percentage"] == 15.85
    assert missing["pollutant_missing_percentages"]["PM2.5"] == 15.57
    assert missing["pollutant_missing_percentages"]["PM10"] == 37.72
    assert missing["pollutant_missing_percentages"]["Xylene"] == 61.32
    assert len(missing["city_missingness"]) == 26


def test_compute_duplicate_analysis(processed_df: pd.DataFrame):
    """Confirm zero exact duplicates and zero (City, Date) collision."""
    dups = compute_duplicate_analysis(processed_df)

    assert dups["duplicate_rows_count"] == 0
    assert dups["duplicate_city_date_count"] == 0


def test_compute_outlier_analysis(processed_df: pd.DataFrame):
    """Test IQR outlier detection yields valid bounds and percentages."""
    outliers = compute_outlier_analysis(processed_df)

    for pol in POLLUTANT_COLUMNS + ["AQI"]:
        assert pol in outliers
        stat = outliers[pol]
        assert stat["q75"] >= stat["q25"]
        assert stat["iqr"] >= 0
        assert stat["upper_bound"] >= stat["lower_bound"]
        assert 0.0 <= stat["outlier_percentage"] <= 100.0

    # Specific known dataset values
    assert outliers["AQI"]["outlier_count"] == 1358
    assert outliers["AQI"]["outlier_percentage"] == 5.46


def test_compute_city_statistics(processed_df: pd.DataFrame):
    """Test city-level aggregations and sorting."""
    city_stats = compute_city_statistics(processed_df)

    assert len(city_stats) == 26
    assert "Delhi" in city_stats
    delhi_stat = city_stats["Delhi"]
    assert delhi_stat["observation_count"] == 2009
    assert delhi_stat["aqi_valid_count"] == 1999
    assert delhi_stat["aqi_mean"] > 200.0


def test_compute_pollutant_summaries(processed_df: pd.DataFrame):
    """Test statistical summaries for each pollutant species."""
    summaries = compute_pollutant_summaries(processed_df)

    assert len(summaries) == len(POLLUTANT_COLUMNS)
    for pol in POLLUTANT_COLUMNS:
        assert pol in summaries
        s = summaries[pol]
        assert s["min"] >= 0.0
        assert s["max"] >= s["min"]
        assert s["valid_count"] > 0


def test_compute_aqi_summary(processed_df: pd.DataFrame):
    """Test composite AQI summary and bucket distribution."""
    aqi_stat = compute_aqi_summary(processed_df)

    assert aqi_stat["valid_count"] == 24850
    assert aqi_stat["missing_count"] == 4681
    assert aqi_stat["mean"] == 166.46
    assert aqi_stat["median"] == 118.0
    assert aqi_stat["min"] == 13.0
    assert aqi_stat["max"] == 2049.0
    assert "Moderate" in aqi_stat["bucket_distribution"]
    assert aqi_stat["bucket_distribution"]["Moderate"] == 8829


def test_compute_temporal_statistics(processed_df: pd.DataFrame):
    """Test yearly and monthly temporal aggregations."""
    temporal = compute_temporal_statistics(processed_df)

    assert "2015" in temporal["yearly_aqi_stats"]
    assert "2020" in temporal["yearly_aqi_stats"]
    assert len(temporal["monthly_aqi_stats"]) == 12

    # Verify seasonal pattern: November (winter) AQI > July (monsoon) AQI
    nov_mean = temporal["monthly_aqi_stats"]["11"]["mean"]
    jul_mean = temporal["monthly_aqi_stats"]["7"]["mean"]
    assert nov_mean > jul_mean


def test_compute_correlations(processed_df: pd.DataFrame):
    """Test Pearson and Spearman correlations between pollutants and AQI."""
    corrs = compute_correlations(processed_df)
    aqi_c = corrs["aqi_correlations"]

    assert "PM2.5" in aqi_c
    assert "PM10" in aqi_c
    assert "NO2" in aqi_c

    # Particulates show highest correlation with AQI
    assert aqi_c["PM10"]["pearson_r"] > 0.75
    assert aqi_c["PM2.5"]["pearson_r"] > 0.60
    assert aqi_c["CO"]["pearson_r"] > 0.60


def test_plot_all_figures(processed_df: pd.DataFrame, tmp_path: Path):
    """Test that all 9 required figures are generated as valid files."""
    figures = plot_all_figures(processed_df, tmp_path)

    expected_files = [
        "aqi_distribution.png",
        "pollutant_distributions.png",
        "correlation_matrix.png",
        "aqi_over_time.png",
        "city_wise_aqi.png",
        "missing_values.png",
        "pm25_vs_aqi.png",
        "pm10_vs_aqi.png",
        "no2_vs_aqi.png",
    ]

    assert len(figures) == 9
    for fname in expected_files:
        fig_file = tmp_path / fname
        assert fig_file.exists(), f"Missing figure file: {fname}"
        assert fig_file.stat().st_size > 1000, (
            f"Figure file is empty/too small: {fname}"
        )


def test_run_eda_pipeline(tmp_path: Path):
    """Integration test: Verify complete pipeline run with real Parquet."""
    report_file = tmp_path / "eda_report.json"
    figures_dir = tmp_path / "figures"

    report, figs = run_eda_pipeline(
        parquet_path="data/processed/cleaned_city_day.parquet",
        report_output=report_file,
        figures_output=figures_dir,
    )

    assert report_file.exists()
    assert len(figs) == 9

    with open(report_file, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["dataset_overview"]["row_count"] == 29531

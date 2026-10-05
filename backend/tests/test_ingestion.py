"""Unit and integration tests for data ingestion, validation, and serialization."""

import json
from pathlib import Path

import pandas as pd
import pytest
from backend.app.data.ingestion import (
    generate_dataset_profile,
    run_ingestion_pipeline,
    save_dataset_profile,
    save_processed_parquet,
    validate_and_normalize,
)
from backend.app.data.schema import (
    CITY_COLUMN,
    DATE_COLUMN,
    POLLUTANT_COLUMNS,
    REQUIRED_COLUMNS,
    TARGET_COLUMN,
)


@pytest.fixture
def sample_valid_raw_df() -> pd.DataFrame:
    """Fixture providing a small, valid raw DataFrame."""
    data = {
        "City": ["Delhi", "Delhi", "Bengaluru"],
        "Date": ["2020-01-01", "2020-01-02", "2020-01-01"],
        "PM2.5": ["120.5", "110.0", "45.2"],
        "PM10": ["210.0", "195.5", "80.0"],
        "NO": ["15.2", "14.8", "5.1"],
        "NO2": ["35.0", "32.1", "12.0"],
        "NOx": ["40.1", "38.5", "14.2"],
        "NH3": ["20.0", "19.5", "8.0"],
        "CO": ["1.2", "1.1", "0.5"],
        "SO2": ["10.5", "11.0", "6.2"],
        "O3": ["45.0", "50.2", "30.1"],
        "Benzene": ["3.5", "3.1", "0.8"],
        "Toluene": ["8.0", "7.5", "2.0"],
        "Xylene": ["2.1", "1.9", "0.4"],
        "AQI": ["310.0", "295.0", "85.0"],
        "AQI_Bucket": ["Very Poor", "Poor", "Satisfactory"],
    }
    return pd.DataFrame(data)


def test_expected_columns_validation(sample_valid_raw_df: pd.DataFrame):
    """Test schema validation succeeds for expected columns and catches missing."""
    # Valid dataframe
    df_clean, report = validate_and_normalize(sample_valid_raw_df)
    assert report.is_valid is True
    assert len(report.missing_columns) == 0
    assert list(df_clean.columns) == REQUIRED_COLUMNS

    # Missing required column
    df_missing = sample_valid_raw_df.drop(columns=["PM2.5"])
    _, report_missing = validate_and_normalize(df_missing)
    assert report_missing.is_valid is False
    assert "PM2.5" in report_missing.missing_columns
    assert any("Missing required columns" in err for err in report_missing.errors)


def test_date_parsing_valid_and_invalid(sample_valid_raw_df: pd.DataFrame):
    """Test date parsing handles valid YYYY-MM-DD dates and records errors."""
    # Valid dates
    df_clean, report = validate_and_normalize(sample_valid_raw_df)
    assert report.invalid_dates_count == 0
    assert pd.api.types.is_datetime64_any_dtype(df_clean[DATE_COLUMN])

    # Invalid dates
    df_invalid_date = sample_valid_raw_df.copy()
    df_invalid_date.loc[0, "Date"] = "invalid-date-format"
    df_invalid_date.loc[1, "Date"] = "2020-02-30"  # Invalid calendar day

    df_clean_inv, report_inv = validate_and_normalize(df_invalid_date)
    assert report_inv.invalid_dates_count == 2
    assert report_inv.is_valid is False
    assert any("invalid Date values" in err for err in report_inv.errors)
    delhi_dates = df_clean_inv[df_clean_inv[CITY_COLUMN] == "Delhi"][DATE_COLUMN]
    assert delhi_dates.isna().all()


def test_numeric_parsing_valid_and_invalid(sample_valid_raw_df: pd.DataFrame):
    """Test numeric columns parse correctly to float64 and catch non-numeric."""
    # Valid numerics
    df_clean, report = validate_and_normalize(sample_valid_raw_df)
    for col in POLLUTANT_COLUMNS + [TARGET_COLUMN]:
        assert pd.api.types.is_float_dtype(df_clean[col])
    assert len(report.invalid_numeric_values) == 0

    # Non-numeric garbage string
    df_invalid_num = sample_valid_raw_df.copy()
    df_invalid_num.loc[0, "PM2.5"] = "not_a_number"
    df_invalid_num.loc[1, "AQI"] = "bad_val"

    _, report_inv = validate_and_normalize(df_invalid_num)
    assert report_inv.is_valid is False
    assert report_inv.invalid_numeric_values.get("PM2.5") == 1
    assert report_inv.invalid_numeric_values.get("AQI") == 1


def test_negative_numeric_values_warning(sample_valid_raw_df: pd.DataFrame):
    """Test that negative numeric values generate validation warnings."""
    df_neg = sample_valid_raw_df.copy()
    df_neg.loc[0, "PM2.5"] = "-15.0"

    _, report = validate_and_normalize(df_neg)
    assert report.negative_values_count.get("PM2.5") == 1
    assert any("negative values" in w for w in report.warnings)


def test_duplicate_detection(sample_valid_raw_df: pd.DataFrame):
    """Test detection of exact row duplicates and (City, Date) collision."""
    # Duplicate complete row
    df_dup_row = pd.concat(
        [sample_valid_raw_df, sample_valid_raw_df.iloc[[0]]],
        ignore_index=True,
    )
    _, report_dup_row = validate_and_normalize(df_dup_row)
    assert report_dup_row.duplicate_rows_count == 1
    assert report_dup_row.duplicate_city_date_count == 1
    assert report_dup_row.is_valid is False

    # Same (City, Date) but differing pollutant values
    df_dup_city_date = sample_valid_raw_df.copy()
    row_mod = df_dup_city_date.iloc[0].to_dict()
    row_mod["PM2.5"] = "999.0"
    df_dup_city_date = pd.concat(
        [df_dup_city_date, pd.DataFrame([row_mod])],
        ignore_index=True,
    )

    _, report_dup_cd = validate_and_normalize(df_dup_city_date)
    assert report_dup_cd.duplicate_rows_count == 0
    assert report_dup_cd.duplicate_city_date_count == 1
    assert report_dup_cd.is_valid is False


def test_metadata_profile_generation(sample_valid_raw_df: pd.DataFrame, tmp_path: Path):
    """Test metadata profile contains all required fields and metrics."""
    df_clean, report = validate_and_normalize(sample_valid_raw_df)
    profile = generate_dataset_profile(
        df=df_clean,
        source_file="test_sample.csv",
        report=report,
        file_size_bytes=1024,
    )

    # Verify root keys
    expected_keys = [
        "metadata_version",
        "generated_at_utc",
        "source",
        "dataset_shape",
        "columns",
        "data_types",
        "geography",
        "temporal_span",
        "integrity_audit",
        "missing_values",
        "target_availability",
        "pollutant_availability",
        "numeric_summary",
        "aqi_bucket_distribution",
        "city_profiles",
    ]
    for key in expected_keys:
        assert key in profile, f"Missing profile key: {key}"

    assert profile["dataset_shape"]["row_count"] == 3
    assert profile["dataset_shape"]["column_count"] == 16
    assert profile["geography"]["city_count"] == 2
    assert "Delhi" in profile["geography"]["cities"]
    assert "Bengaluru" in profile["geography"]["cities"]
    assert profile["temporal_span"]["min_date"] == "2020-01-01"
    assert profile["temporal_span"]["max_date"] == "2020-01-02"
    assert profile["target_availability"]["valid_count"] == 3
    assert profile["target_availability"]["missing_count"] == 0

    # Save to JSON and reload
    json_path = tmp_path / "test_profile.json"
    save_dataset_profile(profile, json_path)
    assert json_path.exists()

    with open(json_path, "r", encoding="utf-8") as f:
        reloaded = json.load(f)
    assert reloaded["dataset_shape"]["row_count"] == 3


def test_processed_parquet_creation_and_reproducibility(
    sample_valid_raw_df: pd.DataFrame, tmp_path: Path
):
    """Test Parquet serialization and deterministic output."""
    df_clean, _ = validate_and_normalize(sample_valid_raw_df)

    parquet_path_1 = tmp_path / "test_1.parquet"
    parquet_path_2 = tmp_path / "test_2.parquet"

    save_processed_parquet(df_clean, parquet_path_1)
    save_processed_parquet(df_clean, parquet_path_2)

    assert parquet_path_1.exists()
    assert parquet_path_2.exists()

    df_read_1 = pd.read_parquet(parquet_path_1)
    df_read_2 = pd.read_parquet(parquet_path_2)

    pd.testing.assert_frame_equal(df_read_1, df_read_2)
    assert len(df_read_1) == 3
    assert list(df_read_1.columns) == REQUIRED_COLUMNS


def test_real_raw_dataset_pipeline(tmp_path: Path):
    """Integration test: Verify pipeline on actual data/raw/city_day.csv."""
    raw_path = Path("data/raw/city_day.csv")
    if not raw_path.exists():
        pytest.skip("Raw dataset not found at data/raw/city_day.csv")

    test_parquet = tmp_path / "output.parquet"
    test_metadata = tmp_path / "profile.json"

    profile = run_ingestion_pipeline(
        raw_path=raw_path,
        output_parquet=test_parquet,
        metadata_path=test_metadata,
    )

    assert profile["integrity_audit"]["is_valid"] is True
    assert profile["dataset_shape"]["row_count"] == 29531
    assert profile["dataset_shape"]["column_count"] == 16
    assert profile["geography"]["city_count"] == 26
    assert profile["temporal_span"]["min_date"] == "2015-01-01"
    assert profile["temporal_span"]["max_date"] == "2020-07-01"
    assert test_parquet.exists()
    assert test_metadata.exists()

    # Verify saved parquet matches expected records
    df_parquet = pd.read_parquet(test_parquet)
    assert len(df_parquet) == 29531
    assert list(df_parquet.columns) == REQUIRED_COLUMNS

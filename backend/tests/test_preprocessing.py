"""Unit and integration tests for the AERIS data preprocessing pipeline."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from backend.app.data.preprocessing import (
    fit_and_apply_imputation,
    fit_and_apply_scalers,
    run_preprocessing_pipeline,
    split_city_chronological,
    validate_input_data,
)
from backend.app.data.preprocessing_artifacts import PreprocessingArtifacts
from backend.app.data.preprocessing_config import PreprocessingConfig
from backend.app.data.schema import POLLUTANT_COLUMNS


@pytest.fixture
def sample_multi_city_df() -> pd.DataFrame:
    """Create a deterministic multi-city multi-date synthetic DataFrame."""
    dates_city_a = pd.date_range("2020-01-01", periods=20, freq="D")
    dates_city_b = pd.date_range("2020-01-01", periods=20, freq="D")

    rows = []
    # City A
    for i, d in enumerate(dates_city_a):
        row = {
            "City": "CityA",
            "Date": d.strftime("%Y-%m-%d"),
            "AQI_Bucket": "Moderate",
        }
        for pol in POLLUTANT_COLUMNS:
            if i == 0 and pol == "PM2.5":
                row[pol] = np.nan
            elif i == 5 and pol == "NO2":
                row[pol] = np.nan
            else:
                row[pol] = float(10.0 + i * 0.5)
        row["AQI"] = float(100.0 + i * 2.0)
        rows.append(row)

    # City B
    for i, d in enumerate(dates_city_b):
        row = {
            "City": "CityB",
            "Date": d.strftime("%Y-%m-%d"),
            "AQI_Bucket": "Poor",
        }
        for pol in POLLUTANT_COLUMNS:
            if i == 2 and pol == "PM10":
                row[pol] = np.nan
            else:
                row[pol] = float(50.0 + i * 1.5)
        row["AQI"] = float(200.0 + i * 3.0)
        rows.append(row)

    return pd.DataFrame(rows).sample(frac=1.0, random_state=42).reset_index(drop=True)


def test_city_date_sorting(sample_multi_city_df: pd.DataFrame):
    """Test that validate_input_data sorts deterministically by City and Date."""
    config = PreprocessingConfig()
    df_sorted, summary = validate_input_data(sample_multi_city_df, config)

    assert df_sorted.iloc[0]["City"] == "CityA"
    assert df_sorted.iloc[0]["Date"] == pd.Timestamp("2020-01-01")
    assert df_sorted.iloc[19]["City"] == "CityA"
    assert df_sorted.iloc[19]["Date"] == pd.Timestamp("2020-01-20")
    assert df_sorted.iloc[20]["City"] == "CityB"
    assert df_sorted.iloc[20]["Date"] == pd.Timestamp("2020-01-01")
    assert summary["unique_cities"] == 2


def test_numeric_conversion_and_validation(sample_multi_city_df: pd.DataFrame):
    """Test numeric column conversion and error detection on corrupted input."""
    config = PreprocessingConfig()
    df_valid, _ = validate_input_data(sample_multi_city_df, config)

    for col in config.all_numeric_model_columns:
        assert pd.api.types.is_float_dtype(df_valid[col])

    # Test corrupted non-numeric string raises ValueError
    df_corrupted = sample_multi_city_df.copy()
    df_corrupted["PM2.5"] = df_corrupted["PM2.5"].astype("object")
    df_corrupted.loc[0, "PM2.5"] = "non_numeric_garbage"
    with pytest.raises(ValueError, match="invalid non-numeric"):
        validate_input_data(df_corrupted, config)


def test_city_isolated_forward_fill():
    """Verify forward-fill respects city boundaries without cross-city bleeding."""
    data = {
        "City": ["CityA", "CityA", "CityB", "CityB"],
        "Date": ["2020-01-01", "2020-01-02", "2020-01-01", "2020-01-02"],
        "AQI_Bucket": ["Good", "Good", "Good", "Good"],
    }
    for pol in POLLUTANT_COLUMNS:
        data[pol] = [10.0, 999.0, np.nan, 25.0]
    data["AQI"] = [50.0, 999.0, np.nan, 75.0]

    df = pd.DataFrame(data)
    config = PreprocessingConfig(train_ratio=0.5, val_ratio=0.25, test_ratio=0.25)
    df_valid, _ = validate_input_data(df, config)

    df_tr, df_va, df_te, medians = fit_and_apply_imputation(
        df_valid, df_valid.iloc[:0], df_valid.iloc[:0], config
    )

    city_b_first_pm25 = df_tr[df_tr["City"] == "CityB"].iloc[0]["PM2.5"]
    assert city_b_first_pm25 != 999.0
    assert city_b_first_pm25 == medians["PM2.5"]


def test_median_imputation_learned_from_train_only():
    """Verify that imputation medians are computed ONLY from training."""
    config = PreprocessingConfig()

    df_train = pd.DataFrame(
        {
            "City": ["CityA", "CityA", "CityA"],
            "Date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
            "PM2.5": [10.0, 20.0, 30.0],
        }
    )
    for pol in POLLUTANT_COLUMNS:
        if pol != "PM2.5":
            df_train[pol] = 10.0
    df_train["AQI"] = 100.0

    df_test = pd.DataFrame(
        {
            "City": ["CityA", "CityA", "CityA"],
            "Date": pd.to_datetime(["2020-01-04", "2020-01-05", "2020-01-06"]),
            "PM2.5": [np.nan, 2000.0, 3000.0],
        }
    )
    for pol in POLLUTANT_COLUMNS:
        if pol != "PM2.5":
            df_test[pol] = 100.0
    df_test["AQI"] = 500.0

    df_tr_imp, _, df_te_imp, medians = fit_and_apply_imputation(
        df_train, df_train.iloc[:0], df_test, config
    )

    assert medians["PM2.5"] == 20.0
    assert df_te_imp.iloc[0]["PM2.5"] == 20.0


def test_chronological_ordering_and_no_train_test_overlap(
    sample_multi_city_df: pd.DataFrame,
):
    """Test that splits preserve chronological ordering with zero overlap."""
    config = PreprocessingConfig(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
    df_valid, _ = validate_input_data(sample_multi_city_df, config)
    df_train, df_val, df_test, _ = split_city_chronological(df_valid, config)

    assert len(df_train) + len(df_val) + len(df_test) == len(df_valid)

    for city in ["CityA", "CityB"]:
        c_tr = df_train[df_train["City"] == city]
        c_va = df_val[df_val["City"] == city]
        c_te = df_test[df_test["City"] == city]

        assert not c_tr.empty
        assert not c_va.empty
        assert not c_te.empty

        assert c_tr["Date"].max() < c_va["Date"].min()
        assert c_va["Date"].max() < c_te["Date"].min()

        keys_tr = set(zip(c_tr["City"], c_tr["Date"]))
        keys_va = set(zip(c_va["City"], c_va["Date"]))
        keys_te = set(zip(c_te["City"], c_te["Date"]))

        assert len(keys_tr.intersection(keys_va)) == 0
        assert len(keys_tr.intersection(keys_te)) == 0
        assert len(keys_va.intersection(keys_te)) == 0


def test_standard_scaler_fitted_only_on_train_and_applied_to_val_test():
    """Verify StandardScaler parameters equal training statistics only."""
    config = PreprocessingConfig()

    df_train = pd.DataFrame(
        {
            "City": ["CityA", "CityA", "CityA"],
            "Date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
            "PM2.5": [10.0, 20.0, 30.0],
        }
    )
    for pol in POLLUTANT_COLUMNS:
        if pol != "PM2.5":
            df_train[pol] = 50.0
    df_train["AQI"] = 100.0

    df_val = df_train.copy()
    df_val["PM2.5"] = [100.0, 200.0, 300.0]

    df_test = df_train.copy()
    df_test["PM2.5"] = [1000.0, 2000.0, 3000.0]

    medians = {col: 10.0 for col in config.all_numeric_model_columns}
    medians["PM2.5"] = 20.0

    df_tr_s, df_va_s, df_te_s, artifacts = fit_and_apply_scalers(
        df_train, df_val, df_test, medians, config
    )

    pm25_idx = artifacts.feature_names.index("PM2.5")
    train_mean = artifacts.scaler.mean_[pm25_idx]
    train_scale = artifacts.scaler.scale_[pm25_idx]

    assert train_mean == pytest.approx(20.0, rel=1e-4)
    assert df_tr_s["scaled_PM2.5"].iloc[1] == pytest.approx(0.0, abs=1e-5)

    expected_val_0 = (100.0 - train_mean) / train_scale
    assert df_va_s["scaled_PM2.5"].iloc[0] == pytest.approx(expected_val_0, rel=1e-4)

    expected_test_0 = (1000.0 - train_mean) / train_scale
    assert df_te_s["scaled_PM2.5"].iloc[0] == pytest.approx(expected_test_0, rel=1e-4)


def test_anti_leakage_synthetic_distribution_shift():
    """MANDATORY LEAKAGE TEST: Synthesize shift and prove scaler immunity."""
    np.random.seed(42)
    n_train = 70
    n_val = 15
    n_test = 15

    train_pm25 = np.random.normal(loc=10.0, scale=2.0, size=n_train)
    val_pm25 = np.random.normal(loc=50.0, scale=5.0, size=n_val)
    test_pm25 = np.random.normal(loc=500.0, scale=20.0, size=n_test)

    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    df = pd.DataFrame(
        {
            "City": ["CityA"] * 100,
            "Date": dates,
            "PM2.5": np.concatenate([train_pm25, val_pm25, test_pm25]),
            "AQI_Bucket": ["Moderate"] * 100,
        }
    )
    for pol in POLLUTANT_COLUMNS:
        if pol != "PM2.5":
            df[pol] = 25.0
    df["AQI"] = 150.0

    config = PreprocessingConfig(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
    df_valid, _ = validate_input_data(df, config)
    df_tr, df_va, df_te, _ = split_city_chronological(df_valid, config)
    df_tr_imp, df_va_imp, df_te_imp, medians = fit_and_apply_imputation(
        df_tr, df_va, df_te, config
    )
    df_tr_s, df_va_s, df_te_s, artifacts = fit_and_apply_scalers(
        df_tr_imp, df_va_imp, df_te_imp, medians, config
    )

    pm25_idx = artifacts.feature_names.index("PM2.5")
    fitted_mean = artifacts.scaler.mean_[pm25_idx]
    fitted_scale = artifacts.scaler.scale_[pm25_idx]

    # Scaler mean must strictly match train distribution (~10.0)
    assert 9.0 < fitted_mean < 11.0, f"Leakage detected: mean={fitted_mean}"
    assert 1.5 < fitted_scale < 2.5, f"Leakage detected: scale={fitted_scale}"

    mean_test_scaled = df_te_s["scaled_PM2.5"].mean()
    assert mean_test_scaled > 200.0, f"Expected z-score > 200: {mean_test_scaled}"


def test_extreme_values_are_not_silently_removed():
    """Verify extreme pollution spikes (e.g. AQI=2049) are preserved."""
    data = {
        "City": ["CityA", "CityA", "CityA"],
        "Date": ["2020-01-01", "2020-01-02", "2020-01-03"],
        "AQI_Bucket": ["Severe", "Severe", "Severe"],
        "PM2.5": [850.0, 920.0, 949.0],
        "AQI": [1200.0, 1500.0, 2049.0],
    }
    for pol in POLLUTANT_COLUMNS:
        if pol != "PM2.5":
            data[pol] = 150.0

    df = pd.DataFrame(data)
    config = PreprocessingConfig(train_ratio=0.34, val_ratio=0.33, test_ratio=0.33)
    df_valid, _ = validate_input_data(df, config)

    assert len(df_valid) == 3
    assert df_valid["PM2.5"].max() == 949.0
    assert df_valid["AQI"].max() == 2049.0


def test_artifacts_save_and_load(tmp_path: Path, sample_multi_city_df: pd.DataFrame):
    """Test full artifact serialization and roundtrip loading."""
    config = PreprocessingConfig(
        artifacts_dir=str(tmp_path / "artifacts"),
        metadata_dir=str(tmp_path / "metadata"),
        scalers_backup_dir=str(tmp_path / "scalers"),
    )
    df_valid, _ = validate_input_data(sample_multi_city_df, config)
    df_tr, df_va, df_te, _ = split_city_chronological(df_valid, config)
    df_tr_imp, df_va_imp, df_te_imp, medians = fit_and_apply_imputation(
        df_tr, df_va, df_te, config
    )
    _, _, _, artifacts = fit_and_apply_scalers(
        df_tr_imp, df_va_imp, df_te_imp, medians, config
    )

    saved_paths = artifacts.save(
        artifacts_dir=config.artifacts_dir,
        scalers_backup_dir=config.scalers_backup_dir,
        metadata_dir=config.metadata_dir,
    )

    assert Path(saved_paths["scaler"]).exists()
    assert Path(saved_paths["imputation_statistics"]).exists()
    assert Path(saved_paths["metadata"]).exists()

    loaded = PreprocessingArtifacts.load(config.artifacts_dir)
    assert loaded.feature_names == artifacts.feature_names
    assert loaded.target_name == artifacts.target_name
    assert loaded.imputation_medians == artifacts.imputation_medians
    np.testing.assert_array_almost_equal(loaded.scaler.mean_, artifacts.scaler.mean_)


def test_aqi_target_valid_flag_preservation():
    """Verify that aqi_target_valid is derived pre-imputation and preserved."""
    config = PreprocessingConfig()
    dates = pd.date_range("2020-01-01", periods=6, freq="D")
    df = pd.DataFrame(
        {
            "City": ["CityA"] * 6,
            "Date": dates,
            "AQI_Bucket": ["Moderate"] * 6,
            "AQI": [100.0, np.nan, 120.0, np.nan, 140.0, 150.0],
        }
    )
    for pol in POLLUTANT_COLUMNS:
        df[pol] = 20.0

    # 1. Validate step derives target validity flag
    df_valid, summary = validate_input_data(df, config)
    assert "aqi_target_valid" in df_valid.columns
    assert summary["observed_target_count"] == 4
    assert summary["missing_target_count"] == 2
    assert df_valid["aqi_target_valid"].tolist() == [
        True,
        False,
        True,
        False,
        True,
        True,
    ]

    # 2. Imputation step fills AQI but preserves false in aqi_target_valid
    df_tr, df_va, df_te, medians = fit_and_apply_imputation(
        df_valid.iloc[:4], df_valid.iloc[4:5], df_valid.iloc[5:], config
    )
    assert df_tr["AQI"].isna().sum() == 0  # Imputed for historical features
    assert df_tr["aqi_target_valid"].tolist() == [True, False, True, False]

    # 3. Scaling step preserves aqi_target_valid intact
    df_tr_s, df_va_s, df_te_s, _ = fit_and_apply_scalers(
        df_tr, df_va, df_te, medians, config
    )
    assert df_tr_s["aqi_target_valid"].tolist() == [True, False, True, False]
    assert df_va_s["aqi_target_valid"].tolist() == [True]
    assert df_te_s["aqi_target_valid"].tolist() == [True]


def test_per_city_chronological_boundaries_all_cities():
    """Verify that every single city strictly satisfies chronological partitioning."""
    src_parquet = Path("data/processed/cleaned_city_day.parquet")
    if not src_parquet.exists():
        pytest.skip("Dataset not found at data/processed/cleaned_city_day.parquet")

    config = PreprocessingConfig(input_parquet_path=str(src_parquet))
    df_valid, _ = validate_input_data(pd.read_parquet(src_parquet), config)
    df_tr, df_va, df_te, meta = split_city_chronological(df_valid, config)

    assert meta["validation_summary"] if "validation_summary" in meta else True

    for city in df_valid["City"].unique():
        c_tr = df_tr[df_tr["City"] == city]
        c_va = df_va[df_va["City"] == city]
        c_te = df_te[df_te["City"] == city]

        assert not c_tr.empty, f"City {city} train is empty"
        assert not c_va.empty, f"City {city} val is empty"
        assert not c_te.empty, f"City {city} test is empty"

        assert c_tr["Date"].max() < c_va["Date"].min(), (
            f"Chronological boundary violated for city {city}: "
            f"train max {c_tr['Date'].max()} >= val min {c_va['Date'].min()}"
        )
        assert c_va["Date"].max() < c_te["Date"].min(), (
            f"Chronological boundary violated for city {city}: "
            f"val max {c_va['Date'].max()} >= test min {c_te['Date'].min()}"
        )


def test_real_dataset_preprocessing_pipeline(tmp_path: Path):
    """Integration test: Execute full pipeline on cleaned_city_day.parquet."""
    src_parquet = Path("data/processed/cleaned_city_day.parquet")
    if not src_parquet.exists():
        pytest.skip("Dataset not found at data/processed/cleaned_city_day.parquet")

    config = PreprocessingConfig(
        input_parquet_path=str(src_parquet),
        output_processed_dir=str(tmp_path / "processed"),
        artifacts_dir=str(tmp_path / "artifacts"),
        metadata_dir=str(tmp_path / "metadata"),
        scalers_backup_dir=str(tmp_path / "scalers"),
    )

    partitions, artifacts, metadata = run_preprocessing_pipeline(config)

    assert len(partitions["train"]) == 20661
    assert len(partitions["val"]) == 4417
    assert len(partitions["test"]) == 4453
    total_len = (
        len(partitions["train"]) + len(partitions["val"]) + len(partitions["test"])
    )
    assert total_len == 29531

    assert (tmp_path / "processed" / "train_city_day.parquet").exists()
    assert (tmp_path / "processed" / "val_city_day.parquet").exists()
    assert (tmp_path / "processed" / "test_city_day.parquet").exists()

    # Verify target validity flags
    assert metadata["target_validity_summary"]["total_observed_targets"] == 24850
    assert metadata["target_validity_summary"]["total_missing_targets"] == 4681

    for split_name, df_part in partitions.items():
        assert "aqi_target_valid" in df_part.columns
        assert (
            df_part["aqi_target_valid"].dtype == bool
            or df_part["aqi_target_valid"].dtype == "bool"
        )
        for col in config.all_numeric_model_columns:
            assert df_part[col].isna().sum() == 0, f"NaN in {split_name} col {col}"
            assert df_part[f"scaled_{col}"].isna().sum() == 0, (
                f"NaN in {split_name} scaled_{col}"
            )

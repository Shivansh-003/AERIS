"""Unit and integration tests for AERIS feature engineering and sequence generation."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from backend.app.data.feature_config import FeatureConfig
from backend.app.data.features import (
    compute_calendar_features,
    compute_lag_features,
    compute_roc_features,
    compute_rolling_features,
    compute_volatility_features,
    engineer_all_features,
    fit_and_scale_engineered_features,
)
from backend.app.data.sequence_artifacts import SequenceArtifacts
from backend.app.data.sequence_config import SequenceConfig
from backend.app.data.sequences import (
    build_sequences_pipeline,
    extract_city_sequences,
)


@pytest.fixture
def sample_timeseries_df() -> pd.DataFrame:
    """Create a deterministic multi-city multi-date synthetic DataFrame for testing."""
    dates_a = pd.date_range("2020-01-01", periods=40, freq="D")
    dates_b = pd.date_range("2020-01-01", periods=40, freq="D")

    rows = []
    # City A: Linear progression
    for i, d in enumerate(dates_a):
        row = {
            "City": "CityA",
            "Date": d.strftime("%Y-%m-%d"),
            "AQI_Bucket": "Moderate",
            "aqi_target_valid": True if i not in (10, 35) else False,
        }
        for pol in [
            "PM2.5",
            "PM10",
            "NO",
            "NO2",
            "NOx",
            "NH3",
            "CO",
            "SO2",
            "O3",
            "Benzene",
            "Toluene",
            "Xylene",
        ]:
            row[pol] = float(10.0 + i * 1.0)
            row[f"scaled_{pol}"] = float((10.0 + i * 1.0 - 25.0) / 10.0)
        row["AQI"] = float(100.0 + i * 2.0)
        row["scaled_AQI"] = float((100.0 + i * 2.0 - 140.0) / 20.0)
        rows.append(row)

    # City B: Step progression
    for i, d in enumerate(dates_b):
        row = {
            "City": "CityB",
            "Date": d.strftime("%Y-%m-%d"),
            "AQI_Bucket": "Poor",
            "aqi_target_valid": True,
        }
        for pol in [
            "PM2.5",
            "PM10",
            "NO",
            "NO2",
            "NOx",
            "NH3",
            "CO",
            "SO2",
            "O3",
            "Benzene",
            "Toluene",
            "Xylene",
        ]:
            row[pol] = float(50.0 + i * 0.5)
            row[f"scaled_{pol}"] = float((50.0 + i * 0.5 - 60.0) / 10.0)
        row["AQI"] = float(200.0 + i * 1.5)
        row["scaled_AQI"] = float((200.0 + i * 1.5 - 230.0) / 20.0)
        rows.append(row)

    return pd.DataFrame(rows)


def test_calendar_features_determinism(sample_timeseries_df: pd.DataFrame):
    """Test deterministic cyclical trigonometric encoding of calendar variables."""
    config = FeatureConfig()
    df_cal, names = compute_calendar_features(sample_timeseries_df, config)

    assert "day_of_week_sin" in names
    assert "day_of_week_cos" in names
    assert "month_sin" in names
    assert "season_sin" in names

    # Verify unit circle bounds: sin^2 + cos^2 == 1.0
    for prefix in ["day_of_week", "month", "season"]:
        sin_vals = df_cal[f"{prefix}_sin"].to_numpy()
        cos_vals = df_cal[f"{prefix}_cos"].to_numpy()
        np.testing.assert_allclose(
            sin_vals**2 + cos_vals**2,
            1.0,
            rtol=1e-5,
            err_msg=f"{prefix} sin/cos not on unit circle",
        )


def test_backward_lags_isolation_per_city(sample_timeseries_df: pd.DataFrame):
    """Test lag calculations strictly respect city boundaries and backward direction."""
    config = FeatureConfig(lag_columns=["PM2.5"], lag_intervals=[1, 3])
    df_lag, _ = compute_lag_features(sample_timeseries_df, config)

    city_a = df_lag[df_lag["City"] == "CityA"].reset_index(drop=True)
    city_b = df_lag[df_lag["City"] == "CityB"].reset_index(drop=True)

    # City A: values are 10.0, 11.0, 12.0 ...
    assert city_a.loc[0, "PM2.5_lag_1"] == 10.0  # anchored
    assert city_a.loc[1, "PM2.5_lag_1"] == 10.0
    assert city_a.loc[5, "PM2.5_lag_1"] == 14.0
    assert city_a.loc[5, "PM2.5_lag_3"] == 12.0

    # City B: first row lag MUST NOT use City A's last row (50.0 vs ~49.0)
    assert city_b.loc[0, "PM2.5_lag_1"] == 50.0
    assert city_b.loc[1, "PM2.5_lag_1"] == 50.0


def test_backward_rolling_statistics(sample_timeseries_df: pd.DataFrame):
    """Test that rolling statistics strictly look backward [t-w+1 ... t]."""
    config = FeatureConfig(
        rolling_columns=["AQI"],
        rolling_mean_windows=[3],
        rolling_std_windows=[7],
        rolling_min_max_windows=[7],
    )
    df_roll, _ = compute_rolling_features(sample_timeseries_df, config)

    city_a = df_roll[df_roll["City"] == "CityA"].reset_index(drop=True)

    # City A AQI: 100, 102, 104, 106, 108...
    # Mean of window 3 at idx 2: (100 + 102 + 104) / 3 = 102.0
    assert city_a.loc[2, "AQI_roll_mean_3"] == pytest.approx(102.0)
    # Mean of window 3 at idx 3: (102 + 104 + 106) / 3 = 104.0
    assert city_a.loc[3, "AQI_roll_mean_3"] == pytest.approx(104.0)

    # Min / Max over 7 days at idx 6: min=100.0, max=112.0
    assert city_a.loc[6, "AQI_roll_min_7"] == pytest.approx(100.0)
    assert city_a.loc[6, "AQI_roll_max_7"] == pytest.approx(112.0)


def test_rate_of_change_and_volatility_features(sample_timeseries_df: pd.DataFrame):
    """Test backward ROC and CV volatility calculations."""
    config = FeatureConfig(
        roc_columns=["PM2.5"],
        roc_intervals=[1],
        volatility_columns=["PM2.5"],
        volatility_windows=[7],
    )
    df_roc, _ = compute_roc_features(sample_timeseries_df, config)
    df_vol, _ = compute_volatility_features(df_roc, config)

    assert "PM2.5_roc_1" in df_vol.columns
    assert "PM2.5_volatility_7" in df_vol.columns
    assert not np.isnan(df_vol["PM2.5_roc_1"]).any()
    assert not np.isnan(df_vol["PM2.5_volatility_7"]).any()


def test_mandatory_synthetic_future_shift_leakage_immunity(
    sample_timeseries_df: pd.DataFrame,
):
    """MANDATORY ANTI-LEAKAGE TEST: t+1 mutation must NOT change features at t."""
    config = FeatureConfig()

    # 1. Base engineered features
    df_base, _ = engineer_all_features(sample_timeseries_df, config)

    # 2. Mutate future records at index 35 (extreme pollution surge of 10,000)
    df_corrupted = sample_timeseries_df.copy()
    for pol in config.base_pollutant_columns + ["AQI"]:
        df_corrupted.loc[35, pol] = 10000.0

    df_corrupted_feat, _ = engineer_all_features(df_corrupted, config)

    # Features at t=34 must remain strictly identical
    row_base_34 = df_base.iloc[34]
    row_corrupted_34 = df_corrupted_feat.iloc[34]

    for col in df_base.columns:
        if col in ["City", "Date", "AQI_Bucket", "aqi_target_valid"]:
            continue
        val_orig = row_base_34[col]
        val_corr = row_corrupted_34[col]
        assert val_orig == val_corr or np.isclose(val_orig, val_corr, rtol=1e-5), (
            f"LEAKAGE DETECTED in column '{col}' at t=34 when t=35 was modified!"
        )


def test_sequence_window_shape_and_target_alignment(
    sample_timeseries_df: pd.DataFrame,
):
    """Verify that extract_city_sequences creates correct (N, 30, D) tensor."""
    seq_config = SequenceConfig(lookback_window=30, forecast_horizon=1)
    feat_config = FeatureConfig()

    df_feat, _ = engineer_all_features(sample_timeseries_df, feat_config)
    feature_cols = [
        c
        for c in df_feat.columns
        if c not in ["City", "Date", "AQI_Bucket", "aqi_target_valid"]
    ]

    city_a = df_feat[df_feat["City"] == "CityA"].reset_index(drop=True)

    x_arr, y_arr, y_s_arr, meta_list, counts = extract_city_sequences(
        df_city=city_a, feature_cols=feature_cols, config=seq_config
    )

    # 40 rows with T=30: candidate indices i=29..38 -> targets at i+1=30..39
    # Row 35 was marked aqi_target_valid=False, so target at i=35 (index 5) is skipped
    assert x_arr.ndim == 3
    assert x_arr.shape[1] == 30
    assert x_arr.shape[2] == len(feature_cols)
    assert len(x_arr) == len(y_arr) == len(meta_list)
    assert counts["total_candidates"] == 10
    assert counts["skipped_missing_target"] == 1
    assert counts["kept_windows"] == 9
    assert len(x_arr) == 9

    # Verify target alignment: Sample 0 has window rows 0..29, target is row 30
    target_row_30_aqi = city_a.loc[30, "AQI"]
    assert y_arr[0] == pytest.approx(target_row_30_aqi)
    assert meta_list[0]["target_aqi"] == pytest.approx(target_row_30_aqi)
    assert meta_list[0]["target_date"] == city_a.loc[30, "Date"]


def test_forbidden_columns_excluded_from_model_features(
    sample_timeseries_df: pd.DataFrame,
):
    """Verify metadata and target columns are strictly excluded from input features."""
    feat_config = FeatureConfig()
    df_tr_f, _, _, _, model_features, _ = fit_and_scale_engineered_features(
        sample_timeseries_df,
        sample_timeseries_df.iloc[:5],
        sample_timeseries_df.iloc[:5],
        feat_config,
    )

    forbidden = ["AQI_Bucket", "aqi_target_valid", "City", "Date", "split"]
    for col in forbidden:
        assert col not in model_features, (
            f"Forbidden column '{col}' present in model features!"
        )


def test_artifacts_save_and_load(tmp_path: Path, sample_timeseries_df: pd.DataFrame):
    """Test SequenceArtifacts serialization and roundtrip loading."""
    seq_config = SequenceConfig(
        artifacts_dir=str(tmp_path / "artifacts"),
        metadata_dir=str(tmp_path / "metadata"),
    )
    feature_names = ["scaled_PM2.5", "month_sin", "PM2.5_lag_1"]
    metadata = {
        "test_key": "test_val",
        "feature_categories": {"base": ["scaled_PM2.5"]},
    }

    artifacts = SequenceArtifacts(
        feature_scaler=None,
        feature_names=feature_names,
        feature_categories={"base": ["scaled_PM2.5"]},
        metadata=metadata,
        config=seq_config,
    )

    saved_files = artifacts.save(
        artifacts_dir=seq_config.artifacts_dir,
        metadata_dir=seq_config.metadata_dir,
    )

    assert Path(saved_files["metadata"]).exists()
    assert Path(saved_files["feature_info"]).exists()

    loaded = SequenceArtifacts.load(seq_config.artifacts_dir)
    assert loaded.feature_names == feature_names
    assert loaded.metadata["test_key"] == "test_val"


def test_real_dataset_sequence_generation_pipeline(tmp_path: Path):
    """Integration test: Full sequence pipeline on partition files."""
    train_parquet = Path("data/processed/train_city_day.parquet")
    val_parquet = Path("data/processed/val_city_day.parquet")
    test_parquet = Path("data/processed/test_city_day.parquet")

    if (
        not train_parquet.exists()
        or not val_parquet.exists()
        or not test_parquet.exists()
    ):
        pytest.skip("Preprocessed partition datasets missing.")

    seq_config = SequenceConfig(
        input_train_parquet=str(train_parquet),
        input_val_parquet=str(val_parquet),
        input_test_parquet=str(test_parquet),
        output_sequences_dir=str(tmp_path / "sequences"),
        artifacts_dir=str(tmp_path / "artifacts"),
        metadata_dir=str(tmp_path / "metadata"),
        lookback_window=30,
        forecast_horizon=1,
        boundary_mode="continuous_history",
        save_format="both",
    )

    feat_config = FeatureConfig()

    datasets, artifacts, metadata = build_sequences_pipeline(
        seq_config=seq_config, feat_config=feat_config
    )

    # 1. Verify sequence sample counts
    assert len(datasets["train"]["X"]) == 15852
    assert len(datasets["val"]["X"]) == 4261
    assert len(datasets["test"]["X"]) == 4312

    # 2. Verify tensor shapes
    n_feats = metadata["feature_count_per_timestep"]
    assert datasets["train"]["X"].shape == (15852, 30, n_feats)
    assert datasets["val"]["X"].shape == (4261, 30, n_feats)
    assert datasets["test"]["X"].shape == (4312, 30, n_feats)

    # 3. Verify PyTorch (.pt) and NumPy (.npz) file creation
    out_dir = tmp_path / "sequences"
    assert (out_dir / "train_sequences.pt").exists()
    assert (out_dir / "val_sequences.pt").exists()
    assert (out_dir / "test_sequences.pt").exists()
    assert (out_dir / "train_sequences.npz").exists()
    assert (out_dir / "val_sequences.npz").exists()
    assert (out_dir / "test_sequences.npz").exists()

    # 4. Verify PyTorch tensor loading and type integrity
    loaded_pt = torch.load(out_dir / "train_sequences.pt", weights_only=False)
    assert isinstance(loaded_pt["X"], torch.Tensor)
    assert loaded_pt["X"].shape == torch.Size([15852, 30, n_feats])
    assert loaded_pt["y"].shape == torch.Size([15852])
    assert not torch.isnan(loaded_pt["X"]).any()
    assert not torch.isnan(loaded_pt["y"]).any()

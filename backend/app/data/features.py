"""AERIS Feature Engineering Pipeline.

Implements leakage-free backward-only temporal, lag, rolling, rate-of-change,
and volatility feature generators strictly bounded per city.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from backend.app.data.feature_config import FeatureConfig
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("aeris.data.features")


def compute_calendar_features(
    df: pd.DataFrame, config: FeatureConfig
) -> Tuple[pd.DataFrame, List[str]]:
    """Compute deterministic cyclical calendar and seasonal representations.

    Args:
        df: Input DataFrame containing a Date column.
        config: Feature engineering configuration.

    Returns:
        Tuple of (enriched DataFrame, list of created feature names).
    """
    df_out = df.copy()
    dt = pd.to_datetime(df_out[config.date_column])

    dow = dt.dt.dayofweek
    month = dt.dt.month
    doy = dt.dt.dayofyear
    season = month.map(config.month_to_season).fillna(0).astype(int)

    cal_features = {
        "day_of_week_sin": np.sin(2.0 * np.pi * dow / 7.0),
        "day_of_week_cos": np.cos(2.0 * np.pi * dow / 7.0),
        "month_sin": np.sin(2.0 * np.pi * month / 12.0),
        "month_cos": np.cos(2.0 * np.pi * month / 12.0),
        "day_of_year_sin": np.sin(2.0 * np.pi * doy / 365.25),
        "day_of_year_cos": np.cos(2.0 * np.pi * doy / 365.25),
        "season_sin": np.sin(2.0 * np.pi * season / 4.0),
        "season_cos": np.cos(2.0 * np.pi * season / 4.0),
    }

    feature_names: List[str] = []
    for feat_name, feat_series in cal_features.items():
        df_out[feat_name] = feat_series.astype("float64")
        feature_names.append(feat_name)

    return df_out, feature_names


def compute_lag_features(
    df: pd.DataFrame, config: FeatureConfig
) -> Tuple[pd.DataFrame, List[str]]:
    """Compute backward-looking lag features strictly within city boundaries.

    Args:
        df: Input DataFrame sorted by [City, Date].
        config: Feature engineering configuration.

    Returns:
        Tuple of (enriched DataFrame, list of created feature names).
    """
    df_out = df.copy()
    grouped = df_out.groupby(config.city_column, group_keys=False)
    created_features: List[str] = []

    for col in config.lag_columns:
        if col not in df_out.columns:
            continue
        for k in config.lag_intervals:
            feat_name = f"{col}_lag_{k}"
            # Backward shift per city; anchor initial rows to first observation
            df_out[feat_name] = (
                grouped[col].transform(lambda s: s.shift(k).bfill()).astype("float64")
            )
            created_features.append(feat_name)

    return df_out, created_features


def compute_rolling_features(
    df: pd.DataFrame, config: FeatureConfig
) -> Tuple[pd.DataFrame, List[str]]:
    """Compute backward-looking rolling statistics strictly within city boundaries.

    Args:
        df: Input DataFrame sorted by [City, Date].
        config: Feature engineering configuration.

    Returns:
        Tuple of (enriched DataFrame, list of created feature names).
    """
    df_out = df.copy()
    grouped = df_out.groupby(config.city_column, group_keys=False)
    created_features: List[str] = []

    for col in config.rolling_columns:
        if col not in df_out.columns:
            continue

        # Rolling means
        for w in config.rolling_mean_windows:
            feat_name = f"{col}_roll_mean_{w}"
            df_out[feat_name] = (
                grouped[col]
                .transform(lambda s: s.rolling(w, min_periods=1).mean())
                .astype("float64")
            )
            created_features.append(feat_name)

        # Rolling standard deviations
        for w in config.rolling_std_windows:
            feat_name = f"{col}_roll_std_{w}"
            df_out[feat_name] = (
                grouped[col]
                .transform(lambda s: s.rolling(w, min_periods=1).std().fillna(0.0))
                .astype("float64")
            )
            created_features.append(feat_name)

        # Rolling min / max
        for w in config.rolling_min_max_windows:
            min_feat = f"{col}_roll_min_{w}"
            max_feat = f"{col}_roll_max_{w}"
            df_out[min_feat] = (
                grouped[col]
                .transform(lambda s: s.rolling(w, min_periods=1).min())
                .astype("float64")
            )
            df_out[max_feat] = (
                grouped[col]
                .transform(lambda s: s.rolling(w, min_periods=1).max())
                .astype("float64")
            )
            created_features.extend([min_feat, max_feat])

    return df_out, created_features


def compute_roc_features(
    df: pd.DataFrame, config: FeatureConfig
) -> Tuple[pd.DataFrame, List[str]]:
    """Compute backward-looking Rate-of-Change (ROC) features per city.

    Formula: ROC_t = (p_t - p_{t-k}) / (|p_{t-k}| + epsilon)

    Args:
        df: Input DataFrame sorted by [City, Date].
        config: Feature engineering configuration.

    Returns:
        Tuple of (enriched DataFrame, list of created feature names).
    """
    df_out = df.copy()
    grouped = df_out.groupby(config.city_column, group_keys=False)
    created_features: List[str] = []

    for col in config.roc_columns:
        if col not in df_out.columns:
            continue
        for k in config.roc_intervals:
            feat_name = f"{col}_roc_{k}"
            prev_val = grouped[col].transform(lambda s: s.shift(k).bfill())
            diff = df_out[col] - prev_val
            denom = prev_val.abs() + config.roc_epsilon
            roc = (diff / denom).clip(config.roc_clip_min, config.roc_clip_max)
            df_out[feat_name] = roc.astype("float64")
            created_features.append(feat_name)

    return df_out, created_features


def compute_volatility_features(
    df: pd.DataFrame, config: FeatureConfig
) -> Tuple[pd.DataFrame, List[str]]:
    """Compute backward-looking pollution volatility (Coefficient of Variation).

    Formula: CV_t = std_t / (|mean_t| + epsilon)

    Args:
        df: Input DataFrame sorted by [City, Date].
        config: Feature engineering configuration.

    Returns:
        Tuple of (enriched DataFrame, list of created feature names).
    """
    df_out = df.copy()
    grouped = df_out.groupby(config.city_column, group_keys=False)
    created_features: List[str] = []

    for col in config.volatility_columns:
        if col not in df_out.columns:
            continue
        for w in config.volatility_windows:
            feat_name = f"{col}_volatility_{w}"
            r_mean = grouped[col].transform(
                lambda s: s.rolling(w, min_periods=1).mean()
            )
            r_std = grouped[col].transform(
                lambda s: s.rolling(w, min_periods=1).std().fillna(0.0)
            )
            cv = (r_std / (r_mean.abs() + config.volatility_epsilon)).clip(
                0.0, config.volatility_clip_max
            )
            df_out[feat_name] = cv.astype("float64")
            created_features.append(feat_name)

    return df_out, created_features


def engineer_all_features(
    df: pd.DataFrame, config: Optional[FeatureConfig] = None
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """Execute complete leakage-free feature engineering on a dataset.

    Args:
        df: Input DataFrame sorted chronologically by [City, Date].
        config: Feature engineering configuration.

    Returns:
        Tuple of (enriched DataFrame, dictionary of feature category lists).
    """
    if config is None:
        config = FeatureConfig()

    df_out = df.sort_values(
        by=[config.city_column, config.date_column], ascending=[True, True]
    ).reset_index(drop=True)

    df_cal, cal_names = compute_calendar_features(df_out, config)
    df_lag, lag_names = compute_lag_features(df_cal, config)
    df_roll, roll_names = compute_rolling_features(df_lag, config)
    df_roc, roc_names = compute_roc_features(df_roll, config)
    df_final, vol_names = compute_volatility_features(df_roc, config)

    feature_categories = {
        "calendar": cal_names,
        "lag": lag_names,
        "rolling": roll_names,
        "roc": roc_names,
        "volatility": vol_names,
    }

    return df_final, feature_categories


def fit_and_scale_engineered_features(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    config: Optional[FeatureConfig] = None,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    Optional[StandardScaler],
    List[str],
    Dict[str, List[str]],
]:
    """Engineer features and fit scaler strictly on the training partition.

    Args:
        df_train: Preprocessed training DataFrame.
        df_val: Preprocessed validation DataFrame.
        df_test: Preprocessed test DataFrame.
        config: Optional FeatureConfig instance.

    Returns:
        Tuple of (train_feat, val_feat, test_feat, fitted_scaler,
                  final_model_feature_names, feature_categories).
    """
    if config is None:
        config = FeatureConfig()

    logger.info("Engineering features across train, validation, and test partitions...")

    df_tr_feat, categories = engineer_all_features(df_train, config)
    df_va_feat, _ = engineer_all_features(df_val, config)
    df_te_feat, _ = engineer_all_features(df_test, config)

    # Collect continuous engineered features needing scaling
    # (Excludes cyclical sin/cos features which are already bounded in [-1, 1])
    continuous_eng_cols = (
        categories["lag"]
        + categories["rolling"]
        + categories["roc"]
        + categories["volatility"]
    )

    feature_scaler: Optional[StandardScaler] = None
    if config.scale_engineered_features and continuous_eng_cols:
        logger.info(
            f"Fitting StandardScaler on {len(continuous_eng_cols)} "
            "engineered continuous features (Train only)..."
        )
        feature_scaler = StandardScaler()
        feature_scaler.fit(df_tr_feat[continuous_eng_cols])

        tr_scaled = feature_scaler.transform(df_tr_feat[continuous_eng_cols])
        va_scaled = feature_scaler.transform(df_va_feat[continuous_eng_cols])
        te_scaled = feature_scaler.transform(df_te_feat[continuous_eng_cols])

        for idx, col in enumerate(continuous_eng_cols):
            df_tr_feat[col] = tr_scaled[:, idx]
            df_va_feat[col] = va_scaled[:, idx]
            df_te_feat[col] = te_scaled[:, idx]

    # Assemble complete ordered list of final model features
    # 1. Base standardized pollutants and AQI
    base_scaled_cols = [
        f"scaled_{col}"
        for col in config.base_pollutant_columns
        if f"scaled_{col}" in df_tr_feat.columns
    ]
    if f"scaled_{config.target_column}" in df_tr_feat.columns:
        base_scaled_cols.append(f"scaled_{config.target_column}")

    # 2. Calendar features (bounded sin/cos)
    cal_cols = categories["calendar"]

    # 3. Continuous engineered features (scaled)
    eng_cols = continuous_eng_cols

    final_model_features = base_scaled_cols + cal_cols + eng_cols

    logger.info(
        f"Final model feature set compiled: {len(final_model_features)} "
        "features per timestep."
    )

    return (
        df_tr_feat,
        df_va_feat,
        df_te_feat,
        feature_scaler,
        final_model_features,
        categories,
    )

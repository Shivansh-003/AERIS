"""AERIS Data Preprocessing, Chronological Partitioning, and Scaling Pipeline."""

import datetime
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from backend.app.data.preprocessing_artifacts import PreprocessingArtifacts
from backend.app.data.preprocessing_config import PreprocessingConfig
from backend.app.data.schema import CITY_COLUMN, DATE_COLUMN
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("aeris.data.preprocessing")


def validate_input_data(
    df: pd.DataFrame, config: PreprocessingConfig
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Validate data integrity, column presence, non-negativity, and sort order.

    Args:
        df: Input DataFrame.
        config: Preprocessing configuration.

    Returns:
        Tuple of validated DataFrame and validation summary dictionary.

    Raises:
        ValueError: If mandatory ID or model columns are missing.
    """
    required_cols = config.id_columns + config.all_numeric_model_columns
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Input DataFrame is missing required columns: {missing_cols}")

    df_out = df.copy()

    # 1. Normalize Date to datetime
    if not pd.api.types.is_datetime64_any_dtype(df_out[DATE_COLUMN]):
        df_out[DATE_COLUMN] = pd.to_datetime(
            df_out[DATE_COLUMN], format="%Y-%m-%d", errors="coerce"
        )
        invalid_dates = int(df_out[DATE_COLUMN].isna().sum())
        if invalid_dates > 0:
            raise ValueError(f"Found {invalid_dates} invalid dates in input data.")

    # 2. Convert City to string
    df_out[CITY_COLUMN] = df_out[CITY_COLUMN].astype("string").str.strip()

    # 3. Numeric conversion and validation
    invalid_numeric_counts: Dict[str, int] = {}
    negative_counts: Dict[str, int] = {}
    infinite_counts: Dict[str, int] = {}

    for col in config.all_numeric_model_columns:
        s = pd.to_numeric(df_out[col], errors="coerce")
        # Check non-numeric strings
        invalid_mask = df_out[col].notna() & s.isna()
        inv_cnt = int(invalid_mask.sum())
        if inv_cnt > 0:
            invalid_numeric_counts[col] = inv_cnt
            raise ValueError(
                f"Column '{col}' has {inv_cnt} invalid non-numeric values."
            )

        # Check infinite values
        inf_mask = np.isinf(s)
        inf_cnt = int(inf_mask.sum())
        if inf_cnt > 0:
            infinite_counts[col] = inf_cnt
            raise ValueError(f"Column '{col}' contains {inf_cnt} infinite values.")

        # Check negative physical concentrations / AQI
        neg_mask = s < 0
        neg_cnt = int(neg_mask.sum())
        if neg_cnt > 0:
            negative_counts[col] = neg_cnt
            logger.warning(f"Column '{col}' contains {neg_cnt} negative values.")

        df_out[col] = s.astype("float64")

    # 4. Target validity tracking (derived strictly BEFORE any imputation/filling)
    df_out[config.target_validity_column] = (
        df_out[config.target_column].notna().astype(bool)
    )

    # 5. Outlier detection (IQR method - reporting only, no deletion)
    outlier_summary: Dict[str, Dict[str, Any]] = {}
    for col in config.all_numeric_model_columns:
        valid_s = df_out[col].dropna()
        if not valid_s.empty:
            q25 = float(valid_s.quantile(0.25))
            q75 = float(valid_s.quantile(0.75))
            iqr = q75 - q25
            lb, ub = q25 - 1.5 * iqr, q75 + 1.5 * iqr
            out_cnt = int(((valid_s < lb) | (valid_s > ub)).sum())
            outlier_summary[col] = {
                "q25": round(q25, 4),
                "q75": round(q75, 4),
                "iqr": round(iqr, 4),
                "lower_bound": round(lb, 4),
                "upper_bound": round(ub, 4),
                "outlier_count": out_cnt,
                "outlier_pct": round((out_cnt / len(valid_s)) * 100.0, 2),
            }

    # 6. Deterministic Chronological Sorting by City and Date
    df_out = df_out.sort_values(
        by=[CITY_COLUMN, DATE_COLUMN], ascending=[True, True]
    ).reset_index(drop=True)

    validation_summary = {
        "total_rows": len(df_out),
        "total_columns": len(df_out.columns),
        "unique_cities": int(df_out[CITY_COLUMN].nunique()),
        "date_min": df_out[DATE_COLUMN].min().strftime("%Y-%m-%d"),
        "date_max": df_out[DATE_COLUMN].max().strftime("%Y-%m-%d"),
        "target_validity_column": config.target_validity_column,
        "observed_target_count": int(df_out[config.target_validity_column].sum()),
        "missing_target_count": int((~df_out[config.target_validity_column]).sum()),
        "invalid_numeric_counts": invalid_numeric_counts,
        "negative_counts": negative_counts,
        "infinite_counts": infinite_counts,
        "outlier_summary": outlier_summary,
    }

    return df_out, validation_summary


def split_city_chronological(
    df: pd.DataFrame, config: PreprocessingConfig
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Chronologically partition dataset into Train, Validation, and Test subsets.

    The split is strictly evaluated within each city's chronological timeline.
    No cross-city contamination occurs.

    Args:
        df: Validated DataFrame sorted by [City, Date].
        config: Preprocessing configuration with split ratios.

    Returns:
        Tuple of (df_train, df_val, df_test, split_metadata).
    """
    train_frames: List[pd.DataFrame] = []
    val_frames: List[pd.DataFrame] = []
    test_frames: List[pd.DataFrame] = []

    city_split_details: Dict[str, Dict[str, Any]] = {}

    for city_name, group in df.groupby(CITY_COLUMN, sort=True):
        city_str = str(city_name)
        city_df = group.sort_values(by=DATE_COLUMN, ascending=True).reset_index(
            drop=True
        )
        n_total = len(city_df)

        n_train = int(n_total * config.train_ratio)
        n_val = int(n_total * config.val_ratio)

        # Partition slices
        c_train = city_df.iloc[:n_train].copy()
        c_val = city_df.iloc[n_train : n_train + n_val].copy()
        c_test = city_df.iloc[n_train + n_val :].copy()

        # Check temporal ordering
        if not c_train.empty and not c_val.empty:
            assert c_train[DATE_COLUMN].max() < c_val[DATE_COLUMN].min(), (
                f"Temporal leak in {city_str}: Train max >= Val min"
            )
        if not c_val.empty and not c_test.empty:
            assert c_val[DATE_COLUMN].max() < c_test[DATE_COLUMN].min(), (
                f"Temporal leak in {city_str}: Val max >= Test min"
            )

        train_frames.append(c_train)
        val_frames.append(c_val)
        test_frames.append(c_test)

        tr_start = (
            c_train[DATE_COLUMN].min().strftime("%Y-%m-%d")
            if not c_train.empty
            else None
        )
        tr_end = (
            c_train[DATE_COLUMN].max().strftime("%Y-%m-%d")
            if not c_train.empty
            else None
        )
        va_start = (
            c_val[DATE_COLUMN].min().strftime("%Y-%m-%d") if not c_val.empty else None
        )
        va_end = (
            c_val[DATE_COLUMN].max().strftime("%Y-%m-%d") if not c_val.empty else None
        )
        te_start = (
            c_test[DATE_COLUMN].min().strftime("%Y-%m-%d") if not c_test.empty else None
        )
        te_end = (
            c_test[DATE_COLUMN].max().strftime("%Y-%m-%d") if not c_test.empty else None
        )

        city_split_details[city_str] = {
            "total_rows": n_total,
            "train_rows": len(c_train),
            "val_rows": len(c_val),
            "test_rows": len(c_test),
            "train_dates": {"start": tr_start, "end": tr_end},
            "val_dates": {"start": va_start, "end": va_end},
            "test_dates": {"start": te_start, "end": te_end},
        }

    df_train = (
        pd.concat(train_frames, ignore_index=True)
        .sort_values(by=[CITY_COLUMN, DATE_COLUMN])
        .reset_index(drop=True)
    )
    df_val = (
        pd.concat(val_frames, ignore_index=True)
        .sort_values(by=[CITY_COLUMN, DATE_COLUMN])
        .reset_index(drop=True)
    )
    df_test = (
        pd.concat(test_frames, ignore_index=True)
        .sort_values(by=[CITY_COLUMN, DATE_COLUMN])
        .reset_index(drop=True)
    )

    split_metadata = {
        "ratios": {
            "train": config.train_ratio,
            "val": config.val_ratio,
            "test": config.test_ratio,
        },
        "row_counts": {
            "train": len(df_train),
            "val": len(df_val),
            "test": len(df_test),
            "total": len(df),
        },
        "overall_dates": {
            "train": {
                "start": df_train[DATE_COLUMN].min().strftime("%Y-%m-%d"),
                "end": df_train[DATE_COLUMN].max().strftime("%Y-%m-%d"),
            },
            "val": {
                "start": df_val[DATE_COLUMN].min().strftime("%Y-%m-%d"),
                "end": df_val[DATE_COLUMN].max().strftime("%Y-%m-%d"),
            },
            "test": {
                "start": df_test[DATE_COLUMN].min().strftime("%Y-%m-%d"),
                "end": df_test[DATE_COLUMN].max().strftime("%Y-%m-%d"),
            },
        },
        "city_splits": city_split_details,
    }

    return df_train, df_val, df_test, split_metadata


def fit_and_apply_imputation(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    config: PreprocessingConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """Impute missing numeric values using city forward-fill and train medians.

    Core Leakage Guardrail:
    1. Temporal forward fill is applied strictly within each City's partition.
    2. Column medians are computed ONLY on the training partition.
    3. The train-learned medians are then applied to fill remaining NaNs in Train,
       Validation, and Test partitions.

    Args:
        df_train: Training DataFrame.
        df_val: Validation DataFrame.
        df_test: Test DataFrame.
        config: Preprocessing configuration.

    Returns:
        Tuple of (df_train_imp, df_val_imp, df_test_imp, learned_train_medians).
    """
    cols = config.all_numeric_model_columns
    df_tr_out = df_train.copy()
    df_va_out = df_val.copy()
    df_te_out = df_test.copy()

    # 1. Forward-fill within each City (respecting chronological order)
    df_tr_out[cols] = df_tr_out.groupby(CITY_COLUMN)[cols].ffill()
    df_va_out[cols] = df_va_out.groupby(CITY_COLUMN)[cols].ffill()
    df_te_out[cols] = df_te_out.groupby(CITY_COLUMN)[cols].ffill()

    # 2. Compute median strictly on the forward-filled TRAINING subset
    learned_medians: Dict[str, float] = {}
    for col in cols:
        med_val = df_tr_out[col].dropna().median()
        if pd.isna(med_val):
            med_val = 0.0
        learned_medians[col] = float(med_val)

    # 3. Impute remaining missing values across partitions with train medians
    for col in cols:
        df_tr_out[col] = df_tr_out[col].fillna(learned_medians[col])
        df_va_out[col] = df_va_out[col].fillna(learned_medians[col])
        df_te_out[col] = df_te_out[col].fillna(learned_medians[col])

    # Assert zero remaining missing values in required model features
    assert df_tr_out[cols].isna().sum().sum() == 0, "Train has remaining NaNs!"
    assert df_va_out[cols].isna().sum().sum() == 0, "Validation has NaNs!"
    assert df_te_out[cols].isna().sum().sum() == 0, "Test has remaining NaNs!"

    return df_tr_out, df_va_out, df_te_out, learned_medians


def fit_and_apply_scalers(
    df_train_imp: pd.DataFrame,
    df_val_imp: pd.DataFrame,
    df_test_imp: pd.DataFrame,
    learned_medians: Dict[str, float],
    config: PreprocessingConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, PreprocessingArtifacts]:
    """Standardize features using StandardScaler fitted ONLY on training data.

    Args:
        df_train_imp: Imputed training DataFrame.
        df_val_imp: Imputed validation DataFrame.
        df_test_imp: Imputed test DataFrame.
        learned_medians: Dictionary of train-learned imputation medians.
        config: Preprocessing configuration.

    Returns:
        Tuple of (df_train_scaled, df_val_scaled, df_test_scaled, artifacts).
    """
    feature_cols = list(config.feature_columns)
    target_col = config.target_column

    # 1. Fit feature scaler strictly on Training partition
    feature_scaler = StandardScaler()
    feature_scaler.fit(df_train_imp[feature_cols])

    # 2. Fit optional target scaler strictly on Training partition
    target_scaler = StandardScaler() if config.scale_target else None
    if target_scaler is not None:
        target_scaler.fit(df_train_imp[[target_col]])

    # 3. Transform partitions
    def _transform_df(
        df_in: pd.DataFrame,
    ) -> pd.DataFrame:
        df_out = df_in.copy()
        scaled_feat_arr = feature_scaler.transform(df_out[feature_cols])
        for idx, col in enumerate(feature_cols):
            df_out[f"scaled_{col}"] = scaled_feat_arr[:, idx]

        if target_scaler is not None:
            scaled_tgt_arr = target_scaler.transform(df_out[[target_col]])
            df_out[f"scaled_{target_col}"] = scaled_tgt_arr[:, 0]

        return df_out

    df_train_scaled = _transform_df(df_train_imp)
    df_val_scaled = _transform_df(df_val_imp)
    df_test_scaled = _transform_df(df_test_imp)

    # 4. Compile training statistics metadata
    training_stats: Dict[str, Dict[str, float]] = {}
    for idx, col in enumerate(feature_cols):
        training_stats[col] = {
            "mean": float(feature_scaler.mean_[idx]),
            "variance": float(feature_scaler.var_[idx]),
            "scale": float(feature_scaler.scale_[idx]),
            "median": float(learned_medians.get(col, 0.0)),
            "min": float(df_train_imp[col].min()),
            "max": float(df_train_imp[col].max()),
        }

    if target_scaler is not None:
        training_stats[target_col] = {
            "mean": float(target_scaler.mean_[0]),
            "variance": float(target_scaler.var_[0]),
            "scale": float(target_scaler.scale_[0]),
            "median": float(learned_medians.get(target_col, 0.0)),
            "min": float(df_train_imp[target_col].min()),
            "max": float(df_train_imp[target_col].max()),
        }

    artifacts = PreprocessingArtifacts(
        scaler=feature_scaler,
        target_scaler=target_scaler,
        imputation_medians=learned_medians,
        feature_names=feature_cols,
        target_name=target_col,
        feature_order=config.all_numeric_model_columns,
        training_stats=training_stats,
        config=config,
    )

    return df_train_scaled, df_val_scaled, df_test_scaled, artifacts


def run_preprocessing_pipeline(
    config: Optional[PreprocessingConfig] = None,
) -> Tuple[Dict[str, pd.DataFrame], PreprocessingArtifacts, Dict[str, Any]]:
    """Execute end-to-end data preprocessing pipeline.

    Args:
        config: Optional PreprocessingConfig instance.

    Returns:
        Tuple of (partitions_dict, artifacts, metadata).
    """
    if config is None:
        config = PreprocessingConfig()

    logger.info("Starting AERIS data preprocessing pipeline...")

    # 1. Load validated source Parquet dataset
    src_path = Path(config.input_parquet_path)
    if not src_path.exists():
        raise FileNotFoundError(f"Validated dataset not found at: {src_path}")
    df_raw = pd.read_parquet(src_path)
    logger.info(f"Loaded {len(df_raw)} records from {src_path}")

    # Track missing values before preprocessing
    pre_missing_counts = {
        col: int(df_raw[col].isna().sum()) for col in config.all_numeric_model_columns
    }

    # 2. Validate input schema and sort deterministically
    df_valid, val_summary = validate_input_data(df_raw, config)

    # 3. Perform city-level chronological train/val/test splitting
    df_train_raw, df_val_raw, df_test_raw, split_meta = split_city_chronological(
        df_valid, config
    )
    logger.info(
        f"Chronological split: Train={len(df_train_raw)}, "
        f"Val={len(df_val_raw)}, Test={len(df_test_raw)}"
    )

    # 4. Leakage-safe Imputation (City forward-fill + Train medians)
    df_train_imp, df_val_imp, df_test_imp, learned_medians = fit_and_apply_imputation(
        df_train_raw, df_val_raw, df_test_raw, config
    )
    logger.info(f"Imputation completed using {len(learned_medians)} train medians.")

    # 5. Fit StandardScaler strictly on Train and transform partitions
    df_train_scaled, df_val_scaled, df_test_scaled, artifacts = fit_and_apply_scalers(
        df_train_imp, df_val_imp, df_test_imp, learned_medians, config
    )
    logger.info("StandardScaler fitted strictly on training partition.")

    # 6. Save processed partition Parquet files
    out_dir = Path(config.output_processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_out_path = out_dir / "train_city_day.parquet"
    val_out_path = out_dir / "val_city_day.parquet"
    test_out_path = out_dir / "test_city_day.parquet"

    df_train_scaled.to_parquet(train_out_path, engine="pyarrow", index=False)
    df_val_scaled.to_parquet(val_out_path, engine="pyarrow", index=False)
    df_test_scaled.to_parquet(test_out_path, engine="pyarrow", index=False)
    logger.info(f"Processed partitions saved to {out_dir}")

    # 7. Compile and save artifacts and metadata
    metadata = {
        "source_dataset": str(config.input_parquet_path).replace("\\", "/"),
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "pipeline_version": config.pipeline_version,
        "split_summary": split_meta,
        "validation_summary": val_summary,
        "target_validity_summary": {
            "target_validity_column": config.target_validity_column,
            "total_observed_targets": int(
                df_valid[config.target_validity_column].sum()
            ),
            "total_missing_targets": int(
                (~df_valid[config.target_validity_column]).sum()
            ),
            "train": {
                "observed": int(df_train_scaled[config.target_validity_column].sum()),
                "missing": int((~df_train_scaled[config.target_validity_column]).sum()),
                "observed_pct": round(
                    float(
                        df_train_scaled[config.target_validity_column].mean() * 100.0
                    ),
                    2,
                ),
            },
            "val": {
                "observed": int(df_val_scaled[config.target_validity_column].sum()),
                "missing": int((~df_val_scaled[config.target_validity_column]).sum()),
                "observed_pct": round(
                    float(df_val_scaled[config.target_validity_column].mean() * 100.0),
                    2,
                ),
            },
            "test": {
                "observed": int(df_test_scaled[config.target_validity_column].sum()),
                "missing": int((~df_test_scaled[config.target_validity_column]).sum()),
                "observed_pct": round(
                    float(df_test_scaled[config.target_validity_column].mean() * 100.0),
                    2,
                ),
            },
        },
        "missing_counts_before_preprocessing": pre_missing_counts,
        "missing_counts_after_preprocessing": {
            "train": {
                col: int(df_train_scaled[col].isna().sum())
                for col in config.all_numeric_model_columns
            },
            "val": {
                col: int(df_val_scaled[col].isna().sum())
                for col in config.all_numeric_model_columns
            },
            "test": {
                col: int(df_test_scaled[col].isna().sum())
                for col in config.all_numeric_model_columns
            },
        },
        "saved_files": {
            "train_parquet": str(train_out_path).replace("\\", "/"),
            "val_parquet": str(val_out_path).replace("\\", "/"),
            "test_parquet": str(test_out_path).replace("\\", "/"),
        },
    }

    artifacts.metadata = metadata
    saved_arts = artifacts.save(
        artifacts_dir=config.artifacts_dir,
        scalers_backup_dir=config.scalers_backup_dir,
        metadata_dir=config.metadata_dir,
    )
    logger.info(f"Preprocessing artifacts saved: {list(saved_arts.keys())}")

    partitions = {
        "train": df_train_scaled,
        "val": df_val_scaled,
        "test": df_test_scaled,
    }

    return partitions, artifacts, metadata

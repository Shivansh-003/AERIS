"""AERIS 30-Day Supervised Sequence Generation Engine.

Transforms engineered multi-city tabular time-series into leakage-safe
sliding lookback tensors (N, 30, D) and single-day-ahead targets (N,)
guaranteed to evaluate strictly on authentic observed CPCB AQI records.
"""

import datetime
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from backend.app.data.feature_config import FeatureConfig
from backend.app.data.features import fit_and_scale_engineered_features
from backend.app.data.sequence_artifacts import SequenceArtifacts
from backend.app.data.sequence_config import SequenceConfig

logger = logging.getLogger("aeris.data.sequences")


def extract_city_sequences(
    df_city: pd.DataFrame,
    feature_cols: List[str],
    config: SequenceConfig,
    target_split: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]], Dict[str, int]]:
    """Extract sliding window sequences for a single city's chronological timeline.

    Args:
        df_city: DataFrame for a single city sorted ascending by Date.
        feature_cols: Ordered list of feature column names.
        config: SequenceConfig instance.
        target_split: Optional partition filter ('train', 'val', 'test').

    Returns:
        Tuple of (X_array, y_array, y_scaled_array, sample_metadata_list, counts_dict).
    """
    t_len = config.lookback_window
    horizon = config.forecast_horizon
    n_rows = len(df_city)

    x_list: List[np.ndarray] = []
    y_list: List[float] = []
    y_scaled_list: List[float] = []
    meta_list: List[Dict[str, Any]] = []

    total_candidate_windows = 0
    skipped_missing_target = 0
    kept_windows = 0

    if n_rows < t_len + horizon:
        counts = {
            "total_candidates": 0,
            "skipped_missing_target": 0,
            "kept_windows": 0,
        }
        return (
            np.empty((0, t_len, len(feature_cols)), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            [],
            counts,
        )

    # Pre-extract numpy matrices for high performance
    feat_matrix = df_city[feature_cols].to_numpy(dtype=np.float32)
    target_vals = df_city[config.target_column].to_numpy(dtype=np.float32)
    target_scaled_vals = df_city[config.scaled_target_column].to_numpy(dtype=np.float32)
    valid_flags = df_city[config.target_validity_column].to_numpy(dtype=bool)
    dates = df_city[config.date_column].to_numpy()
    city_name = str(df_city[config.city_column].iloc[0])

    has_split_col = "split" in df_city.columns
    split_tags = (
        df_city["split"].to_numpy() if has_split_col else np.full(n_rows, "train")
    )

    for i in range(t_len - 1, n_rows - horizon):
        target_idx = i + horizon
        target_split_tag = str(split_tags[target_idx])

        # Filter by partition if specified
        if target_split is not None and target_split_tag != target_split:
            continue

        total_candidate_windows += 1
        is_target_valid = bool(valid_flags[target_idx])

        if config.filter_missing_targets and not is_target_valid:
            skipped_missing_target += 1
            continue

        # Extract window: [i - t_len + 1 ... i] inclusive (length t_len)
        window_x = feat_matrix[i - t_len + 1 : i + 1]
        target_y = float(target_vals[target_idx])
        target_y_scaled = float(target_scaled_vals[target_idx])

        input_start_dt = str(pd.Timestamp(dates[i - t_len + 1]).date())
        input_end_dt = str(pd.Timestamp(dates[i]).date())
        target_dt = str(pd.Timestamp(dates[target_idx]).date())

        x_list.append(window_x)
        y_list.append(target_y)
        y_scaled_list.append(target_y_scaled)

        sample_meta = {
            "city": city_name,
            "input_start_date": input_start_dt,
            "input_end_date": input_end_dt,
            "target_date": target_dt,
            "target_aqi": target_y,
            "target_aqi_scaled": target_y_scaled,
            "target_validity": is_target_valid,
            "split": target_split_tag,
        }
        meta_list.append(sample_meta)
        kept_windows += 1

    if kept_windows > 0:
        x_arr = np.stack(x_list, axis=0)
        y_arr = np.array(y_list, dtype=np.float32)
        y_scaled_arr = np.array(y_scaled_list, dtype=np.float32)
    else:
        x_arr = np.empty((0, t_len, len(feature_cols)), dtype=np.float32)
        y_arr = np.empty((0,), dtype=np.float32)
        y_scaled_arr = np.empty((0,), dtype=np.float32)

    counts = {
        "total_candidates": total_candidate_windows,
        "skipped_missing_target": skipped_missing_target,
        "kept_windows": kept_windows,
    }

    return x_arr, y_arr, y_scaled_arr, meta_list, counts


def build_sequences_pipeline(
    seq_config: Optional[SequenceConfig] = None,
    feat_config: Optional[FeatureConfig] = None,
) -> Tuple[Dict[str, Dict[str, Any]], SequenceArtifacts, Dict[str, Any]]:
    """Execute end-to-end feature engineering and 30-day sequence tensor generation.

    Args:
        seq_config: Optional SequenceConfig instance.
        feat_config: Optional FeatureConfig instance.

    Returns:
        Tuple of (datasets_dict, artifacts, summary_metadata).
    """
    if seq_config is None:
        seq_config = SequenceConfig()
    if feat_config is None:
        feat_config = FeatureConfig()

    logger.info("Starting AERIS sequence generation pipeline...")

    # 1. Load validated and partitioned Parquet datasets
    train_path = Path(seq_config.input_train_parquet)
    val_path = Path(seq_config.input_val_parquet)
    test_path = Path(seq_config.input_test_parquet)

    if not train_path.exists() or not val_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Missing partition files: {train_path}, {val_path}, {test_path}"
        )

    df_train_raw = pd.read_parquet(train_path)
    df_val_raw = pd.read_parquet(val_path)
    df_test_raw = pd.read_parquet(test_path)

    logger.info(
        f"Loaded partition datasets: Train={len(df_train_raw)}, "
        f"Val={len(df_val_raw)}, Test={len(df_test_raw)}"
    )

    # 2. Execute feature engineering and train-only scaling
    (
        df_tr_feat,
        df_va_feat,
        df_te_feat,
        feature_scaler,
        final_model_features,
        feature_categories,
    ) = fit_and_scale_engineered_features(
        df_train=df_train_raw,
        df_val=df_val_raw,
        df_test=df_test_raw,
        config=feat_config,
    )

    # Tag partition splits for continuous history tracking
    df_tr_feat["split"] = "train"
    df_va_feat["split"] = "val"
    df_te_feat["split"] = "test"

    datasets: Dict[str, Dict[str, Any]] = {}
    partition_counts: Dict[str, Dict[str, int]] = {}

    if seq_config.boundary_mode == "continuous_history":
        # Combine into continuous per-city chronological timeline
        df_all_feat = (
            pd.concat([df_tr_feat, df_va_feat, df_te_feat], ignore_index=True)
            .sort_values(
                by=[seq_config.city_column, seq_config.date_column],
                ascending=[True, True],
            )
            .reset_index(drop=True)
        )

        for split_name in ["train", "val", "test"]:
            x_parts: List[np.ndarray] = []
            y_parts: List[np.ndarray] = []
            y_s_parts: List[np.ndarray] = []
            meta_parts: List[Dict[str, Any]] = []

            tot_cand = 0
            tot_skip = 0
            tot_kept = 0

            for _, city_group in df_all_feat.groupby(seq_config.city_column, sort=True):
                city_sorted = city_group.sort_values(
                    by=seq_config.date_column, ascending=True
                ).reset_index(drop=True)
                (
                    c_x,
                    c_y,
                    c_ys,
                    c_meta,
                    c_counts,
                ) = extract_city_sequences(
                    df_city=city_sorted,
                    feature_cols=final_model_features,
                    config=seq_config,
                    target_split=split_name,
                )
                if len(c_x) > 0:
                    x_parts.append(c_x)
                    y_parts.append(c_y)
                    y_s_parts.append(c_ys)
                    meta_parts.extend(c_meta)

                tot_cand += c_counts["total_candidates"]
                tot_skip += c_counts["skipped_missing_target"]
                tot_kept += c_counts["kept_windows"]

            if x_parts:
                part_x = np.concatenate(x_parts, axis=0)
                part_y = np.concatenate(y_parts, axis=0)
                part_ys = np.concatenate(y_s_parts, axis=0)
            else:
                part_x = np.empty(
                    (0, seq_config.lookback_window, len(final_model_features)),
                    dtype=np.float32,
                )
                part_y = np.empty((0,), dtype=np.float32)
                part_ys = np.empty((0,), dtype=np.float32)

            datasets[split_name] = {
                "X": part_x,
                "y": part_y,
                "y_scaled": part_ys,
                "metadata": meta_parts,
            }
            partition_counts[split_name] = {
                "candidates": tot_cand,
                "skipped_missing_target": tot_skip,
                "valid_sequences": tot_kept,
            }

    else:
        # Isolated partition mode (slices partitions independently)
        for split_name, part_df in [
            ("train", df_tr_feat),
            ("val", df_va_feat),
            ("test", df_te_feat),
        ]:
            x_parts = []
            y_parts = []
            y_s_parts = []
            meta_parts = []
            tot_cand = 0
            tot_skip = 0
            tot_kept = 0

            for _, city_group in part_df.groupby(seq_config.city_column, sort=True):
                city_sorted = city_group.sort_values(
                    by=seq_config.date_column, ascending=True
                ).reset_index(drop=True)
                (
                    c_x,
                    c_y,
                    c_ys,
                    c_meta,
                    c_counts,
                ) = extract_city_sequences(
                    df_city=city_sorted,
                    feature_cols=final_model_features,
                    config=seq_config,
                    target_split=None,
                )
                if len(c_x) > 0:
                    x_parts.append(c_x)
                    y_parts.append(c_y)
                    y_s_parts.append(c_ys)
                    meta_parts.extend(c_meta)

                tot_cand += c_counts["total_candidates"]
                tot_skip += c_counts["skipped_missing_target"]
                tot_kept += c_counts["kept_windows"]

            if x_parts:
                part_x = np.concatenate(x_parts, axis=0)
                part_y = np.concatenate(y_parts, axis=0)
                part_ys = np.concatenate(y_s_parts, axis=0)
            else:
                part_x = np.empty(
                    (0, seq_config.lookback_window, len(final_model_features)),
                    dtype=np.float32,
                )
                part_y = np.empty((0,), dtype=np.float32)
                part_ys = np.empty((0,), dtype=np.float32)

            datasets[split_name] = {
                "X": part_x,
                "y": part_y,
                "y_scaled": part_ys,
                "metadata": meta_parts,
            }
            partition_counts[split_name] = {
                "candidates": tot_cand,
                "skipped_missing_target": tot_skip,
                "valid_sequences": tot_kept,
            }

    # 3. Comprehensive Shape & Numeric Integrity Assertions
    for split_name, ds in datasets.items():
        x_mat = ds["X"]
        y_vec = ds["y"]
        y_s_vec = ds["y_scaled"]
        meta_list = ds["metadata"]

        assert x_mat.ndim == 3, f"{split_name} X must be 3D, got {x_mat.ndim}"
        assert x_mat.shape[1] == seq_config.lookback_window, (
            f"{split_name} X sequence length must be {seq_config.lookback_window}"
        )
        assert x_mat.shape[2] == len(final_model_features), (
            f"{split_name} feature dim mismatch: "
            f"{x_mat.shape[2]} vs {len(final_model_features)}"
        )
        assert len(x_mat) == len(y_vec) == len(y_s_vec) == len(meta_list), (
            f"{split_name} length mismatch: X={len(x_mat)}, "
            f"y={len(y_vec)}, meta={len(meta_list)}"
        )

        assert not np.isnan(x_mat).any(), (
            f"NaN values detected in {split_name} X tensor!"
        )
        assert not np.isinf(x_mat).any(), f"Infinite values in {split_name} X tensor!"
        assert not np.isnan(y_vec).any(), (
            f"NaN values detected in {split_name} y vector!"
        )
        assert not np.isnan(y_s_vec).any(), f"NaN values in {split_name} y_scaled!"

        # Assert all kept samples have authentic target validity
        for sample_meta in meta_list:
            assert sample_meta["target_validity"] is True, (
                f"Invalid target leaked into {split_name}: {sample_meta}"
            )

    logger.info("All tensor dimension and numeric integrity assertions passed.")

    # 4. Save sequence datasets (.pt and/or .npz)
    out_dir = Path(seq_config.output_sequences_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_dataset_files: Dict[str, str] = {}

    for split_name, ds in datasets.items():
        # Save PyTorch format (.pt)
        if seq_config.save_format in ("pt", "both"):
            pt_path = out_dir / f"{split_name}_sequences.pt"
            torch_bundle = {
                "X": torch.from_numpy(ds["X"]),
                "y": torch.from_numpy(ds["y"]),
                "y_scaled": torch.from_numpy(ds["y_scaled"]),
                "feature_names": final_model_features,
                "metadata": ds["metadata"],
                "split": split_name,
                "tensor_shape": list(ds["X"].shape),
            }
            torch.save(torch_bundle, pt_path)
            saved_dataset_files[f"{split_name}_pt"] = str(pt_path).replace("\\", "/")

        # Save NumPy compressed archive format (.npz)
        if seq_config.save_format in ("npz", "both"):
            npz_path = out_dir / f"{split_name}_sequences.npz"
            np.savez_compressed(
                npz_path,
                X=ds["X"],
                y=ds["y"],
                y_scaled=ds["y_scaled"],
            )
            saved_dataset_files[f"{split_name}_npz"] = str(npz_path).replace("\\", "/")

    # 5. Compile metadata report
    metadata = {
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "lookback_window": seq_config.lookback_window,
        "forecast_horizon": seq_config.forecast_horizon,
        "boundary_mode": seq_config.boundary_mode,
        "filter_missing_targets": seq_config.filter_missing_targets,
        "feature_count_per_timestep": len(final_model_features),
        "feature_names": final_model_features,
        "feature_categories": feature_categories,
        "tensor_shapes": {
            "train": {
                "X": list(datasets["train"]["X"].shape),
                "y": list(datasets["train"]["y"].shape),
            },
            "val": {
                "X": list(datasets["val"]["X"].shape),
                "y": list(datasets["val"]["y"].shape),
            },
            "test": {
                "X": list(datasets["test"]["X"].shape),
                "y": list(datasets["test"]["y"].shape),
            },
        },
        "target_filtering_summary": partition_counts,
        "saved_files": saved_dataset_files,
    }

    artifacts = SequenceArtifacts(
        feature_scaler=feature_scaler,
        feature_names=final_model_features,
        feature_categories=feature_categories,
        metadata=metadata,
        config=seq_config,
    )

    saved_arts = artifacts.save(
        artifacts_dir=seq_config.artifacts_dir,
        metadata_dir=seq_config.metadata_dir,
    )
    metadata["saved_artifacts"] = saved_arts
    logger.info(
        f"Sequence generation complete. Saved artifacts: {list(saved_arts.keys())}"
    )

    return datasets, artifacts, metadata

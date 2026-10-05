"""Configuration dataclass for AERIS 30-day sequence dataset generation."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from backend.app.data.schema import CITY_COLUMN, DATE_COLUMN, TARGET_COLUMN


@dataclass
class SequenceConfig:
    """Configuration parameters for time-series lookback window tensor construction."""

    # Lookback window and forecast horizon (in days)
    lookback_window: int = 30
    forecast_horizon: int = 1

    # Boundary handling policy across chronological partitions:
    # "continuous_history": Full chronological history within each city is used to
    # construct 30-day input windows for validation/test targets without losing early
    # partition samples. No future data is ever accessed.
    # "isolated_partition": Partitions are sliced in isolation (discards early days).
    boundary_mode: str = "continuous_history"

    # Target validity filtering:
    # When True, samples are kept IF AND ONLY IF the ground-truth target at (t+horizon)
    # is an authentic CPCB observation (aqi_target_valid == True).
    filter_missing_targets: bool = True

    # Column identifiers
    city_column: str = CITY_COLUMN
    date_column: str = DATE_COLUMN
    target_column: str = TARGET_COLUMN
    scaled_target_column: str = "scaled_AQI"
    target_validity_column: str = "aqi_target_valid"

    # Input & Output paths
    input_train_parquet: str = "data/processed/train_city_day.parquet"
    input_val_parquet: str = "data/processed/val_city_day.parquet"
    input_test_parquet: str = "data/processed/test_city_day.parquet"
    output_sequences_dir: str = "data/processed/sequences"
    artifacts_dir: str = "data/artifacts/sequences"
    metadata_dir: str = "data/metadata"

    # Save formats: "pt", "npz", or "both"
    save_format: str = "both"

    # Optional explicit list of model features
    model_feature_names: Optional[List[str]] = None

    def resolve_paths(self, base_dir: Path) -> "SequenceConfig":
        """Resolve relative paths against a base repository directory."""
        return SequenceConfig(
            lookback_window=self.lookback_window,
            forecast_horizon=self.forecast_horizon,
            boundary_mode=self.boundary_mode,
            filter_missing_targets=self.filter_missing_targets,
            city_column=self.city_column,
            date_column=self.date_column,
            target_column=self.target_column,
            scaled_target_column=self.scaled_target_column,
            target_validity_column=self.target_validity_column,
            input_train_parquet=str(base_dir / self.input_train_parquet),
            input_val_parquet=str(base_dir / self.input_val_parquet),
            input_test_parquet=str(base_dir / self.input_test_parquet),
            output_sequences_dir=str(base_dir / self.output_sequences_dir),
            artifacts_dir=str(base_dir / self.artifacts_dir),
            metadata_dir=str(base_dir / self.metadata_dir),
            save_format=self.save_format,
            model_feature_names=self.model_feature_names,
        )

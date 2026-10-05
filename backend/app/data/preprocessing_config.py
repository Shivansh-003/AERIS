"""Configuration dataclass for the AERIS data preprocessing pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from backend.app.data.schema import (
    BUCKET_COLUMN,
    CITY_COLUMN,
    DATE_COLUMN,
    POLLUTANT_COLUMNS,
    TARGET_COLUMN,
)


@dataclass
class PreprocessingConfig:
    """Configuration parameters for data cleaning, splitting, and scaling."""

    # Input & Output file paths
    input_parquet_path: str = "data/processed/cleaned_city_day.parquet"
    output_processed_dir: str = "data/processed"
    artifacts_dir: str = "data/artifacts/preprocessing"
    scalers_backup_dir: str = "artifacts/scalers"
    metadata_dir: str = "data/metadata"

    # Train / Validation / Test split ratios (chronological, per city)
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Feature definitions
    feature_columns: List[str] = field(default_factory=lambda: list(POLLUTANT_COLUMNS))
    target_column: str = TARGET_COLUMN
    target_validity_column: str = "aqi_target_valid"
    id_columns: List[str] = field(default_factory=lambda: [CITY_COLUMN, DATE_COLUMN])
    derived_category_columns: List[str] = field(default_factory=lambda: [BUCKET_COLUMN])

    # Scaling & Imputation
    scaler_type: str = "StandardScaler"
    scale_target: bool = True
    imputation_strategy: str = "city_forward_fill_then_train_median"
    pipeline_version: str = "1.0.0"

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        total_ratio = round(self.train_ratio + self.val_ratio + self.test_ratio, 6)
        if total_ratio != 1.0:
            raise ValueError(
                f"Split ratios must sum to 1.0 (got {total_ratio} from "
                f"{self.train_ratio} + {self.val_ratio} + {self.test_ratio})"
            )
        if self.train_ratio <= 0 or self.val_ratio <= 0 or self.test_ratio <= 0:
            raise ValueError("All split ratios must be strictly positive (> 0).")

    @property
    def all_numeric_model_columns(self) -> List[str]:
        """Return list of all numeric columns to be imputed and scaled."""
        cols = list(self.feature_columns)
        if self.target_column not in cols:
            cols.append(self.target_column)
        return cols

    def resolve_paths(self, base_dir: Path) -> "PreprocessingConfig":
        """Resolve relative paths against a base repository directory."""
        return PreprocessingConfig(
            input_parquet_path=str(base_dir / self.input_parquet_path),
            output_processed_dir=str(base_dir / self.output_processed_dir),
            artifacts_dir=str(base_dir / self.artifacts_dir),
            scalers_backup_dir=str(base_dir / self.scalers_backup_dir),
            metadata_dir=str(base_dir / self.metadata_dir),
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
            feature_columns=self.feature_columns,
            target_column=self.target_column,
            target_validity_column=self.target_validity_column,
            id_columns=self.id_columns,
            derived_category_columns=self.derived_category_columns,
            scaler_type=self.scaler_type,
            scale_target=self.scale_target,
            imputation_strategy=self.imputation_strategy,
            pipeline_version=self.pipeline_version,
        )

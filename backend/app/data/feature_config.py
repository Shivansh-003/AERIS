"""Configuration dataclass for the AERIS feature engineering pipeline."""

from dataclasses import dataclass, field
from typing import Dict, List

from backend.app.data.schema import (
    BUCKET_COLUMN,
    CITY_COLUMN,
    DATE_COLUMN,
    POLLUTANT_COLUMNS,
    TARGET_COLUMN,
)


@dataclass
class FeatureConfig:
    """Configuration parameters for time-series feature engineering."""

    # Base identifier and target definitions
    city_column: str = CITY_COLUMN
    date_column: str = DATE_COLUMN
    target_column: str = TARGET_COLUMN
    target_validity_column: str = "aqi_target_valid"
    bucket_column: str = BUCKET_COLUMN

    # Base pollutant features to incorporate (standardized inputs)
    base_pollutant_columns: List[str] = field(
        default_factory=lambda: list(POLLUTANT_COLUMNS)
    )

    # Backward lag intervals (in days)
    lag_columns: List[str] = field(
        default_factory=lambda: ["PM2.5", "PM10", "NO2", "AQI"]
    )
    lag_intervals: List[int] = field(default_factory=lambda: [1, 3, 7])

    # Backward rolling window parameters (in days)
    rolling_columns: List[str] = field(
        default_factory=lambda: ["PM2.5", "PM10", "NO2", "AQI"]
    )
    rolling_mean_windows: List[int] = field(default_factory=lambda: [3, 7, 14])
    rolling_std_windows: List[int] = field(default_factory=lambda: [7, 14])
    rolling_min_max_windows: List[int] = field(default_factory=lambda: [7])

    # Backward Rate of Change (ROC) parameters
    roc_columns: List[str] = field(
        default_factory=lambda: ["PM2.5", "PM10", "NO2", "CO", "AQI"]
    )
    roc_intervals: List[int] = field(default_factory=lambda: [1, 3, 7])
    roc_epsilon: float = 1e-4
    roc_clip_min: float = -5.0
    roc_clip_max: float = 5.0

    # Backward Volatility (Coefficient of Variation) parameters
    volatility_columns: List[str] = field(
        default_factory=lambda: ["PM2.5", "PM10", "AQI"]
    )
    volatility_windows: List[int] = field(default_factory=lambda: [7, 14])
    volatility_epsilon: float = 1e-4
    volatility_clip_max: float = 5.0

    # Indian Meteorological Seasons Mapping:
    # 0: Winter (Dec, Jan, Feb)
    # 1: Summer / Pre-Monsoon (Mar, Apr, May)
    # 2: Monsoon (Jun, Jul, Aug, Sep)
    # 3: Post-Monsoon / Autumn (Oct, Nov)
    month_to_season: Dict[int, int] = field(
        default_factory=lambda: {
            12: 0,
            1: 0,
            2: 0,
            3: 1,
            4: 1,
            5: 1,
            6: 2,
            7: 2,
            8: 2,
            9: 2,
            10: 3,
            11: 3,
        }
    )

    # Scaling policy for newly engineered continuous features
    scale_engineered_features: bool = True

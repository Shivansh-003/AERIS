"""Dataset schema definitions and validation models for AERIS."""

from dataclasses import dataclass, field
from typing import Any, Dict, List

# Core schema column definitions
CITY_COLUMN: str = "City"
DATE_COLUMN: str = "Date"
TARGET_COLUMN: str = "AQI"
BUCKET_COLUMN: str = "AQI_Bucket"

POLLUTANT_COLUMNS: List[str] = [
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
]

NUMERIC_COLUMNS: List[str] = POLLUTANT_COLUMNS + [TARGET_COLUMN]

REQUIRED_COLUMNS: List[str] = (
    [CITY_COLUMN, DATE_COLUMN] + POLLUTANT_COLUMNS + [TARGET_COLUMN, BUCKET_COLUMN]
)

EXPECTED_AQI_BUCKETS: List[str] = [
    "Good",
    "Satisfactory",
    "Moderate",
    "Poor",
    "Very Poor",
    "Severe",
]

# Standard dtype mappings for pandas dataframe representation
TARGET_DTYPES: Dict[str, str] = {
    CITY_COLUMN: "string",
    DATE_COLUMN: "datetime64[ns]",
    BUCKET_COLUMN: "string",
}
for col in NUMERIC_COLUMNS:
    TARGET_DTYPES[col] = "float64"


@dataclass
class ValidationReport:
    """Structured report containing data validation diagnostics."""

    is_valid: bool = True
    row_count: int = 0
    column_count: int = 0
    missing_columns: List[str] = field(default_factory=list)
    unexpected_columns: List[str] = field(default_factory=list)
    duplicate_rows_count: int = 0
    duplicate_city_date_count: int = 0
    invalid_dates_count: int = 0
    invalid_numeric_values: Dict[str, int] = field(default_factory=dict)
    negative_values_count: Dict[str, int] = field(default_factory=dict)
    invalid_buckets_count: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    def add_error(self, message: str) -> None:
        """Record a validation error and set validity flag to False."""
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str) -> None:
        """Record a validation warning without necessarily invalidating."""
        self.warnings.append(message)

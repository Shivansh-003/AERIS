"""AERIS Data ingestion, validation, profiling, EDA, and preprocessing modules."""

from backend.app.data.eda import (
    compute_aqi_summary,
    compute_city_statistics,
    compute_correlations,
    compute_dataset_overview,
    compute_duplicate_analysis,
    compute_missing_analysis,
    compute_outlier_analysis,
    compute_pollutant_summaries,
    compute_temporal_statistics,
    generate_eda_report,
    plot_all_figures,
    run_eda_pipeline,
    save_eda_report,
)
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
from backend.app.data.ingestion import (
    generate_dataset_profile,
    load_raw_csv,
    run_ingestion_pipeline,
    save_dataset_profile,
    save_processed_parquet,
    validate_and_normalize,
)
from backend.app.data.preprocessing import (
    fit_and_apply_imputation,
    fit_and_apply_scalers,
    run_preprocessing_pipeline,
    split_city_chronological,
    validate_input_data,
)
from backend.app.data.preprocessing_artifacts import PreprocessingArtifacts
from backend.app.data.preprocessing_config import PreprocessingConfig
from backend.app.data.schema import (
    BUCKET_COLUMN,
    CITY_COLUMN,
    DATE_COLUMN,
    EXPECTED_AQI_BUCKETS,
    NUMERIC_COLUMNS,
    POLLUTANT_COLUMNS,
    REQUIRED_COLUMNS,
    TARGET_COLUMN,
    TARGET_DTYPES,
    ValidationReport,
)
from backend.app.data.sequence_artifacts import SequenceArtifacts
from backend.app.data.sequence_config import SequenceConfig
from backend.app.data.sequences import (
    build_sequences_pipeline,
    extract_city_sequences,
)

__all__ = [
    "CITY_COLUMN",
    "DATE_COLUMN",
    "TARGET_COLUMN",
    "BUCKET_COLUMN",
    "POLLUTANT_COLUMNS",
    "NUMERIC_COLUMNS",
    "REQUIRED_COLUMNS",
    "EXPECTED_AQI_BUCKETS",
    "TARGET_DTYPES",
    "ValidationReport",
    "load_raw_csv",
    "validate_and_normalize",
    "generate_dataset_profile",
    "save_dataset_profile",
    "save_processed_parquet",
    "run_ingestion_pipeline",
    "compute_dataset_overview",
    "compute_missing_analysis",
    "compute_duplicate_analysis",
    "compute_outlier_analysis",
    "compute_city_statistics",
    "compute_pollutant_summaries",
    "compute_aqi_summary",
    "compute_temporal_statistics",
    "compute_correlations",
    "generate_eda_report",
    "save_eda_report",
    "plot_all_figures",
    "run_eda_pipeline",
    "PreprocessingConfig",
    "PreprocessingArtifacts",
    "validate_input_data",
    "split_city_chronological",
    "fit_and_apply_imputation",
    "fit_and_apply_scalers",
    "run_preprocessing_pipeline",
    "FeatureConfig",
    "compute_calendar_features",
    "compute_lag_features",
    "compute_rolling_features",
    "compute_roc_features",
    "compute_volatility_features",
    "engineer_all_features",
    "fit_and_scale_engineered_features",
    "SequenceConfig",
    "SequenceArtifacts",
    "extract_city_sequences",
    "build_sequences_pipeline",
]

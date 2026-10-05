"""Preprocessing artifacts container and serialization manager for AERIS."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
from backend.app.data.preprocessing_config import PreprocessingConfig
from sklearn.preprocessing import StandardScaler


@dataclass
class PreprocessingArtifacts:
    """Encapsulates fitted preprocessing objects and metadata."""

    scaler: StandardScaler
    imputation_medians: Dict[str, float]
    feature_names: List[str]
    target_name: str
    feature_order: List[str]
    training_stats: Dict[str, Dict[str, float]]
    config: PreprocessingConfig
    metadata: Dict[str, Any] = field(default_factory=dict)
    target_scaler: Optional[StandardScaler] = None

    def save(
        self,
        artifacts_dir: Path | str,
        scalers_backup_dir: Optional[Path | str] = None,
        metadata_dir: Optional[Path | str] = None,
    ) -> Dict[str, Path]:
        """Serialize preprocessing artifacts to disk.

        Args:
            artifacts_dir: Directory to save primary preprocessing artifacts.
            scalers_backup_dir: Optional secondary backup dir for scaler.
            metadata_dir: Optional destination dir for metadata JSON.

        Returns:
            Dictionary of saved artifact file paths.
        """
        art_dir = Path(artifacts_dir)
        art_dir.mkdir(parents=True, exist_ok=True)
        saved_paths: Dict[str, Path] = {}

        # 1. Save fitted StandardScaler
        scaler_path = art_dir / "scaler.joblib"
        joblib.dump(
            {
                "scaler": self.scaler,
                "target_scaler": self.target_scaler,
                "feature_names": self.feature_names,
                "target_name": self.target_name,
                "feature_order": self.feature_order,
                "training_stats": self.training_stats,
                "version": self.config.pipeline_version,
            },
            scaler_path,
        )
        saved_paths["scaler"] = scaler_path

        # Also backup to artifacts/scalers if requested
        if scalers_backup_dir:
            backup_dir = Path(scalers_backup_dir)
            backup_dir.mkdir(parents=True, exist_ok=True)
            backup_scaler_path = backup_dir / "scaler.joblib"
            joblib.dump(self.scaler, backup_scaler_path)
            saved_paths["scaler_backup"] = backup_scaler_path

        # 2. Save Imputation statistics
        imputation_payload = {
            "strategy": self.config.imputation_strategy,
            "pipeline_version": self.config.pipeline_version,
            "learned_train_medians": self.imputation_medians,
            "feature_order": self.feature_order,
        }
        imputation_path = art_dir / "imputation_statistics.json"
        with open(imputation_path, "w", encoding="utf-8") as f:
            json.dump(imputation_payload, f, indent=2)
        saved_paths["imputation_statistics"] = imputation_path

        # 3. Save comprehensive metadata
        metadata_payload = {
            "metadata_version": self.config.pipeline_version,
            "config": asdict(self.config),
            "feature_order": self.feature_order,
            "imputation_medians": self.imputation_medians,
            "training_statistics": self.training_stats,
            **self.metadata,
        }
        metadata_path = art_dir / "preprocessing_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata_payload, f, indent=2)
        saved_paths["metadata"] = metadata_path

        if metadata_dir:
            meta_dir = Path(metadata_dir)
            meta_dir.mkdir(parents=True, exist_ok=True)
            meta_dest = meta_dir / "preprocessing_metadata.json"
            with open(meta_dest, "w", encoding="utf-8") as f:
                json.dump(metadata_payload, f, indent=2)
            saved_paths["metadata_backup"] = meta_dest

        return saved_paths

    @classmethod
    def load(cls, artifacts_dir: Path | str) -> "PreprocessingArtifacts":
        """Load serialized preprocessing artifacts from disk.

        Args:
            artifacts_dir: Directory containing serialized artifacts.

        Returns:
            Instantiated PreprocessingArtifacts object.
        """
        art_dir = Path(artifacts_dir)
        scaler_file = art_dir / "scaler.joblib"
        metadata_file = art_dir / "preprocessing_metadata.json"

        if not scaler_file.exists():
            raise FileNotFoundError(f"Scaler file not found at: {scaler_file}")
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found at: {metadata_file}")

        scaler_payload = joblib.load(scaler_file)
        with open(metadata_file, "r", encoding="utf-8") as f:
            meta_payload = json.load(f)

        raw_cfg = meta_payload.get("config", {})
        config = PreprocessingConfig(**raw_cfg)

        return cls(
            scaler=scaler_payload["scaler"],
            target_scaler=scaler_payload.get("target_scaler"),
            imputation_medians=meta_payload.get("imputation_medians", {}),
            feature_names=scaler_payload.get("feature_names", config.feature_columns),
            target_name=scaler_payload.get("target_name", config.target_column),
            feature_order=scaler_payload.get(
                "feature_order", config.all_numeric_model_columns
            ),
            training_stats=scaler_payload.get("training_stats", {}),
            config=config,
            metadata=meta_payload,
        )

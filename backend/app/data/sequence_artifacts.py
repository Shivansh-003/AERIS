"""Artifact management for AERIS feature scalers, sequence tensors, and metadata."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
from backend.app.data.sequence_config import SequenceConfig
from sklearn.preprocessing import StandardScaler


@dataclass
class SequenceArtifacts:
    """Container for serialized feature scalers, feature lists, and metadata."""

    feature_scaler: Optional[StandardScaler]
    feature_names: List[str]
    feature_categories: Dict[str, List[str]]
    metadata: Dict[str, Any]
    config: SequenceConfig

    def save(
        self,
        artifacts_dir: str = "data/artifacts/sequences",
        metadata_dir: str = "data/metadata",
    ) -> Dict[str, str]:
        """Save feature scalers, metadata records, and feature inventories."""
        art_path = Path(artifacts_dir)
        meta_path = Path(metadata_dir)

        art_path.mkdir(parents=True, exist_ok=True)
        meta_path.mkdir(parents=True, exist_ok=True)

        saved_files: Dict[str, str] = {}

        # 1. Save feature scaler
        if self.feature_scaler is not None:
            scaler_file = art_path / "feature_scaler.joblib"
            joblib.dump(self.feature_scaler, scaler_file)
            saved_files["feature_scaler"] = str(scaler_file).replace("\\", "/")

        # 2. Save feature list and categories
        feature_info = {
            "feature_names": self.feature_names,
            "feature_count": len(self.feature_names),
            "feature_categories": self.feature_categories,
        }
        feat_file = art_path / "sequence_features.json"
        with open(feat_file, "w", encoding="utf-8") as f:
            json.dump(feature_info, f, indent=2)
        saved_files["feature_info"] = str(feat_file).replace("\\", "/")

        # 3. Save sequence metadata
        meta_record = {
            "metadata": self.metadata,
            "feature_count": len(self.feature_names),
            "feature_names": self.feature_names,
            "config": asdict(self.config),
        }
        meta_file = art_path / "sequence_metadata.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta_record, f, indent=2)
        saved_files["metadata"] = str(meta_file).replace("\\", "/")

        # Backup metadata to data/metadata/
        meta_backup = meta_path / "sequence_metadata.json"
        with open(meta_backup, "w", encoding="utf-8") as f:
            json.dump(meta_record, f, indent=2)
        saved_files["metadata_backup"] = str(meta_backup).replace("\\", "/")

        return saved_files

    @classmethod
    def load(
        cls, artifacts_dir: str = "data/artifacts/sequences"
    ) -> "SequenceArtifacts":
        """Load serialized feature scaler and metadata from disk."""
        art_path = Path(artifacts_dir)

        scaler_file = art_path / "feature_scaler.joblib"
        feature_scaler = joblib.load(scaler_file) if scaler_file.exists() else None

        meta_file = art_path / "sequence_metadata.json"
        if not meta_file.exists():
            raise FileNotFoundError(f"Sequence metadata not found at: {meta_file}")

        with open(meta_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        config_dict = data.get("config", {})
        config = SequenceConfig(**config_dict)

        return cls(
            feature_scaler=feature_scaler,
            feature_names=data.get("feature_names", []),
            feature_categories=data.get("metadata", {}).get("feature_categories", {}),
            metadata=data.get("metadata", {}),
            config=config,
        )

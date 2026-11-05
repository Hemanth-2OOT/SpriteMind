"""Strict Data Classification Registry for SpriteMind Stage 5.5.

Defines the non-negotiable boundary between protected core assets
(datasets, checkpoints, research) and disposable/user-generated assets.
"""
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Set, Optional


class DataClassification(str, Enum):
    """Classification tiers defining lifecycle permissions and immutability."""
    PROTECTED_TRAINING = "protected_training"
    PROTECTED_MODEL = "protected_model"
    PROTECTED_RESEARCH = "protected_research"
    USER_GENERATED = "user_generated"
    TEMPORARY_REGENERABLE = "temporary_regenerable"


# Deletion permissions: Only USER_GENERATED and TEMPORARY_REGENERABLE may be trashed/deleted.
DELETABLE_CLASSIFICATIONS: Set[DataClassification] = {
    DataClassification.USER_GENERATED,
    DataClassification.TEMPORARY_REGENERABLE,
}

IMMUTABLE_CLASSIFICATIONS: Set[DataClassification] = {
    DataClassification.PROTECTED_TRAINING,
    DataClassification.PROTECTED_MODEL,
    DataClassification.PROTECTED_RESEARCH,
}


class ProtectedResourceException(Exception):
    """Raised when an attempt is made to delete, modify, or corrupt a protected resource."""
    pass


class StorageClassifier:
    """Evaluates filesystem paths and metadata to determine strict classification."""

    @classmethod
    def _normalize(cls, path_str: str) -> Optional[str]:
        """Returns a repo-relative normalized posix path, or None if it escapes the repo."""
        root = Path.cwd().resolve()
        raw = Path(path_str)
        try:
            resolved = (raw if raw.is_absolute() else root / raw).resolve(strict=False)
        except OSError:
            return None
        try:
            return resolved.relative_to(root).as_posix()
        except ValueError:
            return None

    @classmethod
    def classify_path(cls, path_str: str) -> DataClassification:
        """Classifies a path based on canonical directory structures and rules."""
        normalized = cls._normalize(path_str)
        if normalized is None:
            # Anything outside the project root is never deletable.
            return DataClassification.PROTECTED_RESEARCH
        p = normalized

        # 1. Protected Model Data
        if p.startswith("models/checkpoints") or p.startswith("models/") or p.endswith(".pt"):
            return DataClassification.PROTECTED_MODEL

        # 2. Protected Training Data
        if (
            p.startswith("data/datasets")
            or p.startswith("data/temporal_dataset")
            or p.startswith("data/composition_dataset")
            or "records.jsonl" in p
            or "train_sequences.jsonl" in p
            or "val_sequences.jsonl" in p
        ):
            return DataClassification.PROTECTED_TRAINING

        # 3. Protected Research and Benchmark Evidence
        if (
            p.startswith("docs/research")
            or p.startswith("data/benchmarks")
            or "temporal_metrics.json" in p
            or "critic_metrics.json" in p
            or "composition_metrics.json" in p
            or "training_history.json" in p
        ):
            return DataClassification.PROTECTED_RESEARCH

        # 4. Temporary and Regenerable Data
        if (
            "__pycache__" in p
            or ".pytest_cache" in p
            or p.startswith("data/cache")
            or p.startswith("data/temp")
            or p.startswith("data/trash")
            or p.endswith(".pyc")
            or p.endswith(".tmp")
            or p.endswith(".log")
        ):
            return DataClassification.TEMPORARY_REGENERABLE

        # 5. User-Generated Assets
        if (
            p.startswith("data/assets")
            or p.startswith("data/asset_packs")
            or p.startswith("data/exports")
            or p.startswith("data/atlases")
            or p.startswith("data/renders")
        ):
            return DataClassification.USER_GENERATED

        # Fail closed: unknown paths are never deletable.
        return DataClassification.PROTECTED_RESEARCH

    @classmethod
    def is_deletable(cls, path_str: str) -> bool:
        """Returns True only if the resource is in a deletable tier."""
        classification = cls.classify_path(path_str)
        return classification in DELETABLE_CLASSIFICATIONS

    @classmethod
    def assert_deletable(cls, path_str: str) -> bool:
        """Raises ProtectedResourceException if path is protected. Returns True if deletable."""
        classification = cls.classify_path(path_str)
        if classification in IMMUTABLE_CLASSIFICATIONS:
            raise ProtectedResourceException(
                f"ILLEGAL OPERATION: '{path_str}' is classified as '{classification.value}' "
                f"and is strictly protected. Models, datasets, and research evidence cannot be deleted."
            )
        return True

"""Stable filesystem anchors for the shipped assets.

Checkpoint paths used to be resolved against the current working directory, so
starting the API from ``backend/`` made every trained checkpoint look missing
and generation silently degraded to an untrained model. Anchoring on the
repository root makes the runtime independent of the launch directory.
"""
import os
from pathlib import Path

DEFAULT_CHECKPOINTS = "models/checkpoints"
DEFAULT_DATA = "data"


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _anchor(candidate: str, default: str, root: Path) -> Path:
    path = Path(candidate)
    if path.is_absolute():
        return path
    if path != Path(default):
        return path
    rooted = root / default
    return rooted if rooted.exists() else path


def resolve_checkpoints_dir(path: "str | Path" = DEFAULT_CHECKPOINTS) -> Path:
    env = os.environ.get("SPRITEMIND_CHECKPOINTS_DIR")
    if env:
        return Path(env).expanduser()
    return _anchor(str(path), DEFAULT_CHECKPOINTS, project_root())


def resolve_data_dir(path: "str | Path" = DEFAULT_DATA) -> Path:
    env = os.environ.get("SPRITEMIND_DATA_DIR")
    if env:
        return Path(env).expanduser()
    return _anchor(str(path), DEFAULT_DATA, project_root())

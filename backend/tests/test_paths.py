"""Checkpoint and data paths must not depend on the launch directory."""
import os
from pathlib import Path

from spritemind.paths import project_root, resolve_checkpoints_dir, resolve_data_dir


def test_default_checkpoints_dir_is_repository_anchored(monkeypatch):
    monkeypatch.chdir(Path(__file__).resolve().parents[2] / "backend")
    resolved = resolve_checkpoints_dir()
    assert resolved.is_absolute()
    assert resolved == project_root() / "models/checkpoints"
    assert resolved.exists()


def test_checkpoints_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("SPRITEMIND_CHECKPOINTS_DIR", str(tmp_path))
    assert resolve_checkpoints_dir() == tmp_path


def test_explicit_relative_checkpoints_dir_is_kept(monkeypatch, tmp_path):
    monkeypatch.delenv("SPRITEMIND_CHECKPOINTS_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    assert resolve_checkpoints_dir("my/ckpts") == Path("my/ckpts")


def test_data_dir_is_repository_anchored(monkeypatch):
    monkeypatch.delenv("SPRITEMIND_DATA_DIR", raising=False)
    monkeypatch.chdir(project_root() / "backend")
    assert resolve_data_dir() == project_root() / "data"


def test_temporal_checkpoint_resolves_from_backend_cwd(monkeypatch):
    from spritemind.ml.temporal import TemporalAnimator

    monkeypatch.chdir(project_root() / "backend")
    TemporalAnimator._cached_model = None
    TemporalAnimator._cached_checkpoint = None
    _, loaded = TemporalAnimator.get_or_load_model()
    assert loaded is True
    assert os.path.basename(TemporalAnimator.last_load_status["checkpoint"]) == (
        "temporal_svan_best.pt"
    )

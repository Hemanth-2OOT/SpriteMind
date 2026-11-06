import json

import pytest
from fastapi.testclient import TestClient

from spritemind.ml import checkpoint_manifest as cm
from spritemind.server.app import app


@pytest.fixture()
def checkpoint_dir(tmp_path):
    payload = tmp_path / "demo.pt"
    payload.write_bytes(b"weights-v1")
    return tmp_path


def test_record_creates_manifest_and_entry(checkpoint_dir):
    entry = cm.describe_file(checkpoint_dir / "demo.pt", parameter_count=42)
    path = cm.record(checkpoint_dir, entry)

    assert path.exists()
    manifest = json.loads(path.read_text())
    assert manifest["checkpoints"]["demo.pt"]["parameter_count"] == 42
    assert cm.describe(checkpoint_dir, "demo.pt")["bytes"] == 10


def test_verify_reports_missing_tampered_and_valid(checkpoint_dir):
    missing = cm.verify(checkpoint_dir, "nope.pt")
    assert missing["ok"] is False and missing["reason"] == "missing file"

    cm.record(checkpoint_dir, cm.describe_file(checkpoint_dir / "demo.pt"))
    assert cm.verify(checkpoint_dir, "demo.pt")["reason"] == "verified"

    (checkpoint_dir / "demo.pt").write_bytes(b"weights-v2-longer")
    tampered = cm.verify(checkpoint_dir, "demo.pt")
    assert tampered["ok"] is False and tampered["reason"] in {"size mismatch", "sha256 mismatch"}


def test_record_preserves_other_entries(checkpoint_dir):
    (checkpoint_dir / "other.pt").write_bytes(b"other")
    cm.record(checkpoint_dir, cm.describe_file(checkpoint_dir / "demo.pt"))
    cm.record(checkpoint_dir, cm.describe_file(checkpoint_dir / "other.pt"))

    names = [entry["file"] for entry in cm.list_entries(checkpoint_dir)]
    assert names == ["demo.pt", "other.pt"]


def test_unrecorded_checkpoint_is_still_accepted(checkpoint_dir):
    result = cm.verify(checkpoint_dir, "demo.pt")

    assert result["ok"] is True
    assert result["recorded"] is False


def test_models_endpoint_reports_integrity():
    response = TestClient(app).get("/api/models")

    assert response.status_code == 200
    body = response.json()
    assert "checkpoints" in body
    assert "temporal_checkpoint_status" in body

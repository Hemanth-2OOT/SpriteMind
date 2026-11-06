import os

import pytest
from fastapi.testclient import TestClient

from spritemind.server.app import app

client = TestClient(app)


@pytest.fixture()
def api_key(monkeypatch):
    monkeypatch.setenv("SPRITEMIND_API_KEY", "s3cret-key")
    return "s3cret-key"


def test_open_access_when_no_key_configured(monkeypatch):
    monkeypatch.delenv("SPRITEMIND_API_KEY", raising=False)

    assert client.get("/api/models").status_code == 200


def test_key_required_once_configured(api_key):
    assert client.get("/api/models").status_code == 401


def test_health_stays_open(api_key):
    assert client.get("/api/health").status_code == 200


def test_header_key_is_accepted(api_key):
    response = client.get("/api/models", headers={"X-API-Key": api_key})

    assert response.status_code == 200


def test_bearer_token_is_accepted(api_key):
    response = client.get("/api/models", headers={"Authorization": f"Bearer {api_key}"})

    assert response.status_code == 200


def test_wrong_key_is_rejected(api_key):
    response = client.get("/api/models", headers={"X-API-Key": "nope"})

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or missing API key"


def test_blank_env_value_disables_the_guard(monkeypatch):
    monkeypatch.setenv("SPRITEMIND_API_KEY", "   ")

    assert os.environ["SPRITEMIND_API_KEY"].strip() == ""
    assert client.get("/api/models").status_code == 200

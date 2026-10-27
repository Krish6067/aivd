"""tests/test_health.py — VOXSHIELD Backend

Tests for GET /api/health and GET /api/model/status.
"""

import pytest
import pytest_asyncio


pytestmark = pytest.mark.asyncio


async def test_health_returns_200(client):
    r = await client.get("/api/health")
    assert r.status_code == 200


async def test_health_body_schema(client):
    r = await client.get("/api/health")
    body = r.json()
    assert body["status"] == "ok"
    assert "version" in body
    assert "timestamp" in body


async def test_model_status_returns_200(client):
    r = await client.get("/api/model/status")
    assert r.status_code == 200


async def test_model_status_ready(client):
    r = await client.get("/api/model/status")
    body = r.json()
    assert body["readiness"] == "ready"
    assert body["model_name"] == "AASIST-MOCK"
    assert body["supported_sample_rate"] == 16000


async def test_model_status_not_ready(client_no_model):
    r = await client_no_model.get("/api/model/status")
    body = r.json()
    assert body["readiness"] == "not_ready"
    assert body["error_detail"] is not None

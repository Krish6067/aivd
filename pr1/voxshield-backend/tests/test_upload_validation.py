"""tests/test_upload_validation.py — VOXSHIELD Backend

Tests for all input validation guards on POST /api/analyze
(no real model weights required — uses mock service).
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.asyncio


# ─── Helpers ──────────────────────────────────────────────────────────────────


async def _post_audio(client, data: bytes, content_type: str = "audio/wav"):
    return await client.post(
        "/api/analyze",
        files={"file": ("test.wav", data, content_type)},
    )


# ─── Missing file ─────────────────────────────────────────────────────────────


async def test_missing_file_returns_422(client):
    r = await client.post("/api/analyze")
    assert r.status_code == 422


# ─── Empty file ───────────────────────────────────────────────────────────────


async def test_empty_file_returns_400(client):
    r = await _post_audio(client, b"")
    assert r.status_code == 400


# ─── Corrupt / non-audio bytes ────────────────────────────────────────────────


async def test_corrupt_bytes_returns_415(client, make_corrupt_bytes):
    r = await _post_audio(client, make_corrupt_bytes)
    assert r.status_code == 415


# ─── Valid WAV ────────────────────────────────────────────────────────────────


async def test_valid_wav_returns_200(client, make_wav_bytes):
    wav = make_wav_bytes(duration_seconds=4.0)
    r   = await _post_audio(client, wav)
    assert r.status_code == 200


async def test_valid_wav_response_schema(client, make_wav_bytes):
    wav  = make_wav_bytes(duration_seconds=4.0)
    r    = await _post_audio(client, wav)
    body = r.json()

    required = {
        "session_id", "classification", "raw_score", "score_type",
        "score_interpretation", "model_name", "model_version",
        "processing_latency_ms", "audio_duration_seconds",
        "sample_rate", "timestamp", "warnings",
    }
    assert required.issubset(body.keys())
    assert 0.0 <= body["raw_score"] <= 1.0
    assert body["score_type"] == "bonafide_probability"
    assert body["classification"] in {
        "likely_authentic", "potential_synthetic_speech", "inconclusive"
    }


# ─── Short audio ──────────────────────────────────────────────────────────────


async def test_too_short_audio_returns_400(client, make_wav_bytes):
    # 0.2 s — below min_audio_duration_seconds=0.5
    wav = make_wav_bytes(duration_seconds=0.2)
    r   = await _post_audio(client, wav)
    assert r.status_code == 400


async def test_short_but_valid_audio_warns(client, make_wav_bytes):
    # 1 s — above minimum but below recommended 4 s
    wav  = make_wav_bytes(duration_seconds=1.0)
    r    = await _post_audio(client, wav)
    assert r.status_code == 200
    body = r.json()
    codes = [w["code"] for w in body["warnings"]]
    assert any("SHORT" in c or "AUDIO" in c for c in codes)


# ─── Silent audio ─────────────────────────────────────────────────────────────


async def test_silent_audio_warns(client, make_wav_bytes):
    wav  = make_wav_bytes(duration_seconds=4.0, silent=True)
    r    = await _post_audio(client, wav)
    assert r.status_code == 200
    body = r.json()
    codes = [w["code"] for w in body["warnings"]]
    assert "SILENT_AUDIO" in codes


# ─── Unsupported MIME type ────────────────────────────────────────────────────


async def test_unsupported_mime_returns_415(client, make_wav_bytes):
    wav = make_wav_bytes(duration_seconds=4.0)
    # Send valid WAV but with wrong MIME type
    r = await client.post(
        "/api/analyze",
        files={"file": ("test.txt", wav, "text/plain")},
    )
    assert r.status_code == 415


# ─── Model not ready ─────────────────────────────────────────────────────────


async def test_model_not_ready_returns_503(client_no_model, make_wav_bytes):
    wav = make_wav_bytes(duration_seconds=4.0)
    r = await client_no_model.post(
        "/api/analyze",
        files={"file": ("test.wav", wav, "audio/wav")},
    )
    assert r.status_code == 503


# ─── Fixtures re-exported for convenience ─────────────────────────────────────


@pytest.fixture()
def make_wav_bytes():
    from tests.conftest import make_wav_bytes as _f
    return _f


@pytest.fixture()
def make_corrupt_bytes():
    from tests.conftest import make_corrupt_bytes as _f
    return _f()

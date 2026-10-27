"""
tests/conftest.py — VOXSHIELD Backend

Shared pytest fixtures and test helpers.
"""

from __future__ import annotations

import io
import struct
import wave
import numpy as np
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.main import app


# ─── Test audio generators ────────────────────────────────────────────────────


def make_wav_bytes(
    duration_seconds: float = 4.0,
    sample_rate: int = 16000,
    frequency: float = 440.0,
    amplitude: float = 0.3,
    silent: bool = False,
) -> bytes:
    """
    Generate a synthetic WAV file in memory.

    Args:
        duration_seconds: Length of audio.
        sample_rate:      Samples per second.
        frequency:        Sine tone frequency (Hz).
        amplitude:        Peak amplitude (0–1).
        silent:           If True, all samples are zero.
    Returns:
        bytes — a valid, in-memory WAV file.
    """
    n_samples = int(duration_seconds * sample_rate)

    if silent:
        samples = np.zeros(n_samples, dtype=np.float32)
    else:
        t       = np.linspace(0, duration_seconds, n_samples, endpoint=False)
        samples = (np.sin(2 * np.pi * frequency * t) * amplitude).astype(np.float32)

    # Write to PCM-16 WAV via Python wave module
    pcm_samples = (samples * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)      # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_samples.tobytes())
    return buf.getvalue()


def make_pcm_float32_bytes(
    duration_seconds: float = 4.0,
    sample_rate: int = 16000,
    frequency: float = 440.0,
    amplitude: float = 0.3,
) -> bytes:
    """
    Generate raw PCM float32 bytes for WebSocket streaming tests.
    Little-endian, mono, 16 kHz.
    """
    n_samples = int(duration_seconds * sample_rate)
    t         = np.linspace(0, duration_seconds, n_samples, endpoint=False)
    samples   = (np.sin(2 * np.pi * frequency * t) * amplitude).astype(np.float32)
    return samples.tobytes()  # little-endian float32


def make_corrupt_bytes() -> bytes:
    """Return bytes that are not a valid audio file."""
    return b"\x00\x01\x02\x03 This is not audio data."


def make_oversized_bytes(mb: float) -> bytes:
    """Generate a byte string of the given size (not valid audio)."""
    return b"X" * int(mb * 1024 * 1024)


# ─── Mock model service ───────────────────────────────────────────────────────


class _MockModelService:
    """
    A model service that returns a fixed score without loading real weights.
    Used for tests that exercise API plumbing without requiring AASIST weights.
    """

    def __init__(self, score: float = 0.87, ready: bool = True):
        self._score = score
        self._ready = ready

    def is_ready(self) -> bool:
        return self._ready

    def get_status(self) -> dict:
        from app.schemas.analysis import ModelReadiness
        return {
            "model_name":    "AASIST-MOCK",
            "model_version": "test",
            "readiness":     ModelReadiness.READY if self._ready else ModelReadiness.NOT_READY,
            "device":        "cpu",
            "weights_path":  "mock",
            "supported_sample_rate": 16000,
            "supported_channels":    1,
            "min_audio_duration_seconds": 0.5,
            "max_audio_duration_seconds": 60.0,
            "error_detail":  None if self._ready else "Mock: model not loaded.",
        }

    def load(self, *a, **kw) -> None:
        pass

    def shutdown(self) -> None:
        pass

    async def infer_async(self, tensor):
        if not self._ready:
            raise RuntimeError("MODEL NOT READY")
        from app.services.model_service import InferenceResult
        return InferenceResult(
            raw_score=self._score,
            llr_score=0.5,
            latency_ms=12.0,
        )


# ─── App client ───────────────────────────────────────────────────────────────


@pytest.fixture()
def mock_model_service(monkeypatch):
    """Patch get_model_service() to return a mock (score=0.87, ready=True)."""
    svc = _MockModelService(score=0.87, ready=True)
    import app.api.analysis   as analysis_mod
    import app.api.health     as health_mod
    import app.api.websocket  as ws_mod
    import app.services.model_service as svc_mod

    monkeypatch.setattr(analysis_mod, "get_model_service", lambda: svc)
    monkeypatch.setattr(health_mod,   "get_model_service", lambda: svc, raising=False)
    monkeypatch.setattr(ws_mod,       "get_model_service", lambda: svc)
    monkeypatch.setattr(svc_mod,      "get_model_service", lambda: svc)
    return svc


@pytest.fixture()
def mock_model_not_ready(monkeypatch):
    """Patch get_model_service() to return a not-ready mock."""
    svc = _MockModelService(score=0.5, ready=False)
    import app.api.analysis  as analysis_mod
    import app.api.websocket as ws_mod
    import app.services.model_service as svc_mod
    monkeypatch.setattr(analysis_mod, "get_model_service", lambda: svc)
    monkeypatch.setattr(ws_mod,       "get_model_service", lambda: svc)
    monkeypatch.setattr(svc_mod,      "get_model_service", lambda: svc)
    return svc


@pytest_asyncio.fixture()
async def client(mock_model_service):
    """Async HTTP client with mock model wired in."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


@pytest_asyncio.fixture()
async def client_no_model(mock_model_not_ready):
    """Async HTTP client when model is NOT ready."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c

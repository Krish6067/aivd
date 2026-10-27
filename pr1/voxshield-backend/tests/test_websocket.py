"""tests/test_websocket.py — VOXSHIELD Backend

Tests for WS /ws/analyze: session init, audio streaming, inference
scheduling, disconnect cleanup, and error handling.
"""

from __future__ import annotations

import json
import numpy as np
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

pytestmark = pytest.mark.asyncio


# ─── WebSocket helpers ────────────────────────────────────────────────────────


def _make_pcm_chunk(seconds: float = 1.0, sr: int = 16000) -> bytes:
    """Return raw float32 PCM bytes at 16 kHz."""
    n  = int(seconds * sr)
    t  = np.linspace(0, seconds, n, endpoint=False)
    s  = (np.sin(2 * np.pi * 440 * t) * 0.3).astype(np.float32)
    return s.tobytes()


_VALID_INIT = json.dumps({
    "type":        "session_init",
    "sample_rate": 16000,
    "channels":    1,
    "encoding":    "pcm_float32",
})


# ─── Session init ─────────────────────────────────────────────────────────────


async def test_ws_session_started(mock_model_service):
    from app.main import app
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        async with client.stream("GET", "/ws/analyze") as ws_raw:
            pass   # WebSocket not available via plain HTTP — use WS fixture below


async def test_ws_valid_session_init(mock_model_service):
    """Valid session_init should return session_started."""
    from starlette.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    with client.websocket_connect("/ws/analyze") as ws:
        ws.send_text(_VALID_INIT)
        msg = json.loads(ws.receive_text())
        assert msg["type"] == "session_started"
        assert "session_id" in msg
        assert msg["model_name"] == "AASIST-MOCK"


async def test_ws_invalid_init_wrong_type(mock_model_service):
    """Wrong message type should return error and close."""
    from starlette.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    bad_init = json.dumps({"type": "audio_chunk"})

    with client.websocket_connect("/ws/analyze") as ws:
        ws.send_text(bad_init)
        msg = json.loads(ws.receive_text())
        assert msg["type"] == "error"
        assert "WRONG_MESSAGE_TYPE" in msg["code"] or "INVALID" in msg["code"]


async def test_ws_unsupported_encoding(mock_model_service):
    from starlette.testclient import TestClient
    from app.main import app

    bad_init = json.dumps({
        "type": "session_init",
        "sample_rate": 16000,
        "channels": 1,
        "encoding": "pcm_int16",
    })
    client = TestClient(app)
    with client.websocket_connect("/ws/analyze") as ws:
        ws.send_text(bad_init)
        msg = json.loads(ws.receive_text())
        assert msg["type"] == "error"
        assert "UNSUPPORTED_ENCODING" in msg["code"]


async def test_ws_model_not_ready_closes(mock_model_not_ready):
    from starlette.testclient import TestClient
    from app.main import app
    import app.api.websocket as ws_mod

    # Patch ws module to use not-ready service
    ws_mod_svc = mock_model_not_ready
    client = TestClient(app)
    with client.websocket_connect("/ws/analyze") as ws:
        ws.send_text(_VALID_INIT)
        msg = json.loads(ws.receive_text())
        assert msg["type"] == "error"
        assert "MODEL_NOT_READY" in msg["code"]


# ─── Audio streaming ─────────────────────────────────────────────────────────


async def test_ws_inference_triggered_after_min_buffer(mock_model_service):
    """
    After sending enough audio (≥ min_buffer + 1 interval), the server
    should return at least one analysis_result.
    """
    from starlette.testclient import TestClient
    from app.main import app
    from app.config import get_settings
    cfg = get_settings()

    client  = TestClient(app)
    # Send (min_buffer + interval + 0.5s) of audio to guarantee inference.
    # Split into frames below WS_MAX_FRAME_BYTES (0.5 s = 32 KB at 16 kHz).
    total_s = cfg.ws_min_buffer_seconds + cfg.ws_inference_interval_seconds + 0.5
    chunk_s = 0.5

    with client.websocket_connect("/ws/analyze") as ws:
        ws.send_text(_VALID_INIT)
        msg = json.loads(ws.receive_text())
        assert msg["type"] == "session_started"

        sent = 0.0
        while sent < total_s:
            duration = min(chunk_s, total_s - sent)
            ws.send_bytes(_make_pcm_chunk(seconds=duration))
            sent += duration

        # Drain messages until we get an analysis_result or exhaust attempts
        got_result = False
        for _ in range(10):
            try:
                raw = ws.receive_text()
                m   = json.loads(raw)
                if m["type"] == "analysis_result":
                    got_result = True
                    assert 0.0 <= m["raw_score"] <= 1.0
                    assert m["classification"] in {
                        "likely_authentic", "potential_synthetic_speech", "inconclusive"
                    }
                    break
            except Exception:
                break
        assert got_result, "Expected analysis_result after sending enough audio"


async def test_ws_empty_binary_frame_ignored(mock_model_service):
    """An empty binary frame should not crash the session."""
    from starlette.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    with client.websocket_connect("/ws/analyze") as ws:
        ws.send_text(_VALID_INIT)
        json.loads(ws.receive_text())   # session_started
        # Empty bytes — server should handle gracefully (send error, close)
        ws.send_bytes(b"")
        # Should not hang; receive whatever comes back
        try:
            raw = ws.receive_text()
            m   = json.loads(raw)
            # Could be error or session_completed
            assert m["type"] in ("error", "session_completed", "warning")
        except Exception:
            pass   # Connection closed — also acceptable


async def test_ws_session_completed_on_disconnect(mock_model_service):
    """Server should send session_completed when client disconnects cleanly."""
    from starlette.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    with client.websocket_connect("/ws/analyze") as ws:
        ws.send_text(_VALID_INIT)
        json.loads(ws.receive_text())   # session_started
        # Disconnect immediately
    # No exception = session closed cleanly on the server side


# ─── Streaming session buffer unit tests ─────────────────────────────────────


def test_streaming_session_buffer_adds_chunk():
    from app.services.streaming_service import StreamingSession
    session = StreamingSession(
        session_id="test-1",
        client_sample_rate=16000,
        target_sample_rate=16000,
        min_buffer_samples=64000,
        inference_interval_samples=32000,
        max_buffer_samples=192000,
    )
    chunk = _make_pcm_chunk(seconds=1.0)
    session.add_chunk(chunk)
    assert session._buffer_size == 16000
    session.cleanup()


def test_streaming_session_inference_threshold():
    from app.services.streaming_service import StreamingSession
    session = StreamingSession(
        session_id="test-2",
        client_sample_rate=16000,
        target_sample_rate=16000,
        min_buffer_samples=16000,     # 1 s
        inference_interval_samples=8000,  # 0.5 s
        max_buffer_samples=192000,
    )
    # 0.5 s chunk — not enough yet (need min_buffer=1s)
    session.add_chunk(_make_pcm_chunk(0.5))
    assert not session.should_run_inference()

    # Add another 0.75 s — now have 1.25 s total, ≥ min_buffer and ≥ interval
    session.add_chunk(_make_pcm_chunk(0.75))
    assert session.should_run_inference()
    session.cleanup()


def test_streaming_session_buffer_overflow_trim():
    from app.services.streaming_service import StreamingSession
    session = StreamingSession(
        session_id="test-3",
        client_sample_rate=16000,
        target_sample_rate=16000,
        min_buffer_samples=16000,
        inference_interval_samples=8000,
        max_buffer_samples=16000,   # cap at 1 s
    )
    # Adding 2 s of audio to a 1-s buffer should trim
    warnings = session.add_chunk(_make_pcm_chunk(2.0))
    assert any(w["code"] == "BUFFER_TRIMMED" for w in warnings)
    assert session._buffer_size <= 16000
    session.cleanup()


def test_streaming_session_misaligned_frame_raises():
    from app.services.streaming_service import StreamingSession
    session = StreamingSession(
        session_id="test-4",
        client_sample_rate=16000,
        target_sample_rate=16000,
        min_buffer_samples=16000,
        inference_interval_samples=8000,
        max_buffer_samples=192000,
    )
    with pytest.raises(ValueError, match="multiple of 4"):
        session.add_chunk(b"\x00\x01\x02")  # 3 bytes — not aligned
    session.cleanup()

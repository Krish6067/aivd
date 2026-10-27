"""tests/test_analysis.py — VOXSHIELD Backend

Tests for the full POST /api/analyze inference pipeline.
Real AASIST weight tests are skipped when weights are not present.
"""

from __future__ import annotations

import os
import pytest
import numpy as np

pytestmark = pytest.mark.asyncio

WEIGHTS_PATH = "models/AASIST/weights/AASIST.pth"
WEIGHTS_AVAILABLE = os.path.isfile(WEIGHTS_PATH)
skip_without_weights = pytest.mark.skipif(
    not WEIGHTS_AVAILABLE,
    reason="AASIST weights not found. Run: python models/AASIST/download_weights.py",
)


# ─── Mock inference: score classification ─────────────────────────────────────


async def test_high_score_classified_as_authentic(client):
    """Mock returns 0.87 → likely_authentic."""
    from tests.conftest import make_wav_bytes
    wav = make_wav_bytes(duration_seconds=4.0)
    r   = await client.post(
        "/api/analyze",
        files={"file": ("test.wav", wav, "audio/wav")},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["classification"] == "likely_authentic"
    assert body["raw_score"] >= 0.70


async def test_response_contains_timing(client):
    """processing_latency_ms must be a positive number."""
    from tests.conftest import make_wav_bytes
    wav  = make_wav_bytes(duration_seconds=4.0)
    r    = await client.post(
        "/api/analyze",
        files={"file": ("test.wav", wav, "audio/wav")},
    )
    body = r.json()
    assert body["processing_latency_ms"] > 0


async def test_response_audio_duration(client):
    """audio_duration_seconds should approximate the generated audio length."""
    from tests.conftest import make_wav_bytes
    wav  = make_wav_bytes(duration_seconds=4.0)
    r    = await client.post(
        "/api/analyze",
        files={"file": ("test.wav", wav, "audio/wav")},
    )
    body = r.json()
    assert abs(body["audio_duration_seconds"] - 4.0) < 0.2


async def test_sample_rate_in_response(client):
    from tests.conftest import make_wav_bytes
    wav  = make_wav_bytes(duration_seconds=4.0)
    r    = await client.post(
        "/api/analyze",
        files={"file": ("test.wav", wav, "audio/wav")},
    )
    body = r.json()
    assert body["sample_rate"] == 16000


# ─── Audio processing unit tests ──────────────────────────────────────────────


def test_decode_audio_wav():
    """decode_audio should return a DecodedAudio with correct properties."""
    from tests.conftest import make_wav_bytes
    from app.services.audio_processing import decode_audio

    wav   = make_wav_bytes(duration_seconds=3.0, sample_rate=16000)
    audio = decode_audio(wav)

    assert audio.sample_rate == 16000
    assert abs(audio.duration_seconds - 3.0) < 0.1
    assert audio.waveform.dtype == np.float32
    assert audio.original_channels == 1


def test_decode_audio_stereo_downmix():
    """Stereo WAV should be downmixed to mono."""
    import io, wave
    sr = 16000
    n  = sr * 2  # 2 seconds
    samples = np.zeros((n, 2), dtype=np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples.tobytes())
    wav_bytes = buf.getvalue()

    from app.services.audio_processing import decode_audio
    audio = decode_audio(wav_bytes)
    assert audio.original_channels == 2
    assert audio.waveform.ndim == 1   # downmixed to mono


def test_preprocess_resamples_to_16k():
    """Audio at 8 kHz should be resampled to 16 kHz by preprocess_for_model."""
    import io, wave
    sr = 8000
    n  = sr * 2
    samples = (np.sin(np.linspace(0, 2*np.pi*440, n)) * 16000).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples.tobytes())
    wav_bytes = buf.getvalue()

    from app.services.audio_processing import decode_audio, preprocess_for_model
    audio  = decode_audio(wav_bytes)
    tensor = preprocess_for_model(audio, target_sr=16000, device="cpu")
    # Resampled duration should still be ~2 seconds at 16 kHz
    assert tensor.shape[0] == 1
    assert abs(tensor.shape[1] - 16000 * 2) < 200


def test_silence_detection():
    from app.utils.validation import is_silent
    silent_arr = np.zeros(16000, dtype=np.float32)
    loud_arr   = np.ones(16000, dtype=np.float32) * 0.5
    assert is_silent(silent_arr) is True
    assert is_silent(loud_arr)   is False


# ─── Real weight tests (skipped when weights not present) ────────────────────


@skip_without_weights
async def test_real_inference_valid_audio():
    """
    Load actual AASIST weights and run inference on a synthetic sine wave.

    The sine wave is NOT representative of real speech, so the score may
    fall into the 'inconclusive' range.  This test only verifies that:
      - Inference completes without error.
      - The score is within [0, 1].
      - The classification is a known value.
    """
    import torch
    from app.services.model_service import ModelService, classify_score

    svc = ModelService()
    svc.load(WEIGHTS_PATH, device="cpu", max_workers=1)
    assert svc.is_ready(), "Model failed to load from real weights"

    from tests.conftest import make_wav_bytes
    from app.services.audio_processing import decode_audio, preprocess_for_model

    wav    = make_wav_bytes(duration_seconds=4.0)
    audio  = decode_audio(wav)
    tensor = preprocess_for_model(audio, target_sr=16000, device="cpu")

    result = await svc.infer_async(tensor)
    assert 0.0 <= result.raw_score <= 1.0

    cls, interp = classify_score(result.raw_score)
    assert cls in {"likely_authentic", "potential_synthetic_speech", "inconclusive"}
    assert len(interp) > 0

    svc.shutdown()

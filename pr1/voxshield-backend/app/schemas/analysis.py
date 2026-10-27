"""
app/schemas/analysis.py — VOXSHIELD Backend

All Pydantic response/request models used by the API.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


# ─── Enumerations ────────────────────────────────────────────────────────────


class Classification(str, Enum):
    """Possible voice authenticity classifications."""
    LIKELY_AUTHENTIC         = "likely_authentic"
    POTENTIAL_SYNTHETIC      = "potential_synthetic_speech"
    INCONCLUSIVE             = "inconclusive"


class ModelReadiness(str, Enum):
    READY      = "ready"
    NOT_READY  = "not_ready"
    LOADING    = "loading"
    ERROR      = "error"


class WsMessageType(str, Enum):
    """WebSocket message type field values."""
    SESSION_INIT       = "session_init"
    SESSION_STARTED    = "session_started"
    AUDIO_CHUNK        = "audio_chunk"   # metadata only (data is binary frame)
    ANALYSIS_RESULT    = "analysis_result"
    WARNING            = "warning"
    ERROR              = "error"
    SESSION_COMPLETED  = "session_completed"
    PING               = "ping"
    PONG               = "pong"


# ─── Shared helpers ───────────────────────────────────────────────────────────

def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _new_session_id() -> str:
    return str(uuid4())


# ─── File Analysis ───────────────────────────────────────────────────────────


class AudioWarning(BaseModel):
    """A non-fatal warning attached to an analysis result."""
    code: str = Field(..., description="Machine-readable warning code")
    message: str = Field(..., description="Human-readable warning detail")


class AnalysisResult(BaseModel):
    """
    Structured result returned by POST /api/analyze and WS analysis_result messages.

    score_type explains what raw_score represents:
      'bonafide_probability' — softmax P(bonafide); higher = more authentic.

    Classification thresholds (based on AASIST validation on ASVspoof 2019 LA):
      >= 0.70 → likely_authentic
      0.40–0.70 → inconclusive
      < 0.40 → potential_synthetic_speech

    These thresholds should be re-validated for your deployment context.
    Detection results are probabilistic and must not be treated as definitive proof.
    """
    session_id: str            = Field(default_factory=_new_session_id)
    classification: Classification
    raw_score: float           = Field(..., ge=0.0, le=1.0,
                                       description="Bonafide softmax probability (0–1)")
    score_type: str            = Field("bonafide_probability",
                                       description="Interpretation of raw_score")
    score_interpretation: str  = Field(..., description="Plain-English score meaning")
    model_name: str
    model_version: str
    processing_latency_ms: float
    audio_duration_seconds: float
    sample_rate: int
    timestamp: str             = Field(default_factory=_utc_now)
    warnings: List[AudioWarning] = Field(default_factory=list)


# ─── Health & Status ─────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    timestamp: str = Field(default_factory=_utc_now)


class ModelStatusResponse(BaseModel):
    model_name: str
    model_version: str
    readiness: ModelReadiness
    device: str
    weights_path: str
    supported_sample_rate: int
    supported_channels: int = 1
    min_audio_duration_seconds: float
    max_audio_duration_seconds: float
    error_detail: Optional[str] = None
    timestamp: str = Field(default_factory=_utc_now)


# ─── WebSocket messages ───────────────────────────────────────────────────────


class WsSessionInit(BaseModel):
    """Client → Server: first JSON message to open a streaming session."""
    type: WsMessageType = WsMessageType.SESSION_INIT
    sample_rate: int        = Field(..., description="Client's audio sample rate (must be 16000)")
    channels: int           = Field(1, description="Channel count (1 = mono, 2 = stereo)")
    encoding: str           = Field("pcm_float32",
                                    description="pcm_float32 = raw IEEE-754 float32, little-endian")


class WsSessionStarted(BaseModel):
    """Server → Client: confirmation after session_init."""
    type: WsMessageType    = WsMessageType.SESSION_STARTED
    session_id: str        = Field(default_factory=_new_session_id)
    model_name: str
    model_version: str
    min_buffer_seconds: float
    inference_interval_seconds: float
    required_sample_rate: int
    required_encoding: str = "pcm_float32"
    timestamp: str         = Field(default_factory=_utc_now)


class WsAnalysisResult(BaseModel):
    """Server → Client: periodic inference result during streaming."""
    type: WsMessageType    = WsMessageType.ANALYSIS_RESULT
    session_id: str
    classification: Classification
    raw_score: float       = Field(..., ge=0.0, le=1.0)
    score_type: str        = "bonafide_probability"
    score_interpretation: str
    buffer_duration_seconds: float
    processing_latency_ms: float
    timestamp: str         = Field(default_factory=_utc_now)
    warnings: List[AudioWarning] = Field(default_factory=list)


class WsWarning(BaseModel):
    type: WsMessageType = WsMessageType.WARNING
    session_id: str
    code: str
    message: str
    timestamp: str = Field(default_factory=_utc_now)


class WsError(BaseModel):
    type: WsMessageType = WsMessageType.ERROR
    session_id: str
    code: str
    message: str
    timestamp: str = Field(default_factory=_utc_now)


class WsSessionCompleted(BaseModel):
    type: WsMessageType = WsMessageType.SESSION_COMPLETED
    session_id: str
    total_audio_seconds: float
    total_inferences: int
    timestamp: str = Field(default_factory=_utc_now)


# ─── Error responses ──────────────────────────────────────────────────────────


class ErrorDetail(BaseModel):
    code: str
    message: str
    timestamp: str = Field(default_factory=_utc_now)

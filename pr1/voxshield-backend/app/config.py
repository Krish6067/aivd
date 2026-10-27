"""
app/config.py — VOXSHIELD Backend

All runtime configuration is read from environment variables (or a .env file).
Pydantic-settings validates types and provides defaults.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Runtime configuration loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── Server ────────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8001
    app_version: str = "0.1.0"
    log_level: str = "info"

    # ── CORS ──────────────────────────────────────────────────────────────────
    # Allow all localhost ports for development + explicit frontend origins
    allowed_origins: str = "http://localhost:8000,http://127.0.0.1:8000,http://localhost:5500,http://127.0.0.1:5500,http://localhost:3000,http://127.0.0.1:3000,http://localhost:8080,http://127.0.0.1:8080"

    @property
    def cors_origins(self) -> List[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]

    # ── Upload limits ─────────────────────────────────────────────────────────
    max_upload_size_mb: float = 50.0
    max_audio_duration_seconds: float = 60.0
    min_audio_duration_seconds: float = 0.5

    @property
    def max_upload_size_bytes(self) -> int:
        return int(self.max_upload_size_mb * 1024 * 1024)

    # ── Accepted MIME types ───────────────────────────────────────────────────
    accepted_mime_types: str = (
        "audio/wav,audio/x-wav,audio/mpeg,audio/mp3,"
        "audio/ogg,audio/flac,audio/x-flac,"
        "audio/mp4,audio/x-m4a,audio/aac"
    )

    @property
    def accepted_mime_set(self) -> set:
        return {m.strip().lower() for m in self.accepted_mime_types.split(",") if m.strip()}

    # ── Model ─────────────────────────────────────────────────────────────────
    model_name: str = "AASIST"
    model_version: str = "1.0.0"
    model_weights_path: str = "models/AASIST/weights/AASIST.pth"
    device: str = ""  # empty = auto-detect

    @property
    def resolved_device(self) -> str:
        """Auto-detect GPU/CPU if DEVICE env var is blank."""
        if self.device:
            return self.device
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    @property
    def weights_path(self) -> Path:
        return Path(self.model_weights_path)

    # ── Inference ─────────────────────────────────────────────────────────────
    model_sample_rate: int = 16000
    max_inference_workers: int = 2

    # ── WebSocket / Streaming ─────────────────────────────────────────────────
    ws_min_buffer_seconds: float = 4.0
    ws_inference_interval_seconds: float = 2.0
    ws_max_buffer_seconds: float = 12.0
    ws_session_timeout_seconds: float = 60.0
    ws_max_sessions: int = 10
    ws_max_frame_bytes: int = 65536

    # ── Temp storage ──────────────────────────────────────────────────────────
    temp_dir: Optional[str] = None

    # ── Derived limits ────────────────────────────────────────────────────────
    @property
    def ws_min_buffer_samples(self) -> int:
        return int(self.ws_min_buffer_seconds * self.model_sample_rate)

    @property
    def ws_inference_interval_samples(self) -> int:
        return int(self.ws_inference_interval_seconds * self.model_sample_rate)

    @property
    def ws_max_buffer_samples(self) -> int:
        return int(self.ws_max_buffer_seconds * self.model_sample_rate)

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        valid = {"debug", "info", "warning", "error", "critical"}
        if v.lower() not in valid:
            raise ValueError(f"log_level must be one of {valid}")
        return v.lower()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached singleton Settings instance."""
    settings = Settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )
    return settings

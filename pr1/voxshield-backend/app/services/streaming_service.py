"""
app/services/streaming_service.py — VOXSHIELD Backend

WebSocket streaming session manager.

Responsibilities:
  - Maintains per-session audio ring buffer.
  - Applies backpressure: rejects chunks when buffer is full.
  - Decides when enough audio has accumulated to run inference.
  - Caps the number of concurrent sessions.
  - Cleans up all resources on disconnect.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import deque
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ─── Single streaming session ─────────────────────────────────────────────────


class StreamingSession:
    """
    Manages audio buffering and inference scheduling for one WebSocket client.

    Audio protocol:
      - Binary frames: raw PCM float32 little-endian, mono, 16 kHz.
      - Client declares sample_rate in session_init.
        If sample_rate ≠ 16000, the backend resamples before buffering.
      - Buffer is a rolling deque capped at max_buffer_samples.
      - Inference fires when accumulated_since_last_infer ≥ inference_interval_samples.

    Thread safety:
      Designed for single-threaded asyncio use per session.
      Model inference is off-loaded via model_service.infer_async() (thread pool).
    """

    def __init__(
        self,
        session_id: str,
        client_sample_rate: int,
        target_sample_rate: int,
        min_buffer_samples: int,
        inference_interval_samples: int,
        max_buffer_samples: int,
    ):
        self.session_id              = session_id
        self.client_sample_rate      = client_sample_rate
        self.target_sample_rate      = target_sample_rate
        self.min_buffer_samples      = min_buffer_samples
        self.inference_interval_samp = inference_interval_samples
        self.max_buffer_samples      = max_buffer_samples

        # Rolling buffer (deque of float32 samples)
        self._buffer: deque = deque()
        self._buffer_size: int = 0

        # Samples added since last inference
        self._since_last_infer: int = 0

        # Counters
        self.total_samples: int    = 0
        self.total_inferences: int = 0
        self.created_at: float     = time.time()
        self.last_activity: float  = time.time()

        logger.debug(
            "Session %s created (client_sr=%d, target_sr=%d, min_buf=%d, interval=%d).",
            session_id, client_sample_rate, target_sample_rate,
            min_buffer_samples, inference_interval_samples,
        )

    # ── Buffer management ─────────────────────────────────────────────────────

    def add_chunk(self, raw_bytes: bytes) -> list[dict]:
        """
        Decode a PCM float32 binary frame and append to the rolling buffer.

        Returns a list of warning dicts if the buffer was trimmed or input
        was malformed (non-fatal).

        Raises:
            ValueError: if the frame is empty or not aligned to 4 bytes.
        """
        if len(raw_bytes) == 0:
            raise ValueError("Received empty audio frame.")
        if len(raw_bytes) % 4 != 0:
            raise ValueError(
                f"Binary frame length {len(raw_bytes)} is not a multiple of 4. "
                "Expected raw PCM float32 (4 bytes per sample)."
            )

        from app.services.audio_processing import pcm_float32_bytes_to_numpy

        chunk = pcm_float32_bytes_to_numpy(raw_bytes)

        # Resample if client SR differs from model SR
        if self.client_sample_rate != self.target_sample_rate:
            import librosa
            chunk = librosa.resample(
                chunk,
                orig_sr=self.client_sample_rate,
                target_sr=self.target_sample_rate,
            )

        n = len(chunk)
        warnings = []

        # Backpressure: a single oversized chunk cannot be made to fit by
        # trimming the existing buffer, so keep only its newest samples.
        if n > self.max_buffer_samples:
            dropped = n - self.max_buffer_samples
            chunk = chunk[-self.max_buffer_samples:]
            n = len(chunk)
            self._buffer.clear()
            self._buffer_size = 0
            warnings.append({
                "code":    "BUFFER_TRIMMED",
                "message": (
                    f"Incoming chunk exceeded maximum buffer size "
                    f"({self.max_buffer_samples} samples). "
                    f"Oldest {dropped} samples discarded."
                ),
            })
        else:
            overflow = (self._buffer_size + n) - self.max_buffer_samples
            if overflow > 0:
                self._trim_oldest(overflow)
                warnings.append({
                    "code":    "BUFFER_TRIMMED",
                    "message": (
                        f"Audio buffer exceeded maximum ({self.max_buffer_samples} samples). "
                        f"Oldest {overflow} samples discarded."
                    ),
                })

        self._buffer.append(chunk)
        self._buffer_size     += n
        self._since_last_infer += n
        self.total_samples    += n
        self.last_activity     = time.time()

        return warnings

    def _trim_oldest(self, n_to_remove: int) -> None:
        """Remove n_to_remove samples from the front of the buffer."""
        removed = 0
        while removed < n_to_remove and self._buffer:
            chunk = self._buffer.popleft()
            chunk_len = len(chunk)
            remaining = n_to_remove - removed

            if chunk_len <= remaining:
                removed           += chunk_len
                self._buffer_size -= chunk_len
            else:
                # Partial trim: keep the newest part of this chunk.
                keep = chunk_len - remaining
                self._buffer.appendleft(chunk[-keep:])
                self._buffer_size -= remaining
                removed           += remaining

    def should_run_inference(self) -> bool:
        """
        Return True when:
          - The buffer holds at least min_buffer_samples, AND
          - At least inference_interval_samples new samples arrived since last inference.
        """
        return (
            self._buffer_size >= self.min_buffer_samples
            and self._since_last_infer >= self.inference_interval_samp
        )

    def get_window(self) -> np.ndarray:
        """
        Return a contiguous float32 NumPy array of the current buffer contents.
        Resets the since-last-infer counter.
        """
        self._since_last_infer  = 0
        self.total_inferences  += 1
        return np.concatenate(list(self._buffer), axis=0)

    # ── Metadata ──────────────────────────────────────────────────────────────

    @property
    def buffer_duration_seconds(self) -> float:
        return self._buffer_size / self.target_sample_rate

    @property
    def total_audio_seconds(self) -> float:
        return self.total_samples / self.target_sample_rate

    def is_timed_out(self, timeout_seconds: float) -> bool:
        return (time.time() - self.last_activity) > timeout_seconds

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def cleanup(self) -> None:
        """Release buffer memory."""
        self._buffer.clear()
        self._buffer_size     = 0
        self._since_last_infer = 0
        logger.debug("Session %s cleaned up.", self.session_id)


# ─── Session registry ─────────────────────────────────────────────────────────


class SessionRegistry:
    """
    Thread-safe (asyncio-safe) registry for concurrent streaming sessions.

    Enforces a maximum session cap to avoid resource exhaustion.
    """

    def __init__(self, max_sessions: int):
        self._max_sessions = max_sessions
        self._sessions: Dict[str, StreamingSession] = {}
        self._lock = asyncio.Lock()

    async def create_session(
        self,
        client_sample_rate: int,
        target_sample_rate: int,
        min_buffer_samples: int,
        inference_interval_samples: int,
        max_buffer_samples: int,
    ) -> StreamingSession:
        """
        Create and register a new session.

        Raises:
            RuntimeError: if max_sessions is already reached.
        """
        async with self._lock:
            if len(self._sessions) >= self._max_sessions:
                raise RuntimeError(
                    f"Maximum concurrent sessions ({self._max_sessions}) reached. "
                    "Please try again later."
                )
            session_id = str(uuid.uuid4())
            session = StreamingSession(
                session_id=session_id,
                client_sample_rate=client_sample_rate,
                target_sample_rate=target_sample_rate,
                min_buffer_samples=min_buffer_samples,
                inference_interval_samples=inference_interval_samples,
                max_buffer_samples=max_buffer_samples,
            )
            self._sessions[session_id] = session
            logger.info("Session %s opened. Active: %d.", session_id, len(self._sessions))
            return session

    async def remove_session(self, session_id: str) -> None:
        async with self._lock:
            session = self._sessions.pop(session_id, None)
            if session:
                session.cleanup()
                logger.info(
                    "Session %s closed. Active: %d.", session_id, len(self._sessions)
                )

    @property
    def active_count(self) -> int:
        return len(self._sessions)


# ─── Singletons ───────────────────────────────────────────────────────────────

_registry: Optional[SessionRegistry] = None


def get_session_registry() -> SessionRegistry:
    global _registry
    if _registry is None:
        from app.config import get_settings
        cfg = get_settings()
        _registry = SessionRegistry(max_sessions=cfg.ws_max_sessions)
    return _registry

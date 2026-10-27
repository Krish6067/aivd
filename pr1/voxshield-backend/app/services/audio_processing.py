"""
app/services/audio_processing.py — VOXSHIELD Backend

Audio decoding, validation, and preprocessing pipeline.

Design principles:
- Never assume a file is valid because of its extension or MIME type.
- Validate decoded content (duration, sample rate, channels, silence).
- Resample only when necessary; preserve model preprocessing assumptions.
- All preprocessing mirrors what AASIST was trained with on ASVspoof 2019.
- Return structured warnings alongside the processed tensor.
"""

from __future__ import annotations

import io
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Tuple

import numpy as np

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

# ─── Data containers ──────────────────────────────────────────────────────────


@dataclass
class DecodedAudio:
    """Raw decoded audio before model-specific preprocessing."""
    waveform: np.ndarray    # float32, shape [samples], range roughly [-1, 1]
    sample_rate: int
    duration_seconds: float
    original_channels: int
    warnings: List[dict] = field(default_factory=list)


@dataclass
class ProcessedAudio:
    """Model-ready audio tensor plus provenance metadata."""
    tensor: "torch.Tensor"           # shape [1, T], float32, on model device
    sample_rate: int
    duration_seconds: float
    decode_time_ms: float
    preprocess_time_ms: float
    warnings: List[dict] = field(default_factory=list)


# ─── Decoding ─────────────────────────────────────────────────────────────────


def decode_audio(data: bytes) -> DecodedAudio:
    """
    Decode raw audio bytes into a float32 NumPy array.

    Tries soundfile first (native, fast, no subprocess).
    Falls back to librosa (which uses soundfile + ffmpeg for MP3/M4A).
    Stereo audio is downmixed to mono by averaging channels.

    Raises:
        ValueError: if the file cannot be decoded.
    """
    import soundfile as sf

    try:
        buf = io.BytesIO(data)
        waveform, sr = sf.read(buf, dtype="float32", always_2d=True)
        # waveform: [samples, channels]
    except Exception as sf_err:
        logger.debug("soundfile failed (%s), trying librosa.", sf_err)
        try:
            import librosa
            buf = io.BytesIO(data)
            waveform_mono, sr = librosa.load(buf, sr=None, mono=False)
            # librosa returns [channels, samples] or [samples] for mono
            if waveform_mono.ndim == 1:
                waveform = waveform_mono[:, np.newaxis]
            else:
                waveform = waveform_mono.T   # → [samples, channels]
        except Exception as lib_err:
            raise ValueError(
                f"Could not decode audio file. "
                f"soundfile: {sf_err}; librosa: {lib_err}. "
                "Ensure the file is a valid WAV, MP3, OGG, FLAC, or M4A."
            ) from lib_err

    n_samples, n_channels = waveform.shape

    # Downmix stereo → mono
    if n_channels > 1:
        waveform_mono = waveform.mean(axis=1)
    else:
        waveform_mono = waveform[:, 0]

    duration = n_samples / sr

    return DecodedAudio(
        waveform=waveform_mono,
        sample_rate=sr,
        duration_seconds=duration,
        original_channels=n_channels,
    )


# ─── Preprocessing ────────────────────────────────────────────────────────────


def preprocess_for_model(
    audio: DecodedAudio,
    target_sr: int = 16000,
    device: str = "cpu",
) -> "torch.Tensor":
    """
    Convert a DecodedAudio object into a model-ready tensor.

    Steps:
    1. Resample to target_sr if needed (using scipy polyphase resampling).
    2. Ensure float32.
    3. No amplitude normalisation — AASIST was trained without it.
    4. Shape: [1, T] on the target device.

    Returns:
        Tensor of shape [1, T] on `device`.
    """
    import torch
    from math import gcd
    from scipy.signal import resample_poly

    waveform = audio.waveform.astype(np.float32)

    if audio.sample_rate != target_sr:
        divisor = gcd(audio.sample_rate, target_sr)
        waveform = resample_poly(
            waveform,
            up=target_sr // divisor,
            down=audio.sample_rate // divisor,
        ).astype(np.float32)

    tensor = torch.from_numpy(waveform).unsqueeze(0)   # [1, T]
    return tensor.to(device)


# ─── Full pipeline (used by REST endpoint) ───────────────────────────────────


def process_upload(
    file_bytes: bytes,
    target_sr: int,
    min_duration: float,
    max_duration: float,
    device: str = "cpu",
) -> ProcessedAudio:
    """
    Complete pipeline: bytes → validated, preprocessed tensor.

    Raises ValueError with a user-friendly message on any failure.
    Returns ProcessedAudio with timing metadata and non-fatal warnings.
    """
    from app.utils.validation import (
        validate_audio_duration,
        validate_sample_rate,
        validate_channels,
        validate_audio_content,
    )
    import torch

    # ── 1. Decode ─────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    audio = decode_audio(file_bytes)
    decode_ms = (time.perf_counter() - t0) * 1000.0

    warnings: List[dict] = list(audio.warnings)

    # ── 2. Validate duration ──────────────────────────────────────────────────
    dur_warnings = validate_audio_duration(
        audio.duration_seconds, min_duration, max_duration
    )
    warnings += [{"code": "SHORT_AUDIO", "message": w} for w in dur_warnings]

    # ── 3. Validate / warn about sample rate ──────────────────────────────────
    sr_warnings = validate_sample_rate(audio.sample_rate, target_sr)
    warnings += [{"code": "RESAMPLED", "message": w} for w in sr_warnings]

    # ── 4. Validate / warn about channels ────────────────────────────────────
    ch_warnings = validate_channels(audio.original_channels)
    warnings += [{"code": "DOWNMIXED", "message": w} for w in ch_warnings]

    # ── 5. Content checks (silence, clipping) ─────────────────────────────────
    content_warnings = validate_audio_content(audio.waveform, audio.duration_seconds)
    warnings += content_warnings

    # ── 6. Preprocess → tensor ────────────────────────────────────────────────
    t1 = time.perf_counter()
    tensor = preprocess_for_model(audio, target_sr=target_sr, device=device)
    preprocess_ms = (time.perf_counter() - t1) * 1000.0

    return ProcessedAudio(
        tensor=tensor,
        sample_rate=target_sr,
        duration_seconds=audio.duration_seconds,
        decode_time_ms=decode_ms,
        preprocess_time_ms=preprocess_ms,
        warnings=warnings,
    )


# ─── Streaming helpers ────────────────────────────────────────────────────────


def pcm_float32_bytes_to_numpy(raw_bytes: bytes) -> np.ndarray:
    """
    Convert raw PCM float32 bytes (little-endian, normalised -1 to 1)
    into a NumPy float32 array.

    Used by the WebSocket streaming handler for each binary audio frame.
    """
    return np.frombuffer(raw_bytes, dtype="<f4").astype(np.float32)


def numpy_window_to_tensor(
    window: np.ndarray,
    device: str = "cpu",
) -> "torch.Tensor":
    """
    Convert a 1-D float32 NumPy array into a [1, T] tensor on `device`.
    No normalisation applied — matches AASIST training assumptions.
    """
    import torch
    tensor = torch.from_numpy(window.copy()).unsqueeze(0)
    return tensor.to(device)

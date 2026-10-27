"""
app/utils/validation.py — VOXSHIELD Backend

Input validation helpers: MIME types, file magic bytes, size, duration guards.
These run before any audio decoding to fail fast on obviously bad input.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ─── Magic-byte signatures ────────────────────────────────────────────────────
# Maps first N bytes pattern → (human label, accepted bool)
# We only accept these formats; everything else is rejected regardless of MIME type.
_MAGIC: list[tuple[bytes, str]] = [
    (b"RIFF",           "WAV"),
    (b"fLaC",           "FLAC"),
    (b"\xff\xfb",       "MP3"),
    (b"\xff\xf3",       "MP3"),
    (b"\xff\xf2",       "MP3"),
    (b"ID3",            "MP3"),
    (b"OggS",           "OGG"),
    # MP4 / M4A: 4-byte offset + 'ftyp'
    (b"\x00\x00\x00\x20ftyp", "M4A/MP4"),
    (b"\x00\x00\x00\x1cftyp", "M4A/MP4"),
    (b"\x00\x00\x00\x18ftyp", "M4A/MP4"),
]


def validate_magic_bytes(data: bytes) -> str:
    """
    Inspect the first 12 bytes of the file data and return the detected format label.

    Raises ValueError if the file header does not match any accepted format.
    This prevents disguised files (e.g. a JPEG renamed to .wav) from being processed.
    """
    header = data[:12]
    for magic, label in _MAGIC:
        if header[: len(magic)] == magic:
            return label
    # M4A 'ftyp' can appear at different offsets — broader scan for safety
    if b"ftyp" in data[:16]:
        return "M4A/MP4"
    raise ValueError(
        "File does not have a recognised audio header. "
        "Accepted formats: WAV, MP3, OGG, FLAC, M4A."
    )


def validate_upload_size(data: bytes, max_bytes: int) -> None:
    """Raise ValueError if the payload exceeds the configured limit."""
    if len(data) > max_bytes:
        max_mb = max_bytes / (1024 * 1024)
        actual_mb = len(data) / (1024 * 1024)
        raise ValueError(
            f"Uploaded file is {actual_mb:.1f} MB; "
            f"maximum allowed is {max_mb:.1f} MB."
        )


def validate_audio_duration(
    duration_seconds: float,
    min_seconds: float,
    max_seconds: float,
) -> list[str]:
    """
    Validate audio duration against configured bounds.
    Returns a list of warning strings (non-fatal) or raises ValueError (fatal).
    """
    warnings: list[str] = []

    if duration_seconds < min_seconds:
        raise ValueError(
            f"Audio is too short ({duration_seconds:.2f}s). "
            f"Minimum is {min_seconds:.2f}s."
        )
    if duration_seconds > max_seconds:
        raise ValueError(
            f"Audio is too long ({duration_seconds:.1f}s). "
            f"Maximum is {max_seconds:.0f}s."
        )
    if duration_seconds < 2.0:
        warnings.append(
            f"Audio is very short ({duration_seconds:.2f}s). "
            "AASIST is most reliable on ≥ 4 seconds of speech. "
            "Result confidence may be reduced."
        )
    return warnings


def validate_sample_rate(actual: int, required: int) -> list[str]:
    """
    Return warnings if the sample rate is not what the model expects.
    The pipeline will resample, but this is logged as a warning.
    """
    if actual != required:
        return [
            f"Audio sample rate is {actual} Hz; "
            f"model requires {required} Hz. "
            "Audio will be resampled automatically."
        ]
    return []


def validate_channels(actual_channels: int) -> list[str]:
    """Stereo input is accepted but triggers a warning (will be mixed to mono)."""
    if actual_channels > 2:
        raise ValueError(
            f"Unsupported channel count: {actual_channels}. "
            "Only mono (1) and stereo (2) are accepted."
        )
    if actual_channels == 2:
        return ["Stereo audio will be mixed down to mono before inference."]
    return []


def is_silent(waveform_array, threshold: float = 1e-4) -> bool:
    """
    Return True if the waveform's RMS energy is below the silence threshold.
    Uses NumPy — array must be a float32 1-D NumPy array.
    """
    import numpy as np
    rms = float(np.sqrt(np.mean(waveform_array.astype(np.float64) ** 2)))
    return rms < threshold


def validate_audio_content(waveform_array, duration_seconds: float) -> list[str]:
    """
    Non-fatal content warnings for very silent or clipped audio.
    Returns a list of AudioWarning-compatible dicts.
    """
    import numpy as np
    warnings = []

    if is_silent(waveform_array):
        warnings.append(
            {
                "code": "SILENT_AUDIO",
                "message": (
                    "The uploaded audio appears to be silent or nearly silent. "
                    "Detection accuracy on silent audio is not meaningful."
                ),
            }
        )

    # Check for clipping (samples at ± 1.0)
    clip_fraction = float(np.mean(np.abs(waveform_array) >= 0.999))
    if clip_fraction > 0.01:
        warnings.append(
            {
                "code": "CLIPPED_AUDIO",
                "message": (
                    f"{clip_fraction * 100:.1f}% of samples are clipped. "
                    "Clipped audio may reduce detection accuracy."
                ),
            }
        )

    return warnings

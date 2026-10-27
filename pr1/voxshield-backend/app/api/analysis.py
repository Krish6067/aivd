"""
app/api/analysis.py — VOXSHIELD Backend

POST /api/analyze — multipart audio file upload + AASIST inference
"""

from __future__ import annotations

import logging
import time
import uuid

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.schemas.analysis import AnalysisResult, AudioWarning, ErrorDetail
from app.services.audio_processing import process_upload
from app.services.model_service import classify_score, get_model_service
from app.utils.validation import validate_magic_bytes, validate_upload_size

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/api/analyze",
    response_model=AnalysisResult,
    status_code=status.HTTP_200_OK,
    summary="Analyze an uploaded audio file for AI-generated speech",
    tags=["Analysis"],
    responses={
        400: {"model": ErrorDetail, "description": "Invalid or unsupported audio"},
        413: {"model": ErrorDetail, "description": "File too large"},
        415: {"model": ErrorDetail, "description": "Unsupported media type"},
        503: {"model": ErrorDetail, "description": "Model not ready"},
    },
)
async def analyze_audio(
    request: Request,
    file: UploadFile = File(..., description="Audio file (WAV, MP3, OGG, FLAC, M4A)"),
) -> AnalysisResult:
    """
    Upload an audio file and receive an anti-spoofing analysis result.

    **Processing pipeline:**
    1. Validate MIME type and file size.
    2. Validate file magic bytes (prevents disguised non-audio files).
    3. Decode audio with soundfile / librosa.
    4. Validate duration, sample rate, channel count, audio content.
    5. Resample to 16 kHz mono if needed.
    6. Run AASIST inference.
    7. Return structured result with score, classification, and warnings.

    **Score interpretation:**
    `raw_score` is the AASIST bonafide softmax probability (0–1).
    Higher values indicate speech more consistent with natural human voice.

    | raw_score | classification |
    |---|---|
    | ≥ 0.70 | likely_authentic |
    | 0.40–0.70 | inconclusive |
    | < 0.40 | potential_synthetic_speech |

    Detection is probabilistic and must not be treated as definitive proof.

    **Audio requirements:**
    - Formats: WAV, MP3, OGG, FLAC, M4A (decoded, not just file extension).
    - Maximum size: configured via MAX_UPLOAD_SIZE_MB (default 50 MB).
    - Duration: 0.5s – 60s (configurable). Best accuracy at ≥ 4 seconds.
    - Sample rate: any (auto-resampled to 16 kHz).
    - Channels: mono or stereo (stereo is downmixed to mono).

    Audio is processed in memory and not stored after the request completes.
    """
    cfg = get_settings()
    svc = get_model_service()
    session_id = str(uuid.uuid4())
    t_start    = time.perf_counter()

    # ── 1. Model readiness check ──────────────────────────────────────────────
    if not svc.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "MODEL_NOT_READY",
                "message": (
                    "The anti-spoofing model is not loaded. "
                    "Check GET /api/model/status for details and download instructions."
                ),
            },
        )

    # ── 2. MIME type validation ───────────────────────────────────────────────
    content_type = (file.content_type or "").lower().split(";")[0].strip()
    if content_type and content_type not in cfg.accepted_mime_set:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail={
                "code":    "UNSUPPORTED_MIME_TYPE",
                "message": (
                    f"Content-Type '{content_type}' is not accepted. "
                    f"Supported types: {', '.join(sorted(cfg.accepted_mime_set))}."
                ),
            },
        )

    # ── 3. Read file bytes ────────────────────────────────────────────────────
    try:
        file_bytes = await file.read()
    except Exception as exc:
        logger.warning("Failed to read upload: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "READ_ERROR", "message": "Could not read the uploaded file."},
        )

    # ── 4. Size validation ────────────────────────────────────────────────────
    try:
        validate_upload_size(file_bytes, cfg.max_upload_size_bytes)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={"code": "FILE_TOO_LARGE", "message": str(exc)},
        )

    if len(file_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "EMPTY_FILE", "message": "Uploaded file is empty."},
        )

    # ── 5. Magic-byte format validation ──────────────────────────────────────
    try:
        detected_format = validate_magic_bytes(file_bytes)
        logger.debug("Detected format: %s for session %s.", detected_format, session_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail={"code": "INVALID_AUDIO_FORMAT", "message": str(exc)},
        )

    # ── 6. Audio decode + preprocess ──────────────────────────────────────────
    try:
        processed = process_upload(
            file_bytes=file_bytes,
            target_sr=cfg.model_sample_rate,
            min_duration=cfg.min_audio_duration_seconds,
            max_duration=cfg.max_audio_duration_seconds,
            device=cfg.resolved_device,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "AUDIO_VALIDATION_ERROR", "message": str(exc)},
        )
    except Exception as exc:
        logger.error("Unexpected audio processing error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code":    "AUDIO_PROCESSING_ERROR",
                "message": "Audio could not be processed. Ensure the file is valid audio.",
            },
        )

    # ── 7. Model inference ────────────────────────────────────────────────────
    try:
        result = await svc.infer_async(processed.tensor)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "INFERENCE_ERROR", "message": str(exc)},
        )

    # ── 8. Classify and build response ────────────────────────────────────────
    classification, interpretation = classify_score(result.raw_score)

    total_latency_ms = (time.perf_counter() - t_start) * 1000.0

    warnings = [AudioWarning(**w) for w in processed.warnings]

    logger.info(
        "Session %s | score=%.4f | %s | %.0f ms total "
        "(decode=%.0f, preproc=%.0f, infer=%.0f)",
        session_id, result.raw_score, classification,
        total_latency_ms, processed.decode_time_ms,
        processed.preprocess_time_ms, result.latency_ms,
    )

    return AnalysisResult(
        session_id=session_id,
        classification=classification,
        raw_score=result.raw_score,
        score_type="bonafide_probability",
        score_interpretation=interpretation,
        model_name=cfg.model_name,
        model_version=cfg.model_version,
        processing_latency_ms=total_latency_ms,
        audio_duration_seconds=processed.duration_seconds,
        sample_rate=processed.sample_rate,
        warnings=warnings,
    )

"""
app/api/websocket.py — VOXSHIELD Backend

WS /ws/analyze — real-time audio streaming and inference endpoint.

Protocol summary
────────────────
Client → Server
  1. JSON text frame:  {"type": "session_init", "sample_rate": 16000,
                        "channels": 1, "encoding": "pcm_float32"}
  2. Binary frames:    Raw PCM float32 little-endian chunks (mono, 16 kHz).
                       Send continuously while audio is being captured.
  3. Disconnect:       Close the WebSocket to end the session.

Server → Client
  • session_started   — once, after valid session_init
  • analysis_result   — every time the inference window threshold is met
  • warning           — non-fatal issues (buffer trim, silent audio, etc.)
  • error             — fatal protocol / validation errors (session closes)
  • session_completed — sent just before the server closes the connection

Binary frame requirements
  - Encoding : IEEE-754 float32, little-endian ("<f4")
  - Channels : mono (1) only — stereo clients must downmix client-side
                OR declare channels=2 and the server will average on receive
  - Max frame : WS_MAX_FRAME_BYTES (default 64 KB ≈ 1 s at 16 kHz)

Security notes
  - Session count capped at WS_MAX_SESSIONS.
  - No audio retained after session teardown.
  - Idle sessions are not automatically evicted here; add a heartbeat
    mechanism (ping/pong) in production.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status

from app.config import get_settings
from app.schemas.analysis import (
    AudioWarning,
    WsAnalysisResult,
    WsError,
    WsMessageType,
    WsSessionCompleted,
    WsSessionInit,
    WsSessionStarted,
    WsWarning,
)
from app.services.audio_processing import numpy_window_to_tensor
from app.services.model_service import classify_score, get_model_service
from app.services.streaming_service import get_session_registry

logger = logging.getLogger(__name__)
router = APIRouter()


# ─── Helpers ─────────────────────────────────────────────────────────────────


async def _send_json(ws: WebSocket, payload: Any) -> None:
    """Send a Pydantic model or dict as a JSON text frame."""
    if hasattr(payload, "model_dump"):
        data = payload.model_dump()
    else:
        data = payload
    await ws.send_text(json.dumps(data))


async def _send_error(ws: WebSocket, session_id: str, code: str, message: str) -> None:
    await _send_json(ws, WsError(session_id=session_id, code=code, message=message))


async def _send_warning(ws: WebSocket, session_id: str, code: str, message: str) -> None:
    await _send_json(ws, WsWarning(session_id=session_id, code=code, message=message))


# ─── Endpoint ────────────────────────────────────────────────────────────────


@router.websocket("/ws/analyze")
async def ws_analyze(websocket: WebSocket) -> None:
    """
    WebSocket endpoint for live audio streaming and real-time anti-spoofing.

    Step-by-step flow:
      1. Accept the connection.
      2. Wait for a session_init JSON message.
      3. Validate model readiness and session parameters.
      4. Open a StreamingSession in the registry.
      5. Send session_started confirmation.
      6. Loop:
           a. Receive binary audio frame OR text message.
           b. Add chunk to session buffer.
           c. If inference threshold met → run AASIST → send analysis_result.
      7. On disconnect or error → cleanup session → send session_completed.
    """
    cfg      = get_settings()
    svc      = get_model_service()
    registry = get_session_registry()

    await websocket.accept()
    session_id = "pending"

    # ── Step 1: Session initialization handshake ──────────────────────────────
    try:
        raw_init = await asyncio.wait_for(
            websocket.receive_text(),
            timeout=cfg.ws_session_timeout_seconds,
        )
    except asyncio.TimeoutError:
        await _send_error(websocket, "pending", "TIMEOUT",
                          "Session init message not received within timeout.")
        await websocket.close(code=1008)
        return
    except WebSocketDisconnect:
        logger.info("Client disconnected before session_init.")
        return

    # Parse session_init
    try:
        init_data = json.loads(raw_init)
        init      = WsSessionInit(**init_data)
    except (json.JSONDecodeError, Exception) as exc:
        await _send_json(websocket, WsError(
            session_id="pending",
            code="INVALID_SESSION_INIT",
            message=f"Could not parse session_init: {exc}. "
                    f"Expected: {{\"type\": \"session_init\", \"sample_rate\": 16000, "
                    f"\"channels\": 1, \"encoding\": \"pcm_float32\"}}",
        ))
        await websocket.close(code=1003)
        return

    # Validate message type
    if init.type != WsMessageType.SESSION_INIT:
        await _send_json(websocket, WsError(
            session_id="pending",
            code="WRONG_MESSAGE_TYPE",
            message=f"Expected type='session_init', got '{init.type}'.",
        ))
        await websocket.close(code=1003)
        return

    # Validate encoding
    if init.encoding != "pcm_float32":
        await _send_json(websocket, WsError(
            session_id="pending",
            code="UNSUPPORTED_ENCODING",
            message=f"Encoding '{init.encoding}' is not supported. "
                    "Only 'pcm_float32' (IEEE-754 float32 little-endian) is accepted.",
        ))
        await websocket.close(code=1003)
        return

    # Validate channels
    if init.channels not in (1, 2):
        await _send_json(websocket, WsError(
            session_id="pending",
            code="UNSUPPORTED_CHANNELS",
            message=f"Channel count {init.channels} is not supported (use 1 or 2).",
        ))
        await websocket.close(code=1003)
        return

    # Validate model readiness
    if not svc.is_ready():
        await _send_json(websocket, WsError(
            session_id="pending",
            code="MODEL_NOT_READY",
            message="AASIST model is not loaded. Check GET /api/model/status.",
        ))
        await websocket.close(code=1013)
        return

    # ── Step 2: Create session ────────────────────────────────────────────────
    try:
        session = await registry.create_session(
            client_sample_rate=init.sample_rate,
            target_sample_rate=cfg.model_sample_rate,
            min_buffer_samples=cfg.ws_min_buffer_samples,
            inference_interval_samples=cfg.ws_inference_interval_samples,
            max_buffer_samples=cfg.ws_max_buffer_samples,
        )
    except RuntimeError as exc:
        await _send_json(websocket, WsError(
            session_id="pending",
            code="SESSION_LIMIT_REACHED",
            message=str(exc),
        ))
        await websocket.close(code=1013)
        return

    session_id = session.session_id

    # ── Step 3: Confirm session start ─────────────────────────────────────────
    model_status = svc.get_status()
    await _send_json(websocket, WsSessionStarted(
        session_id=session_id,
        model_name=model_status["model_name"],
        model_version=model_status["model_version"],
        min_buffer_seconds=cfg.ws_min_buffer_seconds,
        inference_interval_seconds=cfg.ws_inference_interval_seconds,
        required_sample_rate=cfg.model_sample_rate,
    ))

    logger.info("WS session %s started (client_sr=%d).", session_id, init.sample_rate)

    # ── Step 4: Audio streaming loop ─────────────────────────────────────────
    try:
        while True:
            # Receive next message (binary chunk OR text control)
            try:
                message = await asyncio.wait_for(
                    websocket.receive(),
                    timeout=cfg.ws_session_timeout_seconds,
                )
            except asyncio.TimeoutError:
                await _send_error(websocket, session_id, "SESSION_TIMEOUT",
                                  f"No data received for {cfg.ws_session_timeout_seconds}s.")
                break

            # ── Disconnect frame ───────────────────────────────────────────
            if message.get("type") == "websocket.disconnect":
                break

            # ── Text frame (ping / control) ────────────────────────────────
            if "text" in message:
                try:
                    ctrl = json.loads(message["text"])
                    if ctrl.get("type") == WsMessageType.PING:
                        await _send_json(websocket, {"type": "pong",
                                                      "session_id": session_id})
                except Exception:
                    pass   # Ignore unparseable text during streaming
                continue

            # ── Binary frame (audio chunk) ─────────────────────────────────
            if "bytes" not in message:
                continue

            raw_bytes = message["bytes"]
            if not raw_bytes:
                await _send_warning(websocket, session_id, "EMPTY_FRAME",
                                    "Received an empty audio frame. Frame skipped.")
                continue

            # Frame size limit
            if len(raw_bytes) > cfg.ws_max_frame_bytes:
                await _send_warning(websocket, session_id, "FRAME_TOO_LARGE",
                                    f"Frame {len(raw_bytes)} bytes exceeds limit "
                                    f"{cfg.ws_max_frame_bytes}. Frame skipped.")
                continue

            # Add chunk to session buffer
            try:
                chunk_warnings = session.add_chunk(raw_bytes)
            except ValueError as exc:
                await _send_error(websocket, session_id, "INVALID_CHUNK", str(exc))
                break

            # Forward any buffer warnings
            for w in chunk_warnings:
                await _send_warning(websocket, session_id, w["code"], w["message"])

            # ── Inference when threshold is met ────────────────────────────
            if not session.should_run_inference():
                continue

            window = session.get_window()
            tensor = numpy_window_to_tensor(window, device=cfg.resolved_device)

            t_infer = time.perf_counter()
            try:
                infer_result = await svc.infer_async(tensor)
            except RuntimeError as exc:
                await _send_error(websocket, session_id, "INFERENCE_ERROR", str(exc))
                break

            latency_ms = (time.perf_counter() - t_infer) * 1000.0
            classification, interpretation = classify_score(infer_result.raw_score)

            await _send_json(websocket, WsAnalysisResult(
                session_id=session_id,
                classification=classification,
                raw_score=infer_result.raw_score,
                score_interpretation=interpretation,
                buffer_duration_seconds=session.buffer_duration_seconds,
                processing_latency_ms=latency_ms,
            ))

            logger.debug(
                "WS %s | score=%.4f | %s | %.0f ms",
                session_id, infer_result.raw_score, classification, latency_ms,
            )

    except WebSocketDisconnect:
        logger.info("WS session %s: client disconnected.", session_id)
    except Exception as exc:
        logger.error("WS session %s unexpected error: %s", session_id, exc, exc_info=True)
        try:
            await _send_error(websocket, session_id, "INTERNAL_ERROR",
                              "An unexpected server error occurred.")
        except Exception:
            pass
    finally:
        # ── Step 5: Cleanup ───────────────────────────────────────────────────
        total_audio = session.total_audio_seconds
        total_infer = session.total_inferences
        await registry.remove_session(session_id)

        try:
            await _send_json(websocket, WsSessionCompleted(
                session_id=session_id,
                total_audio_seconds=total_audio,
                total_inferences=total_infer,
            ))
            await websocket.close(code=1000)
        except Exception:
            pass   # Already closed by client

        logger.info(
            "WS session %s closed. Audio=%.1fs, Inferences=%d.",
            session_id, total_audio, total_infer,
        )

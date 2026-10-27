"""
app/services/model_service.py — VOXSHIELD Backend

Singleton service that:
  - Loads the AASIST model exactly once at startup.
  - Exposes thread-safe synchronous inference.
  - Reports model readiness and metadata.
  - Runs inference in a thread-pool executor to avoid blocking the event loop.
  - Never generates random scores as a fallback — returns a clear error instead.
"""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ─── Result container ─────────────────────────────────────────────────────────


@dataclass
class InferenceResult:
    """
    Raw output from a single AASIST inference pass.

    raw_score:
      Softmax P(bonafide) in [0, 1].
      Higher = more likely authentic.
      Source: softmax(logits)[1].

    llr_score:
      Log-likelihood ratio = bonafide_logit − spoof_logit.
      Alternative representation; not exposed in API by default.

    latency_ms:
      Wall-clock time for inference only (excludes decode/preprocess).
    """
    raw_score: float      # 0–1 bonafide probability
    llr_score: float      # log-likelihood ratio (bonafide − spoof)
    latency_ms: float


# ─── Classification thresholds ────────────────────────────────────────────────
# Based on AASIST's reported EER of ≈0.83% on ASVspoof 2019 LA.
# Using softmax bonafide probability as the decision variable.
# Re-validate these thresholds before production deployment.

THRESHOLD_AUTHENTIC   = 0.70   # p_bonafide ≥ 0.70 → likely_authentic
THRESHOLD_INCONCLUSIVE = 0.40  # p_bonafide ≥ 0.40 → inconclusive
                                # p_bonafide < 0.40 → potential_synthetic_speech


def classify_score(raw_score: float) -> tuple[str, str]:
    """
    Map a bonafide probability to (classification, interpretation).

    Returns:
        classification: one of the Classification enum values (as string)
        interpretation: plain-English explanation
    """
    from app.schemas.analysis import Classification
    if raw_score >= THRESHOLD_AUTHENTIC:
        return (
            Classification.LIKELY_AUTHENTIC,
            f"Score {raw_score:.3f} ≥ {THRESHOLD_AUTHENTIC} — audio characteristics "
            "are consistent with natural human speech.",
        )
    if raw_score >= THRESHOLD_INCONCLUSIVE:
        return (
            Classification.INCONCLUSIVE,
            f"Score {raw_score:.3f} between {THRESHOLD_INCONCLUSIVE}–{THRESHOLD_AUTHENTIC} "
            "— inconclusive; characteristics are ambiguous. "
            "Longer or higher-quality audio may improve accuracy.",
        )
    return (
        Classification.POTENTIAL_SYNTHETIC,
        f"Score {raw_score:.3f} < {THRESHOLD_INCONCLUSIVE} — audio exhibits characteristics "
        "associated with synthesised or cloned speech. "
        "This is a probabilistic estimate; treat with appropriate caution.",
    )


# ─── Model Service ────────────────────────────────────────────────────────────


class ModelService:
    """
    Singleton wrapper around the AASIST model.

    Thread safety:
      Model weights are loaded once and never modified at inference time.
      PyTorch inference with torch.no_grad() is thread-safe for read-only passes.
      A bounded ThreadPoolExecutor limits concurrent inference to avoid OOM.
    """

    def __init__(self):
        self._model      = None
        self._device     = "cpu"
        self._ready      = False
        self._error: Optional[str] = None
        self._executor: Optional[ThreadPoolExecutor] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def load(self, weights_path: str, device: str, max_workers: int = 2) -> None:
        """
        Load model weights synchronously.  Called once from FastAPI startup.

        Sets _ready = True on success, _error = <message> on failure.
        Never raises — callers check readiness via is_ready() / get_status().
        """
        from app.schemas.analysis import ModelReadiness
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="aasist-infer",
        )
        logger.info("Loading AASIST model from '%s' on device '%s' …", weights_path, device)
        try:
            import torch
            from models.AASIST.model_architecture import AASIST

            self._device = device
            self._model  = AASIST.from_pretrained(weights_path, device)
            self._model.eval()
            self._ready  = True
            logger.info("AASIST model loaded successfully on %s.", device)
        except FileNotFoundError as exc:
            self._error = str(exc)
            logger.warning("AASIST weights not found — model NOT READY.\n%s", exc)
        except Exception as exc:
            self._error = f"{type(exc).__name__}: {exc}"
            logger.error("AASIST failed to load: %s", exc, exc_info=True)

    def shutdown(self) -> None:
        """Release executor on server shutdown."""
        if self._executor:
            self._executor.shutdown(wait=False)
            self._executor = None

    # ── Readiness ─────────────────────────────────────────────────────────────

    def is_ready(self) -> bool:
        return self._ready

    def get_status(self) -> dict:
        from app.schemas.analysis import ModelReadiness
        from app.config import get_settings
        cfg = get_settings()
        return {
            "model_name":    cfg.model_name,
            "model_version": cfg.model_version,
            "readiness":     ModelReadiness.READY if self._ready else ModelReadiness.NOT_READY,
            "device":        self._device,
            "weights_path":  cfg.model_weights_path,
            "supported_sample_rate": cfg.model_sample_rate,
            "supported_channels":    1,
            "min_audio_duration_seconds": cfg.min_audio_duration_seconds,
            "max_audio_duration_seconds": cfg.max_audio_duration_seconds,
            "error_detail":  self._error,
        }

    # ── Synchronous inference (runs in thread pool) ───────────────────────────

    def _infer_sync(self, tensor: "torch.Tensor") -> InferenceResult:
        """
        Run a single forward pass.  Intended to be called inside the executor.

        Args:
            tensor: [1, T] float32 on the model's device.
        Returns:
            InferenceResult with raw_score, llr_score, latency_ms.
        Raises:
            RuntimeError: on any torch error during inference.
        """
        import torch
        import torch.nn.functional as F

        t0 = time.perf_counter()
        with torch.no_grad():
            _, logits = self._model(tensor)   # logits: [1, 2]

        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Softmax bonafide probability
        probs      = F.softmax(logits, dim=-1)
        raw_score  = float(probs[0, 1].item())

        # Log-likelihood ratio (spoof=0, bonafide=1)
        llr_score  = float(logits[0, 1].item() - logits[0, 0].item())

        return InferenceResult(
            raw_score=raw_score,
            llr_score=llr_score,
            latency_ms=latency_ms,
        )

    # ── Async inference (awaitable, non-blocking) ─────────────────────────────

    async def infer_async(self, tensor: "torch.Tensor") -> InferenceResult:
        """
        Run inference asynchronously in the thread pool.

        Raises:
            RuntimeError: if the model is not ready.
        """
        if not self._ready:
            raise RuntimeError(
                "MODEL NOT READY — AASIST weights have not been loaded. "
                "Check /api/model/status for details."
            )
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(self._executor, self._infer_sync, tensor)
        return result


# ─── Singleton ────────────────────────────────────────────────────────────────

_model_service: Optional[ModelService] = None


def get_model_service() -> ModelService:
    """Return the application-wide ModelService singleton."""
    global _model_service
    if _model_service is None:
        _model_service = ModelService()
    return _model_service

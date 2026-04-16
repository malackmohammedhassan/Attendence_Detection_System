"""
Inference routes.

POST   /inference/frame         — Single frame inference (multipart or JSON base64)
POST   /inference/batch         — Batch frame inference
GET    /inference/models         — List available models with load status
POST   /inference/models/{name}/activate — Switch active model
GET    /inference/models/{name}  — Get model metadata
WS     /inference/stream         — Real-time webcam stream inference
"""

from __future__ import annotations

import asyncio
import base64
import logging
from typing import Any, Dict, List, Optional

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from pydantic import BaseModel, Field

from app.config import get_settings
from app.services.inference_service import InferenceResult, inference_service
from app.services.model_manager import model_manager
from app.websocket_manager import ws_manager

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/inference", tags=["inference"])


# ────────────────────────────────────────────── Pydantic schemas ──── #

class FrameBase64Request(BaseModel):
    image_b64: str = Field(..., description="Base64-encoded JPEG or PNG image")
    confidence_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class ActivateModelRequest(BaseModel):
    confidence_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class InferenceResponse(BaseModel):
    model_name: str
    detection_count: int
    detections: List[Dict[str, Any]]
    latency_ms: float
    frame_width: int
    frame_height: int
    confidence_threshold: float
    timestamp: float


def _result_to_response(result: InferenceResult) -> InferenceResponse:
    return InferenceResponse(
        model_name=result.model_name,
        detection_count=len(result.detections),
        detections=[d.as_dict() for d in result.detections],
        latency_ms=result.latency_ms,
        frame_width=result.frame_width,
        frame_height=result.frame_height,
        confidence_threshold=result.confidence_threshold,
        timestamp=result.timestamp,
    )


# ────────────────────────────────────────────── Model management ─── #

@router.get("/models", response_model=List[Dict[str, Any]])
async def list_models() -> List[Dict[str, Any]]:
    """List all registered models with load status and inference stats."""
    return model_manager.list_models()


@router.get("/models/active", response_model=Optional[Dict[str, Any]])
async def get_active_model() -> Optional[Dict[str, Any]]:
    """Return info about the currently active model."""
    name = model_manager.get_active_name()
    if name is None:
        return None
    return model_manager.get_model_info(name)


@router.post("/models/{model_name}/activate", status_code=status.HTTP_200_OK)
async def activate_model(
    model_name: str,
    request: ActivateModelRequest = ActivateModelRequest(),
) -> Dict[str, Any]:
    """Load and switch the active model for inference."""
    # Import here to avoid a module-level circular reference.
    from app.routes.live import inference_broadcaster  # noqa: PLC0415

    # ── Integrity enforcement ──────────────────────────────────────────────
    # When allow_tampered_models=False, block activation of any model whose
    # on-disk SHA-256 no longer matches the registered hash.
    if not settings.allow_tampered_models:
        from app.services.model_registry import model_registry as _reg  # noqa: PLC0415
        integrity = _reg.verify_integrity(model_name)
        if integrity == "tampered":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    f"Model '{model_name}' failed integrity check (SHA-256 mismatch). "
                    "The model file may have been modified after registration. "
                    "Set ALLOW_TAMPERED_MODELS=true env variable to override."
                ),
            )

    try:
        # Acquire the broadcaster's model_op_lock so that a currently-running
        # forward-pass completes before the active model pointer is swapped.
        async with inference_broadcaster.model_op_lock:
            model_manager.set_active(model_name)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
        )

    if request.confidence_threshold is not None:
        inference_service.set_confidence_threshold(request.confidence_threshold)

    return {
        "message": f"Model '{model_name}' is now active.",
        "model_name": model_name,
    }


@router.get("/models/{model_name}", response_model=Dict[str, Any])
async def get_model_info(model_name: str) -> Dict[str, Any]:
    info = model_manager.get_model_info(model_name)
    if info is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_name}' not found.",
        )
    return info


# ────────────────────────────────────────────── Single frame ──────── #

@router.post("/frame", response_model=InferenceResponse)
async def infer_frame_upload(
    file: UploadFile = File(..., description="Image file (JPEG/PNG)"),
    confidence_threshold: Optional[float] = Form(default=None),
) -> InferenceResponse:
    """
    Run inference on an uploaded image file.
    Accepts multipart/form-data with an image file.
    """
    _ensure_active_model()
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )
    try:
        result = await inference_service.infer_from_bytes(image_bytes, confidence_threshold)
    except Exception as exc:
        logger.exception("Inference error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {exc}",
        )
    return _result_to_response(result)


@router.post("/frame/base64", response_model=InferenceResponse)
async def infer_frame_base64(request: FrameBase64Request) -> InferenceResponse:
    """
    Run inference on a base64-encoded image.
    Useful for JavaScript clients sending canvas data.
    """
    _ensure_active_model()
    try:
        # Strip optional data-URI prefix
        b64_data = request.image_b64
        if "," in b64_data:
            b64_data = b64_data.split(",", 1)[1]
        image_bytes = base64.b64decode(b64_data)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid base64 image data.",
        )
    try:
        result = await inference_service.infer_from_bytes(
            image_bytes, request.confidence_threshold
        )
    except Exception as exc:
        logger.exception("Inference error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {exc}",
        )
    return _result_to_response(result)


# ────────────────────────────────────────────── Batch ─────────────── #

@router.post("/batch", response_model=List[InferenceResponse])
async def infer_batch(
    files: List[UploadFile] = File(...),
    confidence_threshold: Optional[float] = Form(default=None),
) -> List[InferenceResponse]:
    """Run inference on a batch of uploaded images (up to max_batch_size)."""
    _ensure_active_model()
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="No files uploaded."
        )
    frames_bytes = [await f.read() for f in files[: settings.inference_max_batch_size]]

    import numpy as np
    loop = asyncio.get_event_loop()
    from concurrent.futures import ThreadPoolExecutor
    _exec = ThreadPoolExecutor(max_workers=4)

    frames = await asyncio.gather(
        *[
            loop.run_in_executor(
                _exec, inference_service._decode_image, fb
            )
            for fb in frames_bytes
        ]
    )
    results = await inference_service.infer_batch(list(frames), confidence_threshold)
    return [_result_to_response(r) for r in results]


# ────────────────────────────────────────────── WebSocket stream ─── #

@router.websocket("/stream")
async def inference_stream(websocket: WebSocket) -> None:
    """
    Real-time inference WebSocket.

    Client sends binary frames (JPEG bytes).
    Server responds with JSON inference results.

    Protocol:
        client → binary:  raw JPEG frame bytes
        server → json:    InferenceResponse payload
        client → text:    {"cmd": "set_threshold", "value": 0.5}  (optional control)
    """
    conn = await ws_manager.connect(websocket, channel="inference-stream")

    if model_manager.get_active_name() is None:
        await conn.send_json({
            "type": "error",
            "message": "No active model. Activate a model via POST /inference/models/{name}/activate first."
        })
        await ws_manager.disconnect(conn)
        return

    await conn.send_json({"type": "ready", "model": model_manager.get_active_name()})

    try:
        while True:
            message = await websocket.receive()

            if message["type"] == "websocket.disconnect":
                break

            if "bytes" in message and message["bytes"]:
                # Binary frame → run inference → reply with JSON
                try:
                    result = await inference_service.infer_from_bytes(message["bytes"])
                    response = {
                        "type": "inference_result",
                        **_result_to_response(result).model_dump(),
                    }
                    await conn.send_json(response)
                except Exception as exc:
                    await conn.send_json({"type": "error", "message": str(exc)})

            elif "text" in message and message["text"]:
                # Control message
                import json
                try:
                    ctrl = json.loads(message["text"])
                    if ctrl.get("cmd") == "set_threshold":
                        inference_service.set_confidence_threshold(float(ctrl["value"]))
                        await conn.send_json({"type": "ack", "threshold": ctrl["value"]})
                    elif ctrl.get("cmd") == "ping":
                        await conn.send_json({"type": "pong"})
                except Exception as exc:
                    await conn.send_json({"type": "error", "message": str(exc)})

    except WebSocketDisconnect:
        pass
    finally:
        await ws_manager.disconnect(conn)


# ────────────────────────────────────────────── Helpers ───────────── #

def _ensure_active_model() -> None:
    if model_manager.get_active_name() is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="No active model set. Activate a model via POST /inference/models/{name}/activate",
        )

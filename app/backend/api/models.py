"""
Model selection API endpoint.

GET  /models               - list available models
GET  /models/active        - get active model info
POST /models/active        - set active model (runtime switch)
"""
from __future__ import annotations

from fastapi import APIRouter, Request, HTTPException

router = APIRouter()


@router.get("/models")
async def list_models(request: Request):
    """List semua model yang tersedia di registry."""
    registry = getattr(request.app.state, "registry", None)
    if registry is None:
        return {"models": [], "active_model_id": "mobilefacenet-pretrained"}

    return {
        "models": registry.list_available(),
        "active_model_id": getattr(request.app.state, "active_model_id", "mobilefacenet-pretrained"),
    }


@router.get("/models/active")
async def get_active_model(request: Request):
    """Get info tentang model yang sedang aktif."""
    active_id = getattr(request.app.state, "active_model_id", "mobilefacenet-pretrained")
    registry = getattr(request.app.state, "registry", None)

    if registry and registry.is_available(active_id):
        runtime = registry.get(active_id)
        return {
            "model_id": active_id,
            "name": runtime.name,
            "version": runtime.version,
            "training_mode": runtime.training_mode,
            "threshold": runtime.threshold,
            "metrics": runtime.metrics,
        }
    return {
        "model_id": active_id,
        "name": active_id,
        "version": "unknown",
        "training_mode": "unknown",
        "threshold": 0.50,
        "metrics": {},
    }


@router.post("/models/active")
async def set_active_model(
    request: Request,
    body: dict,
):
    """Switch active model untuk identification/enrollment.

    Body: {"model_id": "palmnet-lite-scratch"}

    Note: Ini mengubah model runtime hanya untuk session ini.
    Embeddings antar model TIDAK dicampur — jika model berganti,
    enrollment ulang diperlukan untuk template yang kompatibel.
    """
    model_id = body.get("model_id")
    if not model_id:
        raise HTTPException(status_code=422, detail="model_id diperlukan")

    registry = getattr(request.app.state, "registry", None)
    if registry is None:
        raise HTTPException(status_code=503, detail="ModelRegistry tidak tersedia")

    if not registry.is_available(model_id):
        available = [m["id"] for m in registry.list_available()]
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_id}' tidak tersedia. Available: {available}"
        )

    runtime = registry.get(model_id)
    request.app.state.recognizer = runtime
    request.app.state.active_model_id = model_id

    return {
        "ok": True,
        "active_model_id": model_id,
        "name": runtime.name,
        "version": runtime.version,
        "training_mode": runtime.training_mode,
        "threshold": runtime.threshold,
        "warning": (
            "Perubahan model membutuhkan enrollment ulang jika template lama "
            "dihasillkan dari model berbeda."
        ),
    }

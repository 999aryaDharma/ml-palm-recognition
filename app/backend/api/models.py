"""
Model selection discovery API endpoints.

GET  /models             - list available models in registry
GET  /models/{model_id}  - get metadata for specific model
"""
from __future__ import annotations

from fastapi import APIRouter, Request, HTTPException

router = APIRouter()


@router.get("/models")
async def list_models(request: Request):
    """List all available models in the registry."""
    registry = getattr(request.app.state, "registry", None)
    default_id = getattr(request.app.state.settings, "default_model_id", "mobilefacenet-pretrained")

    if registry is None:
        return {"models": [], "default_model_id": default_id}

    return {
        "models": registry.list_available(),
        "default_model_id": default_id,
    }


@router.get("/models/{model_id}")
async def get_model_details(model_id: str, request: Request):
    """Get metadata for a specific model by model_id."""
    registry = getattr(request.app.state, "registry", None)
    if registry is None or not registry.is_available(model_id):
        raise HTTPException(
            status_code=404,
            detail={"error": "model_not_found", "message": f"Model '{model_id}' tidak ditemukan di registry."}
        )

    runtime = registry.get(model_id)
    return {
        "model_id": runtime.model_id,
        "name": runtime.name,
        "version": runtime.version,
        "training_mode": runtime.training_mode,
        "threshold": runtime.threshold,
        "manifest": runtime.manifest,
        "metrics": runtime.metrics,
    }

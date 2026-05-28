from fastapi import APIRouter, Request

router = APIRouter()


@router.get("")
def health(request: Request):
    """
    Health check endpoint.
    Returns backend readiness: model load status and cached user count.
    """
    state = request.app.state

    detector   = getattr(state, "detector",   None)
    recognizer = getattr(state, "recognizer", None)
    cache      = getattr(state, "cache",      None)

    # A model is "loaded" when its internal model attribute is not None
    # (stub implementations set self.model = None until a real .pt file is present)
    detector_loaded   = detector   is not None and getattr(detector,   "model", None) is not None
    recognizer_loaded = recognizer is not None and getattr(recognizer, "model", None) is not None

    return {
        "status": "ok",
        "model_loaded":    recognizer_loaded,
        "detector_loaded": detector_loaded,
        "cached_users":    cache.user_count if cache else 0,
    }
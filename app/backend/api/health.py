from fastapi import APIRouter, Request

router = APIRouter()


@router.get("")
def health(request: Request):
    """Health check endpoint - returns status of backend services."""
    detector = getattr(request.app.state, "detector", None)
    recognizer = getattr(request.app.state, "recognizer", None)
    cache = getattr(request.app.state, "cache", None)

    return {
        "status": "ok",
        "model_loaded": recognizer is not None,
        "detector_loaded": detector is not None,
        "cached_users": cache.user_count if cache else 0,
    }

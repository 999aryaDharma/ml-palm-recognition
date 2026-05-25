from fastapi import APIRouter, Request

router = APIRouter()


@router.get("")
def health(request: Request):
    cache = getattr(request.app.state, "cache", None)
    recognizer = getattr(request.app.state, "recognizer", None)
    detector = getattr(request.app.state, "detector", None)

    return {
        "status": "ok",
        "model_loaded": recognizer is not None,
        "detector_loaded": detector is not None,
        "cached_users": cache.user_count if cache else 0,
    }

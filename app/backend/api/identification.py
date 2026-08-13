from fastapi import APIRouter, Request, UploadFile, File, Form, HTTPException, Depends
from sqlalchemy.orm import Session
from db.database import get_db
from schemas.identification import IdentifyResponse
from services.image_service import upload_to_pil
from services.identification_service import IdentificationService

router = APIRouter()


@router.post("/identify", response_model=IdentifyResponse)
async def identify_palm(
    request: Request,
    image: UploadFile = File(...),
    model_id: str | None = Form(None),
    db: Session = Depends(get_db),
):
    """Identify user from palm image using per-request model_id selection."""
    pil_image = await upload_to_pil(image, request.app.state.settings.max_upload_mb)

    try:
        service = IdentificationService(request.app.state, db, model_id=model_id)
        result, latency_ms = service.identify_palm(pil_image)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail={"error": "model_not_found", "message": f"Model '{model_id}' tidak ditemukan di registry."}
        )


    if result["status"] == "error":
        return IdentifyResponse(
            status="error",
            error_code=result.get("error_code", "detection_failed"),
            message=_error_message(result.get("error_code", "detection_failed")),
            bbox=result.get("bbox"),
            landmarks=result.get("landmarks"),
            score=0.0,
            latency_ms=latency_ms,
            model_id=result.get("model_id"),
            model_version=result.get("model_version"),
        )

    user = None
    if result["status"] == "identified":
        from db.repositories import UserRepository
        from api.users import _to_user_response
        user_repo = UserRepository(db)
        u = user_repo.get(result["user_id"])
        if u:
            user = _to_user_response(u)

    return IdentifyResponse(
        status=result["status"],
        user=user,
        score=round(result["score"], 4),
        latency_ms=latency_ms,
        bbox=result.get("bbox"),
        landmarks=result.get("landmarks"),
        quality_score=result.get("quality_score"),
        model_id=result.get("model_id"),
        model_version=result.get("model_version"),
    )


def _error_message(code: str) -> str:
    messages = {
        "no_hand_detected":     "Tunjukkan telapak tangan ke kamera.",
        "detection_failed":     "Telapak belum terbaca. Pastikan tangan terlihat penuh.",
        "landmarks_occluded":   "Pastikan seluruh jari dan telapak terlihat.",
        "palm_facing_wrong":    "Hadapkan telapak tangan ke kamera.",
        "hand_too_small":       "Dekatkan tangan ke kamera.",
        "fingers_not_open":     "Buka jari sedikit lebih lebar.",
        "image_too_blurry":     "Tahan tangan diam sebentar.",
        "roi_extraction_failed":"Posisikan telapak di tengah frame.",
        "no_templates_enrolled":"Belum ada template terdaftar untuk model ini. Lakukan enrollment terlebih dahulu.",
        "not_enough_templates": "Template biometrik belum lengkap. Selesaikan enrollment terlebih dahulu.",
    }
    return messages.get(code, "Gagal memproses telapak. Coba lagi.")

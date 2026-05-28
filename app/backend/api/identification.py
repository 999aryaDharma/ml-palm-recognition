from fastapi import APIRouter, Request, UploadFile, File, HTTPException
from sqlalchemy.orm import Session
from fastapi import Depends
from db.database import get_db
from schemas.identification import IdentifyResponse, IdentifiedUser
from services.image_service import upload_to_pil
from services.identification_service import IdentificationService

router = APIRouter()


@router.post("/identify", response_model=IdentifyResponse)
async def identify_palm(
    request: Request,
    image: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    Identify user from palm image.

    Returns identified user with score and latency, or unknown status.
    HTTP 400 is raised for quality/detection errors with a structured error body.
    """
    pil_image = await upload_to_pil(image, request.app.state.settings.max_upload_mb)

    service = IdentificationService(request.app.state, db)
    result, latency_ms = service.identify_palm(pil_image)

    # ML / quality failures surface as structured 400 errors
    if result["status"] == "error":
        raise HTTPException(
            status_code=400,
            detail={
                "error": result.get("error_code", "detection_failed"),
                "message": _error_message(result.get("error_code", "detection_failed")),
            },
        )

    user = None
    if result["status"] == "identified":
        user = IdentifiedUser(id=result["user_id"], name=result["user_name"])

    return IdentifyResponse(
        status=result["status"],
        user=user,
        score=round(result["score"], 4),
        latency_ms=latency_ms,
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
        "no_templates_enrolled":"Belum ada template terdaftar. Lakukan enrollment terlebih dahulu.",
    }
    return messages.get(code, "Gagal memproses telapak. Coba lagi.")  
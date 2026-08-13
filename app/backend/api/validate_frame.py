"""
api/validate_frame.py — Quality gate sebelum enrollment template disimpan.

Pipeline HARUS identik dengan add_template:
    parse image → hand detection → ROI extraction → embedding extraction

Kalau lolos validate-frame, dijamin lolos add_template.
Tidak menyimpan apapun ke database.

POST /validate-frame  — cek kualitas frame telapak (full pipeline)
"""
from fastapi import APIRouter, Request, UploadFile, File, HTTPException, Form
from services.enrollment_service import EnrollmentService
from services.image_service import upload_to_pil
from ml.roi import extract_palm_roi

router = APIRouter()


@router.post("/validate-frame")
async def validate_frame(
    request: Request,
    image: UploadFile = File(...),
    model_id: str | None = Form(None),
):
    settings = request.app.state.settings

    pil_image = await upload_to_pil(
        image,
        settings.max_upload_mb,
    )

    try:
        service = EnrollmentService(
            request.app.state,
            model_id=model_id,
        )

        (
            embedding,
            quality_score,
            quality_status,
            target_model_id,
            target_version,
        ) = service.process_template(pil_image)

    except KeyError:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "model_not_found",
                "message": f"Model '{model_id}' tidak ditemukan.",
            },
        )

    except ValueError as exc:
        code = str(exc)

        messages = {
            "detection_failed": "Telapak belum terbaca.",
            "no_hand_detected": "Tunjukkan telapak tangan ke kamera.",
            "roi_extraction_failed": "Posisikan telapak di tengah frame.",
            "image_too_blurry": "Tahan tangan diam sebentar.",
        }

        return {
            "status": "error",
            "error": code,
            "message": messages.get(
                code,
                "Frame tidak valid.",
            ),
        }

    return {
        "valid": True,
        "model_id": target_model_id,
        "model_version": target_version,
        "quality_score": quality_score,
        "quality_status": quality_status,
    }
"""Enrollment quality gate without persistence.

When no model_id is supplied (normal enrollment flow), one shared ROI is validated
against every active registry model. Supplying model_id preserves single-model
validation for debugging/comparison flows.
"""
from fastapi import APIRouter, Request, UploadFile, File, HTTPException, Form

from services.enrollment_service import EnrollmentService
from services.image_service import upload_to_pil

router = APIRouter()


@router.post("/validate-frame")
async def validate_frame(
    request: Request,
    image: UploadFile = File(...),
    model_id: str | None = Form(None),
):
    settings = request.app.state.settings
    pil_image = await upload_to_pil(image, settings.max_upload_mb)

    try:
        service = EnrollmentService(request.app.state, model_id=model_id)

        if model_id:
            (
                _embedding,
                quality_score,
                quality_status,
                target_model_id,
                target_version,
            ) = service.process_template(pil_image)
            models = [
                {
                    "model_id": target_model_id,
                    "model_version": target_version,
                }
            ]
        else:
            model_results, quality_score, quality_status = service.process_template_all(
                pil_image
            )
            models = [
                {
                    "model_id": item["model_id"],
                    "model_version": item["model_version"],
                }
                for item in model_results
            ]

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
            "backend_not_ready": "Model biometrik belum siap.",
        }
        return {
            "status": "error",
            "error": code,
            "message": messages.get(code, "Frame tidak valid."),
        }

    return {
        "valid": True,
        "quality_score": quality_score,
        "quality_status": quality_status,
        "models": models,
    }

"""
api/validate_frame.py — Quality gate sebelum enrollment template disimpan.

Pipeline HARUS identik dengan add_template:
    parse image → hand detection → ROI extraction → embedding extraction

Kalau lolos validate-frame, dijamin lolos add_template.
Tidak menyimpan apapun ke database.

POST /validate-frame  — cek kualitas frame telapak (full pipeline)
"""
from fastapi import APIRouter, Request, UploadFile, File, HTTPException

from services.image_service import upload_to_pil
from ml.roi import extract_palm_roi

router = APIRouter()


@router.post("/validate-frame")
async def validate_frame(
    request: Request,
    image: UploadFile = File(...),
):
    """
    Validasi kualitas frame telapak tangan tanpa menyimpan apapun.

    Menjalankan pipeline LENGKAP: parse → detection → ROI → embedding.
    Pipeline IDENTIK dengan add_template, sehingga frame yang lolos
    validasi ini PASTI bisa disimpan sebagai template (tidak ada false-positive).

    Response 200: frame valid, siap untuk enrollment
    Response 400: frame tidak valid dengan error code untuk hint UI
    """
    settings   = request.app.state.settings
    detector   = getattr(request.app.state, "detector",   None)
    recognizer = getattr(request.app.state, "recognizer", None)

    if detector is None or recognizer is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "backend_not_ready",
                "message": "Backend belum siap. Tunggu server selesai startup.",
            },
        )

    pil_image = await upload_to_pil(image, settings.max_upload_mb)

    # ── Stage 1: Hand detection ───────────────────────────────────────────────
    detection = detector.detect(pil_image)
    if detection is None:
        return {
            "status": "error",
            "error": "detection_failed",
            "message": "Telapak belum terbaca. Pastikan tangan terlihat penuh dan menghadap kamera.",
        }

    # ── Stage 2: ROI extraction ───────────────────────────────────────────────
    roi = extract_palm_roi(pil_image, detection["landmarks"])
    if roi is None:
        return {
            "status": "error",
            "error": "roi_extraction_failed",
            "message": "Area telapak gagal diekstrak. Posisikan telapak di tengah frame.",
        }

    # ── Stage 3: Embedding extraction ────────────────────────────────────────
    # KRITIS: Harus identik dengan add_template agar tidak ada false-positive.
    # Tanpa pengecekan ini, blob yang lolos validate bisa gagal di add_template
    # karena gambar blur atau landmark tidak cukup jelas untuk model.
    embedding = recognizer.extract_embedding(roi)
    if embedding is None:
        return {
            "status": "error",
            "error": "image_too_blurry",
            "message": "Gambar terlalu blur atau tangan kurang jelas. Tahan tangan diam sebentar.",
        }

    return {
        "valid": True,
        "message": "Frame valid. Siap untuk enrollment.",
    }
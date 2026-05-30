"""
api/validate_frame.py — Lightweight quality gate check.

Endpoint ini menerima gambar telapak dan menjalankan detection + ROI pipeline,
tapi TIDAK menyimpan apapun ke database. Hanya mengembalikan apakah frame
memenuhi quality gate atau tidak.

Dipakai oleh frontend enrollment SEBELUM createUser() dipanggil,
sehingga user tidak terbuat di backend dengan 0 template.

GET  /validate-frame  — tidak ada (hanya POST)
POST /validate-frame  — cek kualitas frame telapak
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

    Menjalankan pipeline: parse image → hand detection → ROI extraction.
    Tidak memanggil recognizer (tidak perlu embedding untuk validasi awal).

    Response 200: frame valid, siap untuk enrollment
    Response 400: frame tidak valid dengan error code untuk hint UI
    """
    settings = request.app.state.settings
    detector = getattr(request.app.state, "detector", None)

    if detector is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "backend_not_ready",
                "message": "Backend belum siap. Tunggu server selesai startup.",
            },
        )

    pil_image = await upload_to_pil(image, settings.max_upload_mb)

    # Detection
    detection = detector.detect(pil_image)
    if detection is None:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "detection_failed",
                "message": "Telapak belum terbaca. Pastikan tangan terlihat penuh dan menghadap kamera.",
            },
        )

    # ROI extraction
    roi = extract_palm_roi(pil_image, detection["landmarks"])
    if roi is None:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "roi_extraction_failed",
                "message": "Area telapak gagal diekstrak. Posisikan telapak di tengah frame.",
            },
        )

    return {
        "valid": True,
        "message": "Frame valid. Siap untuk enrollment.",
    }
from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File
from sqlalchemy.orm import Session
import numpy as np

from db.database import get_db
from db.repositories import UserRepository, TemplateRepository
from schemas.users import UserCreateRequest, UserResponse, DeleteUserResponse, TemplateCreateResponse
from services.image_service import upload_to_pil
from schemas.users import (
    UserCreateRequest, 
    UserResponse, 
    DeleteUserResponse, 
    TemplateCreateResponse,
    VerifyReadyResponse  # <--- Tambahkan import ini
)

router = APIRouter()


def _to_user_response(user) -> UserResponse:
    return UserResponse(
        id=user.id,
        name=user.name,
        enrolled_at=user.enrolled_at,
        template_count=len(user.templates or []),
    )


@router.post("", response_model=UserResponse)
def create_user(payload: UserCreateRequest, db: Session = Depends(get_db)):
    repo = UserRepository(db)
    
    # ── MENCEGAH DUPLIKAT NAMA & CLEANUP GHOST USER ──
    existing_users = repo.list_all()
    for u in existing_users:
        # Pengecekan tidak sensitif terhadap huruf besar/kecil dan spasi
        if u.name.strip().lower() == payload.name.strip().lower():
            # Jika user ada tapi jumlah template 0 (kemungkinan sisa error enrollment sebelumnya)
            # Hapus user lama yang cacat agar tidak nyangkut
            if len(u.templates or []) == 0:
                repo.delete(u.id)
            else:
                # Jika user ada dan VALID, tolak dengan HTTP 409 Conflict
                raise HTTPException(
                    status_code=409,
                    detail={"error": "user_exists", "message": f"Pengguna dengan nama '{payload.name}' sudah terdaftar."}
                )
                
    # Buat user baru jika aman
    user = repo.create(payload.name)
    user.templates = []
    return _to_user_response(user)


@router.get("", response_model=list[UserResponse])
def list_users(db: Session = Depends(get_db)):
    repo = UserRepository(db)
    return [_to_user_response(u) for u in repo.list_all()]


@router.get("/{user_id}", response_model=UserResponse)
def get_user(user_id: int, db: Session = Depends(get_db)):
    repo = UserRepository(db)
    user = repo.get(user_id)
    if not user:
        raise HTTPException(
            status_code=404,
            detail={"error": "user_not_found", "message": "User tidak ditemukan."},
        )
    return _to_user_response(user)


@router.delete("/{user_id}", response_model=DeleteUserResponse)
def delete_user(user_id: int, request: Request, db: Session = Depends(get_db)):
    repo = UserRepository(db)
    deleted = repo.delete(user_id)
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail={"error": "user_not_found", "message": "User tidak ditemukan."},
        )

    # ── Fix: refresh cache so deleted user is no longer matched ────────────
    cache = getattr(request.app.state, "cache", None)
    if cache is not None:
        cache.refresh(db)

    return DeleteUserResponse(deleted=True)


@router.post("/{user_id}/templates", response_model=TemplateCreateResponse)
async def add_template(
    user_id: int,
    request: Request,
    image: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    Upload one palm frame as a biometric template.
    Runs detection → ROI → embedding pipeline.
    Returns 400 with error code if quality gate fails.
    """
    user_repo = UserRepository(db)
    if not user_repo.get(user_id):
        raise HTTPException(
            status_code=404,
            detail={"error": "user_not_found", "message": "User tidak ditemukan."},
        )

    pil_image = await upload_to_pil(image, request.app.state.settings.max_upload_mb)

    # ── Run ML pipeline ───────────────────────────────────────────────────────
    detector   = getattr(request.app.state, "detector",   None)
    recognizer = getattr(request.app.state, "recognizer", None)

    if detector is None or recognizer is None:
        raise HTTPException(
            status_code=503,
            detail={"error": "backend_not_ready", "message": "ML model belum dimuat. Tunggu server selesai startup."},
        )

    detection = detector.detect(pil_image)
    if detection is None:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "detection_failed",
                "message": "Telapak belum terbaca. Pastikan tangan terlihat penuh dan menghadap kamera.",
            },
        )

    from ml.roi import extract_palm_roi
    roi = extract_palm_roi(pil_image, detection["landmarks"])
    if roi is None:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "roi_extraction_failed",
                "message": "Area telapak gagal diekstrak. Posisikan telapak di tengah frame.",
            },
        )

    embedding = recognizer.extract_embedding(roi)
    if embedding is None:
        raise HTTPException(
            status_code=400,
            detail={"error": "image_too_blurry", "message": "Gambar terlalu blur. Tahan tangan diam sebentar."},
        )

    quality_score = float(min(1.0, np.linalg.norm(embedding)))

    template_repo = TemplateRepository(db)
    template = template_repo.create(user_id, embedding, quality_score)

    # ── Refresh cache dengan session BARU (bukan session request) ─────────────────
    # Session request (db) masih hold transaction yang baru commit.
    # Menggunakan session yang sama untuk cache.refresh bisa menyebabkan
    # SQLite lock, terutama saat 5 template diupload berurutan.
    cache = getattr(request.app.state, "cache", None)
    if cache is not None:
        try:
            from db.database import SessionLocal
            fresh_db = SessionLocal()
            try:
                cache.refresh(fresh_db)
            finally:
                fresh_db.close()
        except Exception as exc:
            import logging
            logging.getLogger("palm-api").error(
                "Cache refresh failed after template upload: %s", exc
            )

    return TemplateCreateResponse(
        template_id=template.id,
        quality_score=round(quality_score, 4),
        embedding_norm=round(float(np.linalg.norm(embedding)), 4),
    )

@router.get("/{user_id}/verify-ready", response_model=VerifyReadyResponse)
def verify_ready(user_id: int, db: Session = Depends(get_db)):
    repo = UserRepository(db)
    user = repo.get(user_id)
    
    if not user:
        raise HTTPException(
            status_code=404,
            detail={"error": "user_not_found", "message": "User tidak ditemukan."},
        )

    # Hitung jumlah template yang benar-benar tersimpan di database
    template_count = len(user.templates or [])
    required_templates = 5
    
    return VerifyReadyResponse(
        ready=(template_count >= required_templates),
        template_count=template_count,
        required=required_templates
    )
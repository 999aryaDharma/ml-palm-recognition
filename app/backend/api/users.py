from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File, Form
from sqlalchemy.orm import Session
import numpy as np

from db.database import get_db
from db.repositories import UserRepository, TemplateRepository
from schemas.users import (
    UserCreateRequest,
    UserResponse,
    DeleteUserResponse,
    TemplateCreateResponse,
    VerifyReadyResponse,
    UserProfileRequest,
    UserProfileResponse,
    WalletResponse
)
from db.models import UserProfile, Wallet
from services.image_service import upload_to_pil
from services.enrollment_service import EnrollmentService

router = APIRouter()


def _to_user_response(user) -> UserResponse:
    profile = None
    if user.profile:
        profile = UserProfileResponse(
            nik=user.profile.nik,
            kelas_jabatan=user.profile.kelas_jabatan
        )
    wallet = None
    if getattr(user, 'wallet', None):
        wallet = WalletResponse(balance=user.wallet.balance)

    return UserResponse(
        id=user.id,
        name=user.name,
        enrolled_at=user.enrolled_at,
        template_count=len(user.templates or []),
        profile=profile,
        wallet=wallet
    )


@router.post("/{user_id}/profile", response_model=UserResponse)
def add_profile(user_id: int, payload: UserProfileRequest, db: Session = Depends(get_db)):
    repo = UserRepository(db)
    user = repo.get(user_id)
    if not user:
        raise HTTPException(status_code=404, detail={"error": "user_not_found", "message": "User tidak ditemukan."})

    if not user.profile:
        user.profile = UserProfile(user_id=user_id)
    user.profile.nik = payload.nik
    user.profile.kelas_jabatan = payload.kelas_jabatan

    if not user.wallet:
        user.wallet = Wallet(user_id=user_id)
    user.wallet.balance = payload.initial_balance

    db.commit()
    db.refresh(user)

    return _to_user_response(user)


@router.post("", response_model=UserResponse)
def create_user(payload: UserCreateRequest, db: Session = Depends(get_db)):
    repo = UserRepository(db)

    existing_users = repo.list_all()
    for u in existing_users:
        if u.name.strip().lower() == payload.name.strip().lower():
            if len(u.templates or []) == 0:
                repo.delete(u.id)
            else:
                raise HTTPException(
                    status_code=409,
                    detail={"error": "user_exists", "message": f"Pengguna dengan nama '{payload.name}' sudah terdaftar."}
                )

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

    cache = getattr(request.app.state, "cache", None)
    if cache is not None:
        cache.refresh(db)

    return DeleteUserResponse(deleted=True)


@router.post("/{user_id}/templates", response_model=TemplateCreateResponse)
async def add_template(
    user_id: int,
    request: Request,
    image: UploadFile = File(...),
    model_id: str | None = Form(None),
    db: Session = Depends(get_db),
):
    """Upload one palm frame as a biometric template using model_id provenance."""
    user_repo = UserRepository(db)
    if not user_repo.get(user_id):
        raise HTTPException(
            status_code=404,
            detail={"error": "user_not_found", "message": "User tidak ditemukan."},
        )

    pil_image = await upload_to_pil(image, request.app.state.settings.max_upload_mb)

    service = EnrollmentService(request.app.state, model_id=model_id)

    try:
        embedding, quality_score, quality_status, target_model_id, target_version = service.process_template(pil_image)
    except ValueError as exc:
        code = str(exc)
        messages = {
            "detection_failed": "Telapak belum terbaca. Pastikan tangan terlihat penuh.",
            "no_hand_detected": "Tunjukkan telapak tangan ke kamera.",
            "roi_extraction_failed": "Posisikan telapak di tengah frame.",
            "image_too_blurry": "Gambar terlalu blur. Tahan tangan diam sebentar.",
        }
        raise HTTPException(
            status_code=400,
            detail={"error": code, "message": messages.get(code, "Gagal memproses template.")},
        )

    template_repo = TemplateRepository(db)
    template = template_repo.create(
        user_id=user_id,
        embedding=embedding,
        quality_score=quality_score,
        model_id=target_model_id,
        model_version=target_version,
    )

    # Refresh cache for this specific model space
    cache = getattr(request.app.state, "cache", None)
    if cache is not None:
        try:
            from db.database import SessionLocal
            fresh_db = SessionLocal()
            try:
                cache.refresh(fresh_db, model_id=target_model_id, model_version=target_version)
            finally:
                fresh_db.close()
        except Exception as exc:
            import logging
            logging.getLogger("palm-api").error("Cache refresh failed after template upload: %s", exc)

    return TemplateCreateResponse(
        template_id=template.id,
        quality_score=round(quality_score, 4),
        embedding_norm=round(float(np.linalg.norm(embedding)), 4),
    )


@router.get("/{user_id}/verify-ready", response_model=VerifyReadyResponse)
def verify_ready(
    user_id: int,
    model_id: str | None = None,
    db: Session = Depends(get_db),
):
    repo = UserRepository(db)
    user = repo.get(user_id)

    if not user:
        raise HTTPException(
            status_code=404,
            detail={"error": "user_not_found", "message": "User tidak ditemukan."},
        )

    template_repo = TemplateRepository(db)
    templates = template_repo.list_by_user(user_id, model_id=model_id)
    template_count = len(templates)
    required_templates = 5

    return VerifyReadyResponse(
        ready=(template_count >= required_templates),
        template_count=template_count,
        required=required_templates
    )
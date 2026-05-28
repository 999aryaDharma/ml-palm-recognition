from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File
from sqlalchemy.orm import Session
import numpy as np

from db.database import get_db
from db.repositories import UserRepository, TemplateRepository
from schemas.users import UserCreateRequest, UserResponse, DeleteUserResponse, TemplateCreateResponse
from services.image_service import upload_to_pil

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

    # ── Refresh cache so new template is available for matching ───────────────
    cache = getattr(request.app.state, "cache", None)
    if cache is not None:
        cache.refresh(db)

    return TemplateCreateResponse(
        template_id=template.id,
        quality_score=round(quality_score, 4),
        embedding_norm=round(float(np.linalg.norm(embedding)), 4),
    )
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
    MultiModelTemplateCreateResponse,
    MultiModelTemplateItem,
    VerifyReadyResponse,
    VerifyReadyAllResponse,
    ModelReadinessResponse,
    UserProfileRequest,
    UserProfileResponse,
    WalletResponse,
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
            kelas_jabatan=user.profile.kelas_jabatan,
        )
    wallet = None
    if getattr(user, "wallet", None):
        wallet = WalletResponse(balance=user.wallet.balance)

    return UserResponse(
        id=user.id,
        name=user.name,
        enrolled_at=user.enrolled_at,
        template_count=len(user.templates or []),
        profile=profile,
        wallet=wallet,
    )


def _quality_error(code: str) -> HTTPException:
    messages = {
        "detection_failed": "Telapak belum terbaca. Pastikan tangan terlihat penuh.",
        "no_hand_detected": "Tunjukkan telapak tangan ke kamera.",
        "roi_extraction_failed": "Posisikan telapak di tengah frame.",
        "image_too_blurry": "Gambar terlalu blur. Tahan tangan diam sebentar.",
        "backend_not_ready": "Model biometrik belum siap.",
    }
    return HTTPException(
        status_code=400,
        detail={"error": code, "message": messages.get(code, "Gagal memproses template.")},
    )


def _refresh_model_caches(request: Request, model_pairs: set[tuple[str, str]]) -> None:
    cache = getattr(request.app.state, "cache", None)
    if cache is None:
        return

    try:
        from db.database import SessionLocal

        fresh_db = SessionLocal()
        try:
            for model_id, model_version in model_pairs:
                cache.refresh(
                    fresh_db,
                    model_id=model_id,
                    model_version=model_version,
                )
        finally:
            fresh_db.close()
    except Exception as exc:
        import logging

        logging.getLogger("palm-api").error(
            "Cache refresh failed after template upload: %s",
            exc,
        )


@router.post("/{user_id}/profile", response_model=UserResponse)
def add_profile(user_id: int, payload: UserProfileRequest, db: Session = Depends(get_db)):
    repo = UserRepository(db)
    user = repo.get(user_id)
    if not user:
        raise HTTPException(
            status_code=404,
            detail={"error": "user_not_found", "message": "User tidak ditemukan."},
        )

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
    for user in existing_users:
        if user.name.strip().lower() == payload.name.strip().lower():
            if len(user.templates or []) == 0:
                repo.delete(user.id)
            else:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "error": "user_exists",
                        "message": f"Pengguna dengan nama '{payload.name}' sudah terdaftar.",
                    },
                )

    user = repo.create(payload.name)
    user.templates = []
    if not user.wallet:
        wallet = Wallet(user_id=user.id, balance=500000.0)
        db.add(wallet)
        db.commit()
        db.refresh(user)
    return _to_user_response(user)


@router.get("", response_model=list[UserResponse])
def list_users(db: Session = Depends(get_db)):
    repo = UserRepository(db)
    return [_to_user_response(user) for user in repo.list_all()]


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


@router.post("/{user_id}/templates/multi", response_model=MultiModelTemplateCreateResponse)
async def add_template_multi(
    user_id: int,
    request: Request,
    image: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Create one template per active registry model from one captured frame."""
    user_repo = UserRepository(db)
    if not user_repo.get(user_id):
        raise HTTPException(
            status_code=404,
            detail={"error": "user_not_found", "message": "User tidak ditemukan."},
        )

    pil_image = await upload_to_pil(image, request.app.state.settings.max_upload_mb)

    try:
        service = EnrollmentService(request.app.state)
        model_results, quality_score, quality_status = service.process_template_all(pil_image)
    except ValueError as exc:
        raise _quality_error(str(exc))

    template_repo = TemplateRepository(db)
    templates = template_repo.create_many(
        user_id=user_id,
        items=model_results,
        quality_score=quality_score,
    )

    pairs = {(item["model_id"], item["model_version"]) for item in model_results}
    _refresh_model_caches(request, pairs)

    return MultiModelTemplateCreateResponse(
        quality_score=round(quality_score, 4),
        quality_status=quality_status,
        model_count=len(templates),
        templates=[
            MultiModelTemplateItem(
                template_id=template.id,
                model_id=template.model_id,
                model_version=template.model_version,
                embedding_norm=round(
                    float(np.linalg.norm(model_results[index]["embedding"])),
                    4,
                ),
            )
            for index, template in enumerate(templates)
        ],
    )


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

    try:
        service = EnrollmentService(request.app.state, model_id=model_id)
        embedding, quality_score, _quality_status, target_model_id, target_version = (
            service.process_template(pil_image)
        )
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "model_not_found",
                "message": f"Model '{model_id}' tidak ditemukan di registry.",
            },
        )
    except ValueError as exc:
        raise _quality_error(str(exc))

    template_repo = TemplateRepository(db)
    template = template_repo.create(
        user_id=user_id,
        embedding=embedding,
        quality_score=quality_score,
        model_id=target_model_id,
        model_version=target_version,
    )

    _refresh_model_caches(request, {(target_model_id, target_version)})

    return TemplateCreateResponse(
        template_id=template.id,
        quality_score=round(quality_score, 4),
        embedding_norm=round(float(np.linalg.norm(embedding)), 4),
    )


@router.get("/{user_id}/verify-ready-all", response_model=VerifyReadyAllResponse)
def verify_ready_all(
    user_id: int,
    request: Request,
    db: Session = Depends(get_db),
):
    """Return readiness per active model. All active models must be ready."""
    repo = UserRepository(db)
    user = repo.get(user_id)
    if not user:
        raise HTTPException(
            status_code=404,
            detail={"error": "user_not_found", "message": "User tidak ditemukan."},
        )

    registry = getattr(request.app.state, "registry", None)
    available = registry.list_available() if registry else []
    if not available:
        raise HTTPException(
            status_code=503,
            detail={"error": "backend_not_ready", "message": "Model registry belum siap."},
        )

    required = request.app.state.settings.min_template_per_user
    template_repo = TemplateRepository(db)
    models: dict[str, ModelReadinessResponse] = {}

    for model_info in available:
        model_id = model_info["id"]
        runtime = registry.get(model_id)
        count = len(
            template_repo.list_by_user(
                user_id,
                model_id=runtime.model_id,
                model_version=runtime.version,
            )
        )
        models[model_id] = ModelReadinessResponse(
            model_version=runtime.version,
            template_count=count,
            ready=count >= required,
        )

    return VerifyReadyAllResponse(
        ready=all(item.ready for item in models.values()),
        required=required,
        models=models,
    )


@router.get("/{user_id}/verify-ready", response_model=VerifyReadyResponse)
def verify_ready(
    user_id: int,
    model_id: str | None = None,
    model_version: str | None = None,
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
    templates = template_repo.list_by_user(
        user_id,
        model_id=model_id,
        model_version=model_version,
    )
    template_count = len(templates)
    required_templates = 5

    return VerifyReadyResponse(
        ready=(template_count >= required_templates),
        template_count=template_count,
        required=required_templates,
    )

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File
from sqlalchemy.orm import Session
import numpy as np

from db.database import get_db
from db.repositories import UserRepository, TemplateRepository
from schemas.users import UserCreateRequest, UserResponse, DeleteUserResponse, TemplateCreateResponse
from services.image_service import upload_to_pil
from services.enrollment_service import EnrollmentService

router = APIRouter()


def to_user_response(user) -> UserResponse:
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
    return to_user_response(user)


@router.get("", response_model=list[UserResponse])
def list_users(db: Session = Depends(get_db)):
    repo = UserRepository(db)
    return [to_user_response(user) for user in repo.list_all()]


@router.get("/{user_id}", response_model=UserResponse)
def get_user(user_id: int, db: Session = Depends(get_db)):
    repo = UserRepository(db)
    user = repo.get(user_id)
    if not user:
        raise HTTPException(status_code=404, detail={"error": "user_not_found", "message": "User tidak ditemukan."})
    return to_user_response(user)


@router.delete("/{user_id}", response_model=DeleteUserResponse)
def delete_user(user_id: int, db: Session = Depends(get_db)):
    repo = UserRepository(db)
    deleted = repo.delete(user_id)
    if not deleted:
        raise HTTPException(status_code=404, detail={"error": "user_not_found", "message": "User tidak ditemukan."})
    return DeleteUserResponse(deleted=True)


@router.post("/{user_id}/templates", response_model=TemplateCreateResponse)
async def add_template(
    user_id: int,
    request: Request,
    image: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    Upload palm template for enrollment.
    
    Args:
        user_id: User ID to enroll template for
        image: Palm image file
        
    Returns:
        Template creation response with quality score
    """
    # Check if user exists
    user_repo = UserRepository(db)
    user = user_repo.get(user_id)
    if not user:
        raise HTTPException(status_code=404, detail={"error": "user_not_found", "message": "User tidak ditemukan."})

    try:
        # Validate and convert image
        pil_image = await upload_to_pil(image, request.app.state.settings.max_upload_mb)
        
        # Create enrollment service
        service = EnrollmentService(request.app.state)
        
        # Process template (detect hand, extract ROI, extract embedding)
        try:
            embedding, quality_score, quality_status = service.process_template(pil_image)
        except ValueError as e:
            error_code = str(e)
            raise HTTPException(
                status_code=400,
                detail={
                    "error": error_code,
                    "message": f"Template processing failed: {error_code}"
                }
            )
        
        # Save template to database
        template_repo = TemplateRepository(db)
        template = template_repo.create(
            user_id=user_id,
            embedding=embedding,
            quality_score=quality_score,
        )
        
        # Refresh cache after template added
        cache = getattr(request.app.state, "cache", None)
        if cache:
            cache.refresh(db)
        
        return TemplateCreateResponse(
            template_id=template.id,
            quality_score=template.quality_score,
            embedding_norm=np.linalg.norm(embedding),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"error": "internal_server_error", "message": f"Template upload failed: {str(e)}"}
        )


from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File
from sqlalchemy.orm import Session
import numpy as np

from db.database import get_db
from db.repositories import UserRepository, TemplateRepository
from schemas.users import UserCreateRequest, UserResponse, DeleteUserResponse, TemplateCreateResponse
from services.identification import PalmService

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
    palm_service: PalmService = request.app.state.palm_service
    
    # Check if user exists
    user_repo = UserRepository(db)
    user = user_repo.get(user_id)
    if not user:
        raise HTTPException(status_code=404, detail={"error": "user_not_found", "message": "User tidak ditemukan."})

    try:
        image_bytes = await image.read()
        embedding, quality_score = palm_service.process_image(image_bytes)
        
        template_repo = TemplateRepository(db)
        template = template_repo.create(
            user_id=user_id,
            embedding=embedding,
            quality_score=quality_score,
        )
        
        return TemplateCreateResponse(
            template_id=template.id,
            quality_score=template.quality_score,
            embedding_norm=np.linalg.norm(embedding),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail={"error": "image_processing_failed", "message": str(e)})
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": "internal_server_error", "message": "An unexpected error occurred."})

from pydantic import BaseModel, Field
from datetime import datetime


class UserCreateRequest(BaseModel):
    name: str = Field(..., min_length=2, max_length=120)


class UserResponse(BaseModel):
    id: int
    name: str
    enrolled_at: datetime
    template_count: int = 0


class TemplateCreateResponse(BaseModel):
    template_id: int
    quality_score: float
    embedding_norm: float


class DeleteUserResponse(BaseModel):
    deleted: bool

from pydantic import BaseModel, Field
from datetime import datetime


class UserCreateRequest(BaseModel):
    name: str = Field(..., min_length=2, max_length=120)


class UserProfileRequest(BaseModel):
    nik: str = Field(None, max_length=50)
    kelas_jabatan: str = Field(None, max_length=100)
    initial_balance: float = Field(0.0, ge=0.0)

class UserProfileResponse(BaseModel):
    nik: str | None
    kelas_jabatan: str | None

class WalletResponse(BaseModel):
    balance: float

class UserResponse(BaseModel):
    id: int
    name: str
    enrolled_at: datetime
    template_count: int = 0
    profile: UserProfileResponse | None = None
    wallet: WalletResponse | None = None


class TemplateCreateResponse(BaseModel):
    template_id: int
    quality_score: float
    embedding_norm: float


class DeleteUserResponse(BaseModel):
    deleted: bool

class VerifyReadyResponse(BaseModel):
    ready: bool
    template_count: int
    required: int = 5

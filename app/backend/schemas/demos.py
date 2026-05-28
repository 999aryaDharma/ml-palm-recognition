from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional


# ── Payment ───────────────────────────────────────────────────────────────────

class PaymentPayRequest(BaseModel):
    user_id: int
    amount: int = Field(..., ge=1)
    merchant: str = "Toko Maju Jaya"
    match_score: float = 0.0


class PaymentPayResponse(BaseModel):
    status: str
    transaction_id: str
    user: dict
    amount: int
    merchant: str
    timestamp: datetime


# ── Attendance ────────────────────────────────────────────────────────────────

class AttendanceCheckinRequest(BaseModel):
    user_id: int
    mode: str = Field(default="checkin", pattern="^(checkin|checkout)$")
    match_score: float = 0.0


class AttendanceCheckinResponse(BaseModel):
    status: str
    user: dict
    mode: str
    timestamp: datetime


# ── Access Control ────────────────────────────────────────────────────────────

class AccessCheckResponse(BaseModel):
    status: str          # "granted" | "denied"
    user: Optional[dict] = None
    score: float
    latency_ms: int
    reason: str          # "authorized" | "not_authorized" | "unknown_user"


class AuthorizedUserResponse(BaseModel):
    user_id: int
    name: str
    authorized: bool


# ── Patient ───────────────────────────────────────────────────────────────────

class PatientCheckinRequest(BaseModel):
    user_id: int
    match_score: float = 0.0


class PatientCheckinResponse(BaseModel):
    status: str
    user: dict
    patient: dict
    timestamp: datetime


# ── Demo Logs ─────────────────────────────────────────────────────────────────

class DemoLogResponse(BaseModel):
    id: int
    user: Optional[dict] = None
    demo_type: str
    timestamp: datetime
    match_score: float
    payload: dict
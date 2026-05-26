"""
Demo Schemas
Pydantic models for demo module requests and responses
"""

from pydantic import BaseModel


class PaymentPayRequest(BaseModel):
    """Request body for payment demo."""
    user_id: int
    amount: float
    merchant: str


class PaymentPayResponse(BaseModel):
    """Response for payment demo."""
    transaction_id: str
    amount: float
    merchant: str
    status: str


class AttendanceCheckinRequest(BaseModel):
    """Request body for attendance check-in."""
    user_id: int
    location: str


class AttendanceCheckinResponse(BaseModel):
    """Response for attendance check-in."""
    status: str
    location: str
    user_id: int


class AccessCheckRequest(BaseModel):
    """Request body for access control check."""
    user_id: int
    door_id: str


class AccessCheckResponse(BaseModel):
    """Response for access control check."""
    status: str
    door_id: str
    user_id: int


class PatientCheckinRequest(BaseModel):
    """Request body for patient check-in."""
    user_id: int
    hospital_id: str


class PatientCheckinResponse(BaseModel):
    """Response for patient check-in."""
    status: str
    hospital_id: str
    user_id: int


class DemoLog(BaseModel):
    """Demo log entry."""
    id: int
    user_id: int | None
    demo_type: str
    match_score: float
    timestamp: str


class DemoLogsResponse(BaseModel):
    """Response for demo logs query."""
    logs: list[DemoLog]
    total: int

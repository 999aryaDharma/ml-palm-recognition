from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File
from sqlalchemy.orm import Session
from datetime import datetime
import json

from db.database import get_db
from db.repositories import UserRepository, DemoLogRepository
from schemas.demos import (
    PaymentPayRequest, PaymentPayResponse,
    AttendanceCheckinRequest, AttendanceCheckinResponse,
    AccessCheckResponse,
    PatientCheckinRequest, PatientCheckinResponse,
    AuthorizedUserResponse,
)

router = APIRouter()

# ── In-memory authorized set (reset on restart, fine for demo) ──────────────
_AUTHORIZED_USER_IDS: set[int] = set()

# ── Mock patient data ────────────────────────────────────────────────────────
_PATIENT_DATA: dict[int, dict] = {
    1: {"nik": "3201****0001", "dokter": "dr. Wijaya, Sp.PD",  "jadwal": "Senin 10:00", "last_visit": "12 Apr 2026", "rekam_medik": "RM-2024-0042"},
    2: {"nik": "3271****0002", "dokter": "dr. Putri, Sp.A",    "jadwal": "Senin 11:30", "last_visit": "03 Mei 2026", "rekam_medik": "RM-2025-0118"},
    3: {"nik": "3174****0003", "dokter": "dr. Santoso, Sp.JP", "jadwal": "Selasa 09:00","last_visit": "20 Apr 2026", "rekam_medik": "RM-2025-0291"},
}


def _get_user_or_404(user_id: int, db: Session):
    repo = UserRepository(db)
    user = repo.get(user_id)
    if not user:
        raise HTTPException(
            status_code=404,
            detail={"error": "user_not_found", "message": "User tidak ditemukan."},
        )
    return user


# ── Payment ──────────────────────────────────────────────────────────────────

@router.post("/payment/pay", response_model=PaymentPayResponse)
def payment_pay(payload: PaymentPayRequest, db: Session = Depends(get_db)):
    """Record a palm-authorized payment transaction."""
    user = _get_user_or_404(payload.user_id, db)

    from uuid import uuid4
    txn_id = f"PAY-{uuid4().hex[:10].upper()}"

    log_repo = DemoLogRepository(db)
    log_repo.create(
        user_id=user.id,
        demo_type="payment",
        payload={"transaction_id": txn_id, "amount": payload.amount, "merchant": payload.merchant},
        match_score=payload.match_score,
    )

    return PaymentPayResponse(
        status="success",
        transaction_id=txn_id,
        user={"id": user.id, "name": user.name},
        amount=payload.amount,
        merchant=payload.merchant,
        timestamp=datetime.utcnow(),
    )


# ── Attendance ────────────────────────────────────────────────────────────────

@router.post("/attendance/checkin", response_model=AttendanceCheckinResponse)
def attendance_checkin(payload: AttendanceCheckinRequest, db: Session = Depends(get_db)):
    """Record check-in or check-out for a user."""
    user = _get_user_or_404(payload.user_id, db)

    log_repo = DemoLogRepository(db)
    log = log_repo.create(
        user_id=user.id,
        demo_type="attendance",
        payload={"mode": payload.mode},
        match_score=payload.match_score,
    )

    return AttendanceCheckinResponse(
        status="success",
        user={"id": user.id, "name": user.name},
        mode=payload.mode,
        timestamp=log.timestamp,
    )


# ── Access Control ────────────────────────────────────────────────────────────

@router.get("/access/authorized", response_model=list[AuthorizedUserResponse])
def list_authorized_users(db: Session = Depends(get_db)):
    """List all users with their authorization status."""
    repo = UserRepository(db)
    users = repo.list_all()
    return [
        AuthorizedUserResponse(
            user_id=u.id,
            name=u.name,
            authorized=(u.id in _AUTHORIZED_USER_IDS),
        )
        for u in users
    ]


@router.put("/access/authorized/{user_id}")
def toggle_authorized(user_id: int, authorized: bool, db: Session = Depends(get_db)):
    """Toggle authorization for a specific user."""
    user = _get_user_or_404(user_id, db)
    if authorized:
        _AUTHORIZED_USER_IDS.add(user_id)
    else:
        _AUTHORIZED_USER_IDS.discard(user_id)
    return {"user_id": user_id, "name": user.name, "authorized": authorized}


@router.post("/access/check", response_model=AccessCheckResponse)
async def access_check(
    request: Request,
    image: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Identify palm then check authorization. Returns granted/denied."""
    from services.image_service import upload_to_pil
    from services.identification_service import IdentificationService

    pil_image = await upload_to_pil(image, request.app.state.settings.max_upload_mb)
    service = IdentificationService(request.app.state, db)
    result, latency_ms = service.identify_palm(pil_image)

    identified = result["status"] == "identified"
    granted = identified and (result["user_id"] in _AUTHORIZED_USER_IDS)

    user_payload = None
    if identified:
        user_payload = {"id": result["user_id"], "name": result["user_name"]}

    log_repo = DemoLogRepository(db)
    log_repo.create(
        user_id=result.get("user_id"),
        demo_type="access",
        payload={"granted": granted, "latency_ms": latency_ms},
        match_score=result["score"],
    )

    return AccessCheckResponse(
        status="granted" if granted else "denied",
        user=user_payload,
        score=round(result["score"], 4),
        latency_ms=latency_ms,
        reason="authorized" if granted else ("not_authorized" if identified else "unknown_user"),
    )


# ── Patient Check-in ──────────────────────────────────────────────────────────

@router.post("/patient/checkin", response_model=PatientCheckinResponse)
def patient_checkin(payload: PatientCheckinRequest, db: Session = Depends(get_db)):
    """Record patient check-in and return patient card data."""
    user = _get_user_or_404(payload.user_id, db)

    patient = _PATIENT_DATA.get(
        user.id,
        {
            "nik": f"32**********{user.id:04d}",
            "dokter": "dr. Demo, Sp.U",
            "jadwal": "Hari ini",
            "last_visit": "—",
            "rekam_medik": f"RM-DEMO-{user.id:04d}",
        },
    )

    log_repo = DemoLogRepository(db)
    log = log_repo.create(
        user_id=user.id,
        demo_type="patient",
        payload={"patient": patient},
        match_score=payload.match_score,
    )

    return PatientCheckinResponse(
        status="success",
        user={"id": user.id, "name": user.name},
        patient=patient,
        timestamp=log.timestamp,
    )
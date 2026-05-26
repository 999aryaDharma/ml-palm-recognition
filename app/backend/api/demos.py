from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from db.database import get_db
from schemas.demos import (
    PaymentPayRequest, PaymentPayResponse,
    AttendanceCheckinRequest, AttendanceCheckinResponse,
    AccessCheckRequest, AccessCheckResponse,
    PatientCheckinRequest, PatientCheckinResponse,
)
from services.demo_service import DemoService

router = APIRouter()


@router.post("/payment/pay", response_model=PaymentPayResponse)
def payment_pay(
    payload: PaymentPayRequest,
    db: Session = Depends(get_db),
):
    """
    Process payment demo action.
    
    Args:
        payload: Payment details
        
    Returns:
        Payment result with transaction ID
    """
    try:
        service = DemoService(db)
        result = service.process_payment(payload.user_id, payload.amount, payload.merchant)
        return PaymentPayResponse(**result)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail={"error": "payment_failed", "message": str(e)}
        )


@router.post("/attendance/checkin", response_model=AttendanceCheckinResponse)
def attendance_checkin(
    payload: AttendanceCheckinRequest,
    db: Session = Depends(get_db),
):
    """
    Process attendance check-in demo action.
    
    Args:
        payload: Attendance details
        
    Returns:
        Check-in result
    """
    try:
        service = DemoService(db)
        result = service.process_attendance(payload.user_id, payload.location)
        return AttendanceCheckinResponse(**result)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail={"error": "attendance_failed", "message": str(e)}
        )


@router.post("/access/check", response_model=AccessCheckResponse)
def access_check(
    payload: AccessCheckRequest,
    db: Session = Depends(get_db),
):
    """
    Process access control check demo action.
    
    Args:
        payload: Access check details
        
    Returns:
        Access result
    """
    try:
        service = DemoService(db)
        result = service.process_access(payload.user_id, payload.door_id)
        return AccessCheckResponse(**result)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail={"error": "access_denied", "message": str(e)}
        )


@router.post("/patient/checkin", response_model=PatientCheckinResponse)
def patient_checkin(
    payload: PatientCheckinRequest,
    db: Session = Depends(get_db),
):
    """
    Process patient check-in demo action.
    
    Args:
        payload: Patient check-in details
        
    Returns:
        Check-in result
    """
    try:
        service = DemoService(db)
        result = service.process_patient(payload.user_id, payload.hospital_id)
        return PatientCheckinResponse(**result)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail={"error": "patient_checkin_failed", "message": str(e)}
        )

"""
Demo Service Module
Handles demo module operations (payment, attendance, access, patient)
"""

from sqlalchemy.orm import Session
from db.models import DemoLog
import json


class DemoService:
    """Service for managing demo module operations."""
    
    def __init__(self, db: Session):
        """Initialize demo service."""
        self.db = db
    
    def log_demo(
        self,
        demo_type: str,
        user_id: int | None,
        match_score: float = 0.0,
        payload: dict = None
    ) -> DemoLog:
        """
        Log demo action to database.
        
        Args:
            demo_type: Type of demo (payment, attendance, access, patient)
            user_id: Identified user ID or None
            match_score: Match score if applicable
            payload: Additional data to store
            
        Returns:
            Created DemoLog record
        """
        from db.repositories import DemoLogRepository
        
        repo = DemoLogRepository(self.db)
        log = repo.create(
            user_id=user_id,
            demo_type=demo_type,
            payload=payload or {},
            match_score=match_score
        )
        return log
    
    def process_payment(self, user_id: int, amount: float, merchant: str) -> dict:
        """
        Process payment demo action.
        
        Args:
            user_id: User who authorized payment
            amount: Payment amount
            merchant: Merchant name
            
        Returns:
            Payment result dict with transaction_id
        """
        txn_id = f"TXN-{hash(f'{user_id}-{amount}-{merchant}') % 10000:05d}"
        
        self.log_demo(
            demo_type="payment",
            user_id=user_id,
            match_score=0.95,
            payload={
                "amount": amount,
                "merchant": merchant,
                "transaction_id": txn_id
            }
        )
        
        return {
            "transaction_id": txn_id,
            "amount": amount,
            "merchant": merchant,
            "status": "success"
        }
    
    def process_attendance(self, user_id: int, location: str) -> dict:
        """
        Process attendance demo action.
        
        Args:
            user_id: User checking in
            location: Location/department
            
        Returns:
            Attendance result dict
        """
        self.log_demo(
            demo_type="attendance",
            user_id=user_id,
            match_score=0.92,
            payload={
                "location": location,
                "action": "check_in"
            }
        )
        
        return {
            "status": "checked_in",
            "location": location,
            "user_id": user_id
        }
    
    def process_access(self, user_id: int, door_id: str) -> dict:
        """
        Process access control demo action.
        
        Args:
            user_id: User requesting access
            door_id: Door/area identifier
            
        Returns:
            Access result dict
        """
        self.log_demo(
            demo_type="access",
            user_id=user_id,
            match_score=0.94,
            payload={
                "door_id": door_id,
                "action": "access_granted"
            }
        )
        
        return {
            "status": "access_granted",
            "door_id": door_id,
            "user_id": user_id
        }
    
    def process_patient(self, user_id: int, hospital_id: str) -> dict:
        """
        Process patient check-in demo action.
        
        Args:
            user_id: Patient checking in
            hospital_id: Hospital identifier
            
        Returns:
            Patient check-in result dict
        """
        self.log_demo(
            demo_type="patient",
            user_id=user_id,
            match_score=0.91,
            payload={
                "hospital_id": hospital_id,
                "action": "check_in"
            }
        )
        
        return {
            "status": "checked_in",
            "hospital_id": hospital_id,
            "user_id": user_id
        }

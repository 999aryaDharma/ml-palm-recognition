from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from datetime import datetime

from db.database import get_db
from db.models import DemoLog

router = APIRouter()


@router.get("")
def get_demo_logs(
    demo_type: str | None = Query(None),
    limit: int = Query(100, ge=1, le=500),
    db: Session = Depends(get_db),
):
    """
    Get demo logs.
    
    Args:
        demo_type: Filter by demo type (payment, attendance, access, patient), optional
        limit: Maximum number of logs to return
        
    Returns:
        List of demo logs
    """
    query = db.query(DemoLog).order_by(DemoLog.timestamp.desc())
    
    if demo_type:
        query = query.filter(DemoLog.demo_type == demo_type)
    
    logs = query.limit(limit).all()
    
    return {
        "logs": [
            {
                "id": log.id,
                "user_id": log.user_id,
                "demo_type": log.demo_type,
                "match_score": log.match_score,
                "timestamp": log.timestamp.isoformat(),
            }
            for log in logs
        ],
        "total": len(logs),
    }


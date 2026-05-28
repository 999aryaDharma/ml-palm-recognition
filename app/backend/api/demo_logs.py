from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from db.database import get_db
from db.repositories import DemoLogRepository
from schemas.demos import DemoLogResponse

router = APIRouter()


def _to_response(log) -> DemoLogResponse:
    return DemoLogResponse(
        id=log.id,
        user={"id": log.user.id, "name": log.user.name} if log.user else None,
        demo_type=log.demo_type,
        timestamp=log.timestamp,
        match_score=round(log.match_score, 4),
        payload=log.payload,
    )


@router.get("", response_model=list[DemoLogResponse])
def list_demo_logs(
    demo_type: str | None = Query(default=None, description="Filter by demo type: payment|attendance|access|patient"),
    limit: int = Query(default=20, ge=1, le=200),
    db: Session = Depends(get_db),
):
    """Return recent demo activity logs, newest first."""
    repo = DemoLogRepository(db)
    logs = repo.list(demo_type=demo_type, limit=limit)
    return [_to_response(log) for log in logs]


@router.post("")
def create_demo_log(
    user_id: int | None = None,
    demo_type: str = "generic",
    match_score: float = 0.0,
    payload: dict = {},
    db: Session = Depends(get_db),
):
    """Create a demo log entry (called by frontend after demo action)."""
    repo = DemoLogRepository(db)
    log = repo.create(user_id=user_id, demo_type=demo_type, payload=payload, match_score=match_score)
    return {"log_id": log.id}
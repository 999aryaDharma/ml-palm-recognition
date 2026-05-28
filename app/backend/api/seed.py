from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session
import numpy as np

from db.database import get_db
from db.repositories import UserRepository, TemplateRepository

router = APIRouter()

DEMO_USERS = [
    "Budi Santoso",
    "Siti Nurhaliza",
    "Ahmad Wijaya",
    "Nadia Putri",
    "Raka Saputra",
]
TEMPLATES_PER_USER = 5  # matches enrollment min


@router.post("/seed-demo-data")
def seed_demo_data(request: Request, db: Session = Depends(get_db)):
    """
    Seed database with demo users and random-embedding templates.
    Idempotent: skips users that already exist by name.
    """
    user_repo = UserRepository(db)
    template_repo = TemplateRepository(db)

    existing_names = {u.name for u in user_repo.list_all()}
    created = 0

    for name in DEMO_USERS:
        if name in existing_names:
            continue
        user = user_repo.create(name)
        for _ in range(TEMPLATES_PER_USER):
            # Random unit-norm embedding (stands in until real ML available)
            emb = np.random.rand(128).astype(np.float32)
            emb /= np.linalg.norm(emb) + 1e-8
            template_repo.create(user.id, emb, quality_score=0.85 + np.random.rand() * 0.14)
        created += 1

    # Refresh in-memory cache so seeded users are immediately identifiable
    cache = getattr(request.app.state, "cache", None)
    if cache is not None:
        cache.refresh(db)

    return {
        "status": "ok",
        "seeded_users": created,
        "total_users": len(DEMO_USERS),
        "message": f"Seeded {created} new users ({len(DEMO_USERS) - created} already existed).",
    }


@router.delete("/seed-demo-data")
def reset_demo_data(request: Request, db: Session = Depends(get_db)):
    """
    Delete ALL users, templates, and demo logs.
    Use with care — intended for demo reset only.
    """
    from db.models import DemoLog, Template, User

    db.query(DemoLog).delete()
    db.query(Template).delete()
    db.query(User).delete()
    db.commit()

    # Clear cache
    cache = getattr(request.app.state, "cache", None)
    if cache is not None:
        cache._users = []

    return {"status": "ok", "deleted": True, "message": "All demo data deleted."}
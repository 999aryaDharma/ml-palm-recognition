from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
import numpy as np

from db.database import get_db
from db.repositories import UserRepository, TemplateRepository
from ml.cache import EmbeddingCache

router = APIRouter()


DEMO_USERS = [
    {"name": "Budi Santoso", "templates": 5},
    {"name": "Siti Nurhaliza", "templates": 5},
    {"name": "Ahmad Wijaya", "templates": 5},
]


@router.post("/seed-demo-data")
def seed_demo_data(
    request,
    db: Session = Depends(get_db),
):
    """
    Seed database with demo users and templates.
    
    Returns:
        Confirmation message with created user count
    """
    try:
        user_repo = UserRepository(db)
        template_repo = TemplateRepository(db)
        
        created_count = 0
        
        for user_data in DEMO_USERS:
            # Create user
            user = user_repo.create(user_data["name"])
            
            # Add dummy templates (random embeddings)
            for i in range(user_data["templates"]):
                embedding = np.random.rand(128).astype(np.float32)
                quality_score = 0.85 + np.random.rand() * 0.15
                template_repo.create(user.id, embedding, quality_score)
            
            created_count += 1
        
        # Refresh cache
        cache = getattr(request.app.state, "cache", None)
        if cache:
            cache.refresh(db)
        
        return {
            "status": "success",
            "message": f"Seeded {created_count} demo users with templates",
            "users_created": created_count,
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
        }


@router.delete("/seed-demo-data")
def reset_demo_data(
    request,
    db: Session = Depends(get_db),
):
    """
    Delete all demo data from database.
    
    Returns:
        Confirmation message
    """
    try:
        from db.models import User, DemoLog
        
        # Delete all demo logs
        db.query(DemoLog).delete()
        
        # Delete all users (cascade will delete templates)
        db.query(User).delete()
        
        db.commit()
        
        # Clear cache
        cache = getattr(request.app.state, "cache", None)
        if cache:
            cache._users = []
        
        return {
            "status": "success",
            "message": "All demo data deleted",
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
        }


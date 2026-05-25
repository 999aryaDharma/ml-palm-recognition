import io
import numpy as np
from sqlalchemy.orm import Session, joinedload
from db.models import User, Template, DemoLog
from datetime import datetime
import json


def embedding_to_blob(embedding: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, embedding.astype(np.float32))
    return buffer.getvalue()


def blob_to_embedding(blob: bytes) -> np.ndarray:
    buffer = io.BytesIO(blob)
    buffer.seek(0)
    return np.load(buffer).astype(np.float32)


class UserRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(self, name: str) -> User:
        user = User(name=name.strip())
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)
        return user

    def list_all(self) -> list[User]:
        return (
            self.db.query(User)
            .options(joinedload(User.templates))
            .order_by(User.enrolled_at.desc())
            .all()
        )

    def get(self, user_id: int) -> User | None:
        return (
            self.db.query(User)
            .options(joinedload(User.templates))
            .filter(User.id == user_id)
            .first()
        )

    def delete(self, user_id: int) -> bool:
        user = self.get(user_id)
        if not user:
            return False
        self.db.delete(user)
        self.db.commit()
        return True


class TemplateRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(self, user_id: int, embedding: np.ndarray, quality_score: float) -> Template:
        template = Template(
            user_id=user_id,
            embedding=embedding_to_blob(embedding),
            quality_score=quality_score,
        )
        self.db.add(template)
        self.db.commit()
        self.db.refresh(template)
        return template

    def list_by_user(self, user_id: int) -> list[Template]:
        return self.db.query(Template).filter(Template.user_id == user_id).all()

    def list_all_grouped(self) -> list[dict]:
        users = (
            self.db.query(User)
            .options(joinedload(User.templates))
            .all()
        )
        return [
            {
                "user_id": user.id,
                "user_name": user.name,
                "embeddings": [blob_to_embedding(t.embedding) for t in user.templates],
            }
            for user in users
            if len(user.templates) > 0
        ]


class DemoLogRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(self, user_id: int | None, demo_type: str, payload: dict, match_score: float = 0.0) -> DemoLog:
        log = DemoLog(
            user_id=user_id,
            demo_type=demo_type,
            payload_json=json.dumps(payload),
            match_score=match_score,
        )
        self.db.add(log)
        self.db.commit()
        self.db.refresh(log)
        return log

    def list(self, demo_type: str | None = None, limit: int = 20) -> list[DemoLog]:
        query = self.db.query(DemoLog).options(joinedload(DemoLog.user))
        if demo_type:
            query = query.filter(DemoLog.demo_type == demo_type)
        return query.order_by(DemoLog.timestamp.desc()).limit(limit).all()

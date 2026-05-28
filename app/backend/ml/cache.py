"""
Embedding Cache — in-memory store for fast identification.
Loaded at startup; refreshed after every enroll/delete.
"""
from __future__ import annotations
from typing import Optional
from sqlalchemy.orm import Session


class EmbeddingCache:
    def __init__(self):
        self._users: list[dict] = []

    def warm_up(self, db: Session) -> None:
        from db.repositories import TemplateRepository
        repo = TemplateRepository(db)
        self._users = repo.list_all_grouped()

    def refresh(self, db: Session) -> None:
        """Full reload from DB — called after enroll / delete."""
        self.warm_up(db)

    def get_all(self) -> list[dict]:
        return self._users

    def get_user(self, user_id: int) -> Optional[dict]:
        return next((u for u in self._users if u["user_id"] == user_id), None)

    @property
    def user_count(self) -> int:
        return len(self._users)
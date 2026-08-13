"""
Embedding Cache — in-memory store for fast biometric identification.

Segmented by (model_id, model_version) namespace to ensure embeddings from
different models/versions are NEVER mixed during matching.
"""
from __future__ import annotations
from typing import Optional
from sqlalchemy.orm import Session


class EmbeddingCache:
    def __init__(self):
        # Key: (model_id, model_version) -> list[dict]
        self._store: dict[tuple[str, str], list[dict]] = {}

    def warm_up(
        self,
        db: Session,
        model_id: str | None = None,
        model_version: str | None = None,
    ) -> None:
        from db.repositories import TemplateRepository
        repo = TemplateRepository(db)

        if model_id and model_version:
            grouped = repo.list_all_grouped(model_id=model_id, model_version=model_version)
            self._store[(model_id, model_version)] = grouped
        else:
            from db.models import Template
            pairs = db.query(Template.model_id, Template.model_version).distinct().all()
            self._store.clear()
            if not pairs:
                self._store[("mobilefacenet-pretrained", "1.0.0")] = []
            for m_id, m_ver in pairs:
                if m_id and m_ver:
                    grouped = repo.list_all_grouped(model_id=m_id, model_version=m_ver)
                    self._store[(m_id, m_ver)] = grouped

    def refresh(
        self,
        db: Session,
        model_id: str | None = None,
        model_version: str | None = None,
    ) -> None:
        """Full or namespace-specific reload from DB."""
        self.warm_up(db, model_id=model_id, model_version=model_version)

    def get_all(
        self,
        model_id: str = "mobilefacenet-pretrained",
        model_version: str = "1.0.0",
    ) -> list[dict]:
        """Return user template records ONLY for requested (model_id, model_version) namespace."""
        return self._store.get((model_id, model_version), [])

    def get_user(
        self,
        user_id: int,
        model_id: str = "mobilefacenet-pretrained",
        model_version: str = "1.0.0",
    ) -> Optional[dict]:
        users = self.get_all(model_id=model_id, model_version=model_version)
        return next((u for u in users if u["user_id"] == user_id), None)

    @property
    def user_count(self) -> int:
        all_users = set()
        for user_list in self._store.values():
            for u in user_list:
                all_users.add(u["user_id"])
        return len(all_users)
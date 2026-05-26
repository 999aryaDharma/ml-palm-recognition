"""
Embedding Cache Module
In-memory cache for user embeddings to speed up identification
"""

import numpy as np
from sqlalchemy.orm import Session
from typing import Optional, List


class EmbeddingCache:
    """In-memory cache for user embeddings."""
    
    def __init__(self):
        """Initialize empty cache."""
        self._users: List[dict] = []
        self._is_warmed = False
    
    def warm_up(self, db: Session) -> None:
        """
        Load all user embeddings into memory.
        
        Args:
            db: SQLAlchemy session
        """
        from db.repositories import TemplateRepository
        
        repo = TemplateRepository(db)
        self._users = repo.list_all_grouped()
        self._is_warmed = True
        print(f"[EmbeddingCache] Warmed up with {len(self._users)} users")
    
    def refresh(self, db: Session) -> None:
        """
        Refresh cache with latest embeddings from database.
        
        Args:
            db: SQLAlchemy session
        """
        self.warm_up(db)
    
    def get_all(self) -> List[dict]:
        """
        Get all cached users.
        
        Returns:
            List of user dicts with user_id, user_name, embeddings
        """
        return self._users
    
    def get_user(self, user_id: int) -> Optional[dict]:
        """
        Get specific user from cache.
        
        Args:
            user_id: User ID
            
        Returns:
            User dict or None if not found
        """
        for user in self._users:
            if user["user_id"] == user_id:
                return user
        return None
    
    @property
    def user_count(self) -> int:
        """Get number of cached users."""
        return len(self._users)
    
    @property
    def is_warmed(self) -> bool:
        """Check if cache is warmed up."""
        return self._is_warmed

from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from schemas.users import UserResponse


class IdentifyResponse(BaseModel):
    status: str                        # "identified" | "unknown"
    user: Optional[UserResponse] = None
    score: float
    latency_ms: int
    error_code: Optional[str] = None
    message: Optional[str] = None
    bbox: Optional[Dict[str, float]] = None
    landmarks: Optional[List[Dict[str, float]]] = None
    quality_score: Optional[float] = None
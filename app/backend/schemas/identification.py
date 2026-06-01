from pydantic import BaseModel
from typing import Optional, List, Dict, Any


class IdentifiedUser(BaseModel):
    id: int
    name: str


class IdentifyResponse(BaseModel):
    status: str                        # "identified" | "unknown"
    user: Optional[IdentifiedUser] = None
    score: float
    latency_ms: int
    error_code: Optional[str] = None  # populated when status is "error" (surfaced as HTTP 400)
    bbox: Optional[Dict[str, float]] = None
    landmarks: Optional[List[Dict[str, float]]] = None
    quality_score: Optional[float] = None
from pydantic import BaseModel


class IdentifiedUser(BaseModel):
    id: int
    name: str


class IdentifyResponse(BaseModel):
    status: str
    user: IdentifiedUser | None = None
    score: float
    latency_ms: int

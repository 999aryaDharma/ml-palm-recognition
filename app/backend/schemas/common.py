from pydantic import BaseModel


class ErrorDetail(BaseModel):
    error: str
    message: str


class DeleteResponse(BaseModel):
    deleted: bool

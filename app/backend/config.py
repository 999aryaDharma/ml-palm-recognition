from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    app_name: str = "Palm Biometric API"
    app_version: str = "0.5.0"
    debug: bool = True

    database_url: str = "sqlite:///./palm_biometric.db"

    cors_origins: list[str] = ["*"]

    hand_landmarker_path: str = "ml/models/hand_landmarker.task"
    recognizer_model_path: str = "ml/models/palm_recognizer.pt"
    threshold_path: str = "ml/models/threshold.json"
    default_threshold: float = 0.70

    max_upload_mb: int = 8
    min_template_per_user: int = 5
    top_k_templates: int = 3

    class Config:
        env_file = ".env"


@lru_cache
def get_settings() -> Settings:
    return Settings()

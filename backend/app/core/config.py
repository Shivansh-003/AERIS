from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "AERIS"
    PROJECT_FULL_NAME: str = "Adaptive Environmental Risk & Intelligence System"
    API_V1_STR: str = "/api/v1"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True

    # Server Configuration
    BACKEND_HOST: str = "0.0.0.0"
    BACKEND_PORT: int = 8000
    LOG_LEVEL: str = "INFO"

    # CORS Configuration
    CORS_ORIGINS: str = "http://localhost:3000,http://127.0.0.1:3000"

    # Storage Paths
    DATA_DIR: str = "data"
    ARTIFACTS_DIR: str = "artifacts"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    @property
    def cors_origins_list(self) -> List[str]:
        return [
            origin.strip()
            for origin in self.CORS_ORIGINS.split(",")
            if origin.strip()
        ]


@lru_cache()
def get_settings() -> Settings:
    return Settings()

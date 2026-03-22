from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """App configuration settings."""

    # API Settings
    API_TITLE: str = "SDOptimizer Backend"
    API_DESCRIPTION: str = "Backend API for SDOptimizer."
    API_VERSION: str = "0.1.0"
    API_ROOT_PATH: str = "/api"
    DEBUG: bool = False

    # Security
    SECRET_KEY: str = "your-secret-key-change-in-production"

    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    TEMP_DIR: Path = BASE_DIR / "app" / "temp"  # User-generated (temporary)

    # Model validation
    MAX_MODEL_FILE_SIZE: int = 50 * 1024 * 1024  # 50 MB

    # CORS
    CORS_ORIGINS: list[str] = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://localhost",
    ]

    # Session Management
    SESSION_MAX_AGE_HOURS: int = 24

    # Logging
    LOGGING_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache
def get_settings() -> Settings:
    """Get cached settings."""
    return Settings()


# Global instance
settings = get_settings()

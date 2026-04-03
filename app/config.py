"""
Application configuration loaded from environment variables.
Uses pydantic-settings for validation and type coercion.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment / .env file."""

    # Authentication
    API_KEY: str = "changeme-secret-key"

    # Groq AI (free tier)
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.3-70b-versatile"

    # spaCy
    SPACY_MODEL: str = "en_core_web_sm"

    # Limits
    MAX_FILE_SIZE_MB: int = 10

    # Logging
    LOG_LEVEL: str = "INFO"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
    }


@lru_cache
def get_settings() -> Settings:
    """Return cached settings instance (singleton)."""
    return Settings()

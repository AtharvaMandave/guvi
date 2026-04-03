"""
API key authentication via x-api-key header.
Uses timing-safe comparison to prevent timing attacks.
"""

import secrets

from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader

from app.config import get_settings

# Header extractor — auto_error=False so we can return a custom message
_api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)


async def verify_api_key(
    api_key: str | None = Security(_api_key_header),
) -> str:
    """
    FastAPI dependency that validates the x-api-key header.
    Raises 401 if missing or invalid.
    """
    settings = get_settings()

    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing x-api-key header",
        )

    if not secrets.compare_digest(api_key, settings.API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )

    return api_key

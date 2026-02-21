"""API package."""

from .schemas import (
    Base64AudioRequest,
    DetectionResponse,
    ErrorResponse,
    Base64EncodeResponse
)
from .auth import verify_api_key
from .routes import APIRoutes

__all__ = [
    'Base64AudioRequest',
    'DetectionResponse',
    'ErrorResponse',
    'Base64EncodeResponse',
    'verify_api_key',
    'APIRoutes',
]

"""API schemas and models."""

from pydantic import BaseModel, Field, validator
from typing import Optional

from config.settings import SUPPORTED_LANGUAGES


class Base64AudioRequest(BaseModel):
    """Request model for base64 audio detection."""
    
    language: str = Field(
        ...,
        description="Language of the audio: Tamil, English, Hindi, Malayalam, Telugu"
    )
    audioFormat: str = Field(..., description="Audio format (mp3)")
    audioBase64: str = Field(..., description="Base64 encoded audio file")
    threshold: Optional[float] = Field(
        None,
        description="Custom detection threshold (0.0-1.0)"
    )

    @validator('language')
    def validate_language(cls, v):
        """Validate language is supported."""
        language_lower = v.lower()
        for lang in SUPPORTED_LANGUAGES:
            if lang.lower() == language_lower:
                return lang
        raise ValueError(f"Language must be one of: {', '.join(SUPPORTED_LANGUAGES)}")
    
    @validator('audioFormat')
    def validate_format(cls, v):
        """Validate audio format."""
        if v.lower() != "mp3":
            raise ValueError("Only MP3 format is supported")
        return v.lower()

    class Config:
        json_schema_extra = {
            "example": {
                "language": "Tamil",
                "audioFormat": "mp3",
                "audioBase64": "SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU2LjM2LjEwMAAAAAAA..."
            }
        }


class DetectionResponse(BaseModel):
    """Response model for detection results."""
    
    status: str = Field(..., description="Status of the request: 'success' or 'error'")
    classification: str = Field(
        ...,
        description="Classification: 'AI_GENERATED' or 'HUMAN'"
    )
    confidenceScore: float = Field(
        ...,
        description="Confidence score (0.0-1.0). Higher values indicate greater confidence"
    )


class ErrorResponse(BaseModel):
    """Response model for errors."""
    
    status: str = Field("error", description="Status of the request")
    message: str = Field(..., description="Error message")


class Base64EncodeResponse(BaseModel):
    """Response model for base64 encoding."""
    
    status: str = Field(..., description="Status of the request")
    filename: str = Field(..., description="Original filename")
    fileSize: int = Field(..., description="File size in bytes")
    base64Length: int = Field(..., description="Length of base64 string")
    audioBase64: str = Field(..., description="Base64 encoded audio string")

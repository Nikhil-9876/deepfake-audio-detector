"""API route handlers."""

import os
import uuid
import tempfile
import base64
from fastapi import File, UploadFile, Depends, HTTPException
from typing import Optional

from api.schemas import (
    Base64AudioRequest,
    DetectionResponse,
    Base64EncodeResponse
)
from api.auth import verify_api_key
from config.settings import SUPPORTED_LANGUAGES, MAX_AUDIO_DURATION


class APIRoutes:
    """API route handlers."""
    
    def __init__(self, detector):
        """
        Initialize routes with detector instance.
        
        Args:
            detector: AudioDetector instance
        """
        self.detector = detector
    
    async def root(self):
        """Root endpoint with API information."""
        return {
            "message": "AI Audio Detector API - Voice Classification System",
            "version": "1.0.0",
            "description": "Detects AI-generated vs Human voice across multiple languages",
            "supported_languages": SUPPORTED_LANGUAGES,
            "max_audio_duration": f"{MAX_AUDIO_DURATION}s",
            "processing_method": "Sliding window analysis for complete audio coverage",
            "authentication": "Required: x-api-key header",
            "endpoints": {
                "POST /api/detect-from-file": "Upload audio file directly (requires API key)",
                "POST /api/voice-detection": "Detect from base64 MP3 audio (requires API key)",
                "POST /api/encode-to-base64": "Encode audio file to base64 (requires API key)",
                "GET /health": "Health check endpoint",
                "GET /docs": "Interactive API documentation"
            },
            "classification_types": ["AI_GENERATED", "HUMAN"],
            "confidence_range": "Confidence scores range from 0.0 to 1.0"
        }
    
    async def health_check(self):
        """Health check endpoint."""
        from config.settings import DEVICE, MODEL_PATH, OPTIMAL_THRESHOLD
        
        return {
            "status": "healthy",
            "device": str(DEVICE),
            "model_loaded": os.path.exists(MODEL_PATH),
            "threshold": OPTIMAL_THRESHOLD,
            "supported_languages": SUPPORTED_LANGUAGES,
            "api_version": "1.0.0"
        }
    
    async def voice_detection(
        self,
        request: Base64AudioRequest,
        api_key: str = Depends(verify_api_key)
    ) -> DetectionResponse:
        """
        Detect AI-generated voice from base64 encoded audio.
        
        Args:
            request: Base64AudioRequest with audio data
            api_key: Validated API key
        
        Returns:
            DetectionResponse with classification results
        """
        try:
            result = self.detector.detect(
                audio_input=request.audioBase64,
                is_base64=True,
                language=request.language,
                threshold=request.threshold,
                base64_format=request.audioFormat
            )
            return DetectionResponse(**result)
        
        except (ValueError, Exception):
            return DetectionResponse(
                status="error",
                classification="HUMAN",
                confidenceScore=0.0
            )
    
    async def detect_from_base64(
        self,
        request: Base64AudioRequest,
        api_key: str = Depends(verify_api_key)
    ) -> DetectionResponse:
        """Legacy endpoint - use /api/voice-detection instead."""
        return await self.voice_detection(request, api_key)
    
    async def detect_from_file(
        self,
        file: UploadFile = File(..., description="Audio file (MP3, WAV, FLAC, etc.)"),
        language: str = "English",
        threshold: Optional[float] = None,
        api_key: str = Depends(verify_api_key)
    ) -> DetectionResponse:
        """
        Direct audio file upload endpoint.
        
        Args:
            file: Uploaded audio file
            language: Language of the audio
            threshold: Custom detection threshold
            api_key: Validated API key
        
        Returns:
            DetectionResponse with classification results
        """
        # Validate language
        validated_language = self._validate_language(language)
        if validated_language is None:
            return DetectionResponse(
                status="error",
                classification="HUMAN",
                confidenceScore=0.0
            )
        
        # Validate threshold
        if threshold is not None and (threshold < 0.0 or threshold > 1.0):
            return DetectionResponse(
                status="error",
                classification="HUMAN",
                confidenceScore=0.0
            )
        
        # Save uploaded file temporarily
        temp_path = self._save_temp_file(file)
        
        try:
            # Write uploaded file content
            content = await file.read()
            with open(temp_path, "wb") as f:
                f.write(content)
            
            # Process the audio file
            result = self.detector.detect(
                audio_input=temp_path,
                is_base64=False,
                language=validated_language,
                threshold=threshold,
                base64_format=None
            )
            
            return DetectionResponse(**result)
        
        except (ValueError, Exception):
            return DetectionResponse(
                status="error",
                classification="HUMAN",
                confidenceScore=0.0
            )
        finally:
            self._cleanup_temp_file(temp_path)
    
    async def encode_to_base64(
        self,
        file: UploadFile = File(..., description="Audio file to encode to base64"),
        api_key: str = Depends(verify_api_key)
    ) -> Base64EncodeResponse:
        """
        Encode audio file to base64.
        
        Args:
            file: Audio file to encode
            api_key: Validated API key
        
        Returns:
            Base64EncodeResponse with encoded audio
        """
        try:
            content = await file.read()
            audio_base64 = base64.b64encode(content).decode('utf-8')
            
            return Base64EncodeResponse(
                status="success",
                filename=file.filename or "unknown",
                fileSize=len(content),
                base64Length=len(audio_base64),
                audioBase64=audio_base64
            )
        
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail={
                    "status": "error",
                    "message": f"Error encoding file: {str(e)}",
                },
            )
    
    @staticmethod
    def _validate_language(language: str) -> Optional[str]:
        """Validate and normalize language string."""
        language_lower = language.lower()
        for lang in SUPPORTED_LANGUAGES:
            if lang.lower() == language_lower:
                return lang
        return None
    
    @staticmethod
    def _save_temp_file(file: UploadFile) -> str:
        """Create temporary file path for uploaded file."""
        temp_dir = tempfile.gettempdir()
        file_ext = os.path.splitext(file.filename or "audio.mp3")[1] or ".mp3"
        return os.path.join(temp_dir, f"upload_{uuid.uuid4().hex}{file_ext}")
    
    @staticmethod
    def _cleanup_temp_file(path: str):
        """Clean up temporary file."""
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass

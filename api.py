"""
AI Audio Detector API - Main Entry Point

A modular API for detecting AI-generated vs human speech.
"""

import warnings
warnings.filterwarnings("ignore")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.settings import (
    DEVICE,
    validate_api_key_setup,
    load_optimal_threshold,
    SUPPORTED_LANGUAGES,
    MAX_AUDIO_DURATION,
)
from models import ModelManager
from core import AudioDetector
from api import APIRoutes, DetectionResponse

# Initialize configuration
print("Using device:", DEVICE)
validate_api_key_setup()

# Load optimal threshold
OPTIMAL_THRESHOLD = load_optimal_threshold()

# Initialize models
model_manager = ModelManager()
wavlm, aasist, ocsoft = model_manager.load_models()

# Initialize detector
detector = AudioDetector(wavlm, aasist, ocsoft)

# Initialize routes
routes = APIRoutes(detector)

# Create FastAPI app
app = FastAPI(
    title="AI Audio Detector API",
    description="API for detecting AI-generated vs human speech",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return await routes.root()


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return await routes.health_check()


@app.post("/api/voice-detection", response_model=DetectionResponse)
async def voice_detection(request, api_key=None):
    """
    Detect AI-generated voice from base64 encoded audio.
    
    Required Headers:
    - **x-api-key**: Your API key for authentication
    
    Request Body:
    - **language**: Language of the audio (Tamil, English, Hindi, Malayalam, Telugu)
    - **audioFormat**: Audio format (mp3)
    - **audioBase64**: Base64 encoded audio file
    """
    from fastapi import Depends
    from api.schemas import Base64AudioRequest
    from api.auth import verify_api_key
    
    # This is a workaround to properly inject dependencies
    return await routes.voice_detection(request, api_key or Depends(verify_api_key))


@app.post("/detect/base64", response_model=DetectionResponse)
async def detect_from_base64(request, api_key=None):
    """Legacy endpoint - use /api/voice-detection instead."""
    from fastapi import Depends
    from api.schemas import Base64AudioRequest
    from api.auth import verify_api_key
    
    return await routes.detect_from_base64(request, api_key or Depends(verify_api_key))


@app.post("/api/detect-from-file", response_model=DetectionResponse)
async def detect_from_file(file, language: str = "English", threshold=None, api_key=None):
    """
    Direct audio file upload endpoint - no base64 encoding needed!
    
    Upload an audio file directly and get AI detection results.
    The API handles all preprocessing automatically.
    
    Required Headers:
    - **x-api-key**: Your API key for authentication
    
    Form Data:
    - **file**: Audio file to analyze (MP3, WAV, FLAC, etc.)
    - **language**: Language of the audio (optional, default: English)
    - **threshold**: Custom detection threshold 0.0-1.0 (optional)
    
    Returns the same DetectionResponse as /api/voice-detection
    """
    from fastapi import Depends, File, UploadFile
    from api.auth import verify_api_key
    
    return await routes.detect_from_file(file, language, threshold, api_key or Depends(verify_api_key))


@app.post("/api/encode-to-base64")
async def encode_audio_to_base64(file, api_key=None):
    """
    Upload an audio file and get back its base64 encoded string.
    Useful for testing the voice detection API.
    
    Required Headers:
    - **x-api-key**: Your API key for authentication
    
    Request:
    - **file**: Audio file to encode (any format)
    """
    from fastapi import Depends, File, UploadFile
    from api.auth import verify_api_key
    
    return await routes.encode_to_base64(file, api_key or Depends(verify_api_key))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

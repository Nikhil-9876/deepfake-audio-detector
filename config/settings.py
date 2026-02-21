"""Configuration settings for AI Audio Detector API."""

import os
import torch

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# API Configuration
API_KEY = os.getenv("API_KEY")

# Supported languages
SUPPORTED_LANGUAGES = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]

# Audio processing settings
SAMPLE_RATE = 16000
TARGET_DURATION = 5.0  # Each window will be 5 seconds
MAX_AUDIO_DURATION = 60.0  # Maximum audio duration to process (60 seconds)
SLIDING_WINDOW_HOP = 2.5  # Hop size in seconds for sliding window (50% overlap)

# Normalization settings
NORM_TYPE = "peak"
RMS_TARGET = 0.1
SILENCE_THRESHOLD = 1e-4

# Audio validation settings
MIN_RMS_ENERGY = 0.005  # Minimum RMS to not be considered silent
MAX_SILENCE_RATIO = 0.9  # Max 90% of audio can be silent
MIN_SPEECH_PROB = 0.3  # Minimum speech-like characteristics
MAX_ZERO_CROSSING_RATE = 0.7  # Music/noise has higher ZCR
MIN_ZERO_CROSSING_RATE = 0.02  # Too low = likely not speech
MAX_SPECTRAL_CENTROID = 5000  # Hz, above this is likely noise/music
MIN_SPECTRAL_CENTROID = 150  # Hz, below this is likely rumble/noise
MAX_CLIPPING_RATIO = 0.02  # Max 2% samples can be clipped

# Preprocessing settings
USE_DENOISE = False  # MUST match training config
DENOISE_N_FFT = 1024
DENOISE_HOP_LENGTH = 256
DENOISE_NOISE_PERCENTILE = 10
DENOISE_THRESHOLD_MULT = 1.5
DENOISE_ATTENUATION = 0.2

USE_BANDPASS = True
HIGHPASS_CUTOFF_HZ = 80.0
LOWPASS_CUTOFF_HZ = 7800.0

# Model architecture settings
DROPOUT_P = 0.3

# Ensemble weights
AASIST_WEIGHT = 0.6
OCSOFT_WEIGHT = 0.4

# Model paths
MODEL_PATH = "best_model.pt"
THRESHOLD_PATH = "optimal_threshold.txt"

# Optimal threshold (can be loaded from file)
OPTIMAL_THRESHOLD = 0.5


def load_optimal_threshold() -> float:
    """Load optimal threshold from file if available."""
    if os.path.exists(THRESHOLD_PATH):
        with open(THRESHOLD_PATH, 'r') as f:
            threshold = float(f.read().strip())
        print(f"Loaded optimal threshold: {threshold:.4f}")
        return threshold
    else:
        print(f"Using default threshold: {OPTIMAL_THRESHOLD:.4f}")
        return OPTIMAL_THRESHOLD


def validate_api_key_setup():
    """Validate API key setup and print status."""
    if API_KEY:
        print("✓ API key loaded from environment variable")
    else:
        print("⚠️  WARNING: API_KEY not set! Set API_KEY environment variable.")

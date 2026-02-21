"""Configuration module for AI Audio Detector."""

from .settings import *

__all__ = [
    'DEVICE',
    'API_KEY',
    'SUPPORTED_LANGUAGES',
    'SAMPLE_RATE',
    'TARGET_DURATION',
    'MAX_AUDIO_DURATION',
    'SLIDING_WINDOW_HOP',
    'NORM_TYPE',
    'RMS_TARGET',
    'SILENCE_THRESHOLD',
    'validate_api_key_setup',
    'load_optimal_threshold',
]

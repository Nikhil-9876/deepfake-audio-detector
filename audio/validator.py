"""Audio quality validation."""

import numpy as np
import librosa
from typing import Dict

from config.settings import (
    MIN_RMS_ENERGY,
    MAX_SILENCE_RATIO,
    MAX_ZERO_CROSSING_RATE,
    MIN_ZERO_CROSSING_RATE,
    MAX_SPECTRAL_CENTROID,
    MIN_SPECTRAL_CENTROID,
    MAX_CLIPPING_RATIO,
)


class AudioValidator:
    """Validates audio quality and content."""
    
    @staticmethod
    def validate(wav: np.ndarray, sr: int) -> Dict[str, float]:
        """
        Validate audio quality and content.
        
        Args:
            wav: Audio waveform array
            sr: Sample rate
        
        Returns:
            Dictionary with validation metrics
        
        Raises:
            ValueError: If audio should be rejected
        """
        if len(wav) == 0:
            raise ValueError("Audio is empty")
        
        # Run all validation checks
        validation_result = {
            "rms": AudioValidator._check_silence(wav, sr),
            "silence_ratio": AudioValidator._check_silence_ratio(wav, sr),
            "clipping_ratio": AudioValidator._check_clipping(wav),
        }
        
        # Check speech-like characteristics
        speech_metrics = AudioValidator._check_speech_characteristics(wav, sr)
        validation_result.update(speech_metrics)
        
        return validation_result
    
    @staticmethod
    def _check_silence(wav: np.ndarray, sr: int) -> float:
        """Check if audio is too quiet."""
        rms = np.sqrt(np.mean(wav ** 2))
        if rms < MIN_RMS_ENERGY:
            raise ValueError(
                f"Audio is too quiet (RMS: {rms:.6f}). Please provide clear audio."
            )
        return float(rms)
    
    @staticmethod
    def _check_silence_ratio(wav: np.ndarray, sr: int) -> float:
        """Check percentage of silent frames."""
        frame_length = int(0.02 * sr)  # 20ms frames
        hop_length = frame_length // 2
        frames = librosa.util.frame(wav, frame_length=frame_length, hop_length=hop_length)
        frame_rms = np.sqrt(np.mean(frames ** 2, axis=0))
        silence_ratio = np.sum(frame_rms < MIN_RMS_ENERGY * 0.5) / len(frame_rms)
        
        if silence_ratio > MAX_SILENCE_RATIO:
            raise ValueError(
                f"Audio contains {silence_ratio*100:.1f}% silence. "
                "Please provide clear speech."
            )
        return float(silence_ratio)
    
    @staticmethod
    def _check_clipping(wav: np.ndarray) -> float:
        """Check for audio clipping/distortion."""
        clipping_ratio = np.sum(np.abs(wav) > 0.99) / len(wav)
        if clipping_ratio > MAX_CLIPPING_RATIO:
            raise ValueError(
                f"Audio is clipped/distorted ({clipping_ratio*100:.1f}% samples). "
                "Please provide undistorted audio."
            )
        return float(clipping_ratio)
    
    @staticmethod
    def _check_speech_characteristics(wav: np.ndarray, sr: int) -> Dict[str, float]:
        """Check if audio is speech-like (not music/noise)."""
        non_speech_indicators = 0
        
        # Zero Crossing Rate
        zcr = np.mean(librosa.zero_crossings(wav))
        if zcr > MAX_ZERO_CROSSING_RATE or zcr < MIN_ZERO_CROSSING_RATE:
            non_speech_indicators += 1
        
        # Spectral Centroid
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=wav, sr=sr))
        if spectral_centroid > MAX_SPECTRAL_CENTROID or spectral_centroid < MIN_SPECTRAL_CENTROID:
            non_speech_indicators += 1
        
        # Spectral Rolloff
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=wav, sr=sr, roll_percent=0.85))
        if rolloff > 10000 or rolloff < 800:
            non_speech_indicators += 1
        
        # Only reject if 2 or more indicators suggest non-speech
        if non_speech_indicators >= 2:
            raise ValueError(
                f"Audio does not appear to be clear speech "
                f"(ZCR: {zcr:.3f}, Centroid: {spectral_centroid:.0f}Hz, "
                f"Rolloff: {rolloff:.0f}Hz). Please provide speech-only audio."
            )
        
        return {
            "zero_crossing_rate": float(zcr),
            "spectral_centroid": float(spectral_centroid),
            "spectral_rolloff": float(rolloff),
        }

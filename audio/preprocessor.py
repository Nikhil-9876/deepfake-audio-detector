"""Audio preprocessing and feature extraction."""

import torch
import torchaudio
import librosa
import numpy as np
from typing import List, Tuple, Optional

from config.settings import (
    DEVICE,
    SAMPLE_RATE,
    TARGET_DURATION,
    MAX_AUDIO_DURATION,
    SLIDING_WINDOW_HOP,
    NORM_TYPE,
    RMS_TARGET,
    SILENCE_THRESHOLD,
    USE_DENOISE,
    USE_BANDPASS,
    HIGHPASS_CUTOFF_HZ,
    LOWPASS_CUTOFF_HZ,
    DENOISE_N_FFT,
    DENOISE_HOP_LENGTH,
    DENOISE_NOISE_PERCENTILE,
    DENOISE_THRESHOLD_MULT,
    DENOISE_ATTENUATION,
)


class AudioPreprocessor:
    """Handles audio preprocessing for inference."""
    
    @staticmethod
    def preprocess(
        wav: np.ndarray,
        sr: int,
        return_multiple: bool = False
    ) -> Tuple[List[torch.Tensor], float]:
        """
        Preprocess audio for inference.
        
        Args:
            wav: Audio waveform array
            sr: Sample rate
            return_multiple: If True, return all windows; if False, return center window
        
        Returns:
            Tuple of (list of audio tensors, audio duration)
        """
        if len(wav) == 0:
            raise ValueError("Empty audio file")
        if not np.isfinite(wav).all():
            raise ValueError("Invalid audio values")
        
        # Check duration
        audio_duration = len(wav) / sr
        if audio_duration > MAX_AUDIO_DURATION:
            raise ValueError(
                f"Audio too long ({audio_duration:.1f}s). "
                f"Maximum duration is {MAX_AUDIO_DURATION}s."
            )
        
        # Apply preprocessing pipeline
        wav = AudioPreprocessor._denoise(wav, sr)
        wav = AudioPreprocessor._apply_bandpass(wav, sr)
        wav = AudioPreprocessor._normalize(wav)
        
        # Extract windows
        windows = AudioPreprocessor._extract_windows(wav, sr, audio_duration, return_multiple)
        
        # Convert to tensors
        tensors = [
            torch.tensor(w).float().unsqueeze(0).to(DEVICE)
            for w in windows
        ]
        
        return tensors, audio_duration
    
    @staticmethod
    def _denoise(wav: np.ndarray, sr: int) -> np.ndarray:
        """Apply spectral gating denoise."""
        if not USE_DENOISE:
            return wav
        if wav.size == 0 or not np.isfinite(wav).all():
            return wav
        
        stft = librosa.stft(wav, n_fft=DENOISE_N_FFT, hop_length=DENOISE_HOP_LENGTH)
        mag = np.abs(stft)
        phase = np.exp(1j * np.angle(stft))
        
        noise_floor = np.percentile(mag, DENOISE_NOISE_PERCENTILE, axis=1, keepdims=True)
        thresh = noise_floor * float(DENOISE_THRESHOLD_MULT)
        
        mask = (mag >= thresh).astype(np.float32)
        mag_d = mag * mask + mag * (1.0 - mask) * float(DENOISE_ATTENUATION)
        
        stft_d = mag_d * phase
        wav_out = librosa.istft(stft_d, hop_length=DENOISE_HOP_LENGTH, length=len(wav))
        return wav_out.astype(np.float32)
    
    @staticmethod
    def _apply_bandpass(wav: np.ndarray, sr: int) -> np.ndarray:
        """Apply bandpass filter to focus on speech band."""
        if not USE_BANDPASS:
            return wav
        
        wav_t = torch.tensor(wav).float()
        wav_t = torchaudio.functional.highpass_biquad(
            wav_t, sr, cutoff_freq=HIGHPASS_CUTOFF_HZ
        )
        wav_t = torchaudio.functional.lowpass_biquad(
            wav_t, sr, cutoff_freq=LOWPASS_CUTOFF_HZ
        )
        return wav_t.cpu().numpy()
    
    @staticmethod
    def _normalize(wav: np.ndarray) -> np.ndarray:
        """Normalize audio waveform."""
        if abs(wav).max() < SILENCE_THRESHOLD:
            return wav
        
        if NORM_TYPE == "peak":
            wav = wav / max(abs(wav).max(), 1e-6)
        elif NORM_TYPE == "rms":
            rms = np.sqrt(np.mean(wav**2))
            if rms > 1e-6:
                wav = wav * (RMS_TARGET / rms)
                wav = np.clip(wav, -1.0, 1.0)
        
        return wav
    
    @staticmethod
    def _extract_windows(
        wav: np.ndarray,
        sr: int,
        audio_duration: float,
        return_multiple: bool
    ) -> List[np.ndarray]:
        """Extract windows from audio."""
        target_length = int(TARGET_DURATION * sr)
        current_length = len(wav)
        windows = []
        
        if audio_duration <= TARGET_DURATION:
            # Short audio: pad to target length
            window = AudioPreprocessor._extract_crop(wav, target_length, crop_type="center")
            windows.append(window)
        elif not return_multiple:
            # Single window: use center
            window = AudioPreprocessor._extract_crop(wav, target_length, crop_type="center")
            windows.append(window)
        else:
            # Long audio: sliding window
            hop_length = int(SLIDING_WINDOW_HOP * sr)
            start_positions = list(range(0, current_length - target_length + 1, hop_length))
            
            # Always include the last window
            if start_positions[-1] != current_length - target_length:
                start_positions.append(current_length - target_length)
            
            for start in start_positions:
                window = wav[start:start + target_length]
                windows.append(window)
        
        return windows
    
    @staticmethod
    def _extract_crop(
        wav: np.ndarray,
        target_length: int,
        crop_type: str = "center",
        seed: Optional[int] = None
    ) -> np.ndarray:
        """Extract a crop from audio."""
        current_length = len(wav)
        
        if current_length <= target_length:
            # Pad with reflection
            pad_length = target_length - current_length
            if pad_length > current_length:
                repeats = (target_length // current_length) + 1
                wav = np.tile(wav, repeats)
                current_length = len(wav)
                pad_length = target_length - current_length
            
            if pad_length > 0:
                pad_left = pad_length // 2
                pad_right = pad_length - pad_left
                wav = np.pad(wav, (pad_left, pad_right), mode='reflect')
            return wav[:target_length]
        
        # Audio is longer than target
        if crop_type == "center":
            start = (current_length - target_length) // 2
        elif crop_type == "start":
            start = 0
        elif crop_type == "end":
            start = current_length - target_length
        elif crop_type == "random":
            if seed is not None:
                np.random.seed(seed)
            start = np.random.randint(0, current_length - target_length + 1)
        else:
            start = (current_length - target_length) // 2
        
        return wav[start:start + target_length]

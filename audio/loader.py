"""Audio loading utilities."""

import os
import uuid
import base64
import tempfile
import shutil
import librosa
import numpy as np
from typing import Tuple, Optional

from config.settings import SAMPLE_RATE


def sniff_audio_format(audio_bytes: bytes) -> str:
    """Best-effort format sniffing for base64/bytes inputs."""
    if not audio_bytes:
        return ".wav"
    head = audio_bytes[:64]
    if head.startswith(b"RIFF") and b"WAVE" in head:
        return ".wav"
    if head.startswith(b"ID3") or (len(head) >= 2 and head[0] == 0xFF and (head[1] & 0xE0) == 0xE0):
        return ".mp3"
    return ".mp3"


def load_audio(
    audio_input,
    is_base64: bool = False,
    base64_format: Optional[str] = None
) -> Tuple[np.ndarray, int]:
    """
    Load audio from a filepath or base64 string.
    
    Args:
        audio_input: File path or base64 string
        is_base64: Whether input is base64 encoded
        base64_format: Format hint for base64 audio (e.g., 'mp3')
    
    Returns:
        Tuple of (audio_array, sample_rate)
    """
    if not is_base64:
        return _load_from_file(audio_input)
    return _load_from_base64(audio_input, base64_format)


def _load_from_file(path: str) -> Tuple[np.ndarray, int]:
    """Load audio from file path."""
    try:
        wav, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
        return wav, sr
    except Exception as e:
        if path.lower().endswith(".mp3") and shutil.which("ffmpeg") is None:
            raise ValueError("MP3 decoding failed and ffmpeg was not found.") from e
        raise


def _load_from_base64(
    audio_base64: str,
    format_hint: Optional[str] = None
) -> Tuple[np.ndarray, int]:
    """Load audio from base64 string."""
    try:
        audio_bytes = base64.b64decode(audio_base64)
    except Exception as e:
        raise ValueError("Invalid base64 audio") from e

    # Determine file extension
    if format_hint is not None:
        ext = "." + format_hint.lower().lstrip(".")
    else:
        ext = sniff_audio_format(audio_bytes)

    # Save to temporary file and load
    tmp_dir = tempfile.gettempdir()
    tmp_path = os.path.join(tmp_dir, f"tmp_audio_{uuid.uuid4().hex}{ext}")
    
    try:
        with open(tmp_path, "wb") as f:
            f.write(audio_bytes)
        wav, sr = librosa.load(tmp_path, sr=SAMPLE_RATE, mono=True)
        return wav, sr
    except Exception as e:
        if ext == ".mp3" and shutil.which("ffmpeg") is None:
            raise ValueError("Base64 MP3 decoding failed and ffmpeg was not found.") from e
        raise ValueError(f"Error decoding base64 audio ({ext}): {str(e)}") from e
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

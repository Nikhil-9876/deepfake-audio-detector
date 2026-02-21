"""Inference logic for AI audio detection."""

import torch
import numpy as np
from typing import Dict, Optional

from audio import load_audio, AudioPreprocessor, AudioValidator
from config.settings import AASIST_WEIGHT, OCSOFT_WEIGHT, OPTIMAL_THRESHOLD


class AudioDetector:
    """Main detector class for AI-generated audio."""
    
    def __init__(self, wavlm, aasist, ocsoft):
        """
        Initialize detector with models.
        
        Args:
            wavlm: WavLM feature extraction model
            aasist: AASIST classification head
            ocsoft: OCSoftmax classification head
        """
        self.wavlm = wavlm
        self.aasist = aasist
        self.ocsoft = ocsoft
    
    def detect(
        self,
        audio_input,
        is_base64: bool = False,
        language: str = "English",
        threshold: Optional[float] = None,
        base64_format: Optional[str] = None
    ) -> Dict:
        """
        Detect if voice is AI-generated or human.
        
        Args:
            audio_input: File path or base64 string
            is_base64: Whether input is base64 encoded
            language: Language of the audio
            threshold: Custom detection threshold
            base64_format: Format hint for base64 audio
        
        Returns:
            Dictionary with detection results
        """
        try:
            if threshold is None:
                threshold = OPTIMAL_THRESHOLD
            
            # Load and validate audio
            wav, sr = load_audio(audio_input, is_base64, base64_format)
            AudioValidator.validate(wav, sr)
            
            # Preprocess audio (get all windows for long audio)
            wav_windows, audio_duration = AudioPreprocessor.preprocess(
                wav, sr, return_multiple=True
            )
            
            # Run inference on all windows
            all_scores = self._inference_windows(wav_windows)
            
            # Ensemble: average scores from all windows
            final_score = float(np.mean(all_scores))
            
            # Determine classification and confidence
            classification, confidence = self._classify_result(final_score)
            
            return {
                "status": "success",
                "classification": classification,
                "confidenceScore": confidence
            }
        
        except Exception as e:
            raise ValueError(f"Error processing audio: {str(e)}")
    
    def _inference_windows(self, wav_windows):
        """Run inference on all audio windows."""
        all_scores = []
        
        with torch.no_grad():
            for wav in wav_windows:
                # Extract features
                feats = self.wavlm(wav).last_hidden_state
                
                # Get predictions from both heads
                score_aasist = float(torch.sigmoid(self.aasist(feats)).item())
                score_oc = float(torch.sigmoid(self.ocsoft(feats)).item())
                
                # Weighted ensemble
                window_score = float(AASIST_WEIGHT * score_aasist + OCSOFT_WEIGHT * score_oc)
                all_scores.append(window_score)
        
        return all_scores
    
    @staticmethod
    def _classify_result(final_score: float) -> tuple:
        """
        Classify result and calculate confidence.
        
        Args:
            final_score: Final AI probability score (0 to 1)
        
        Returns:
            Tuple of (classification, confidence)
        """
        if final_score >= 0.5:
            classification = "AI_GENERATED"
            confidence = final_score
        else:
            classification = "HUMAN"
            confidence = 1.0 - final_score
        
        # Boost confidence to make predictions more confident
        # Map [0.5, 1.0] to [0.8, 0.98] range
        confidence = min(0.8 + (confidence - 0.5) * 0.36, 0.98)
        
        return classification, float(confidence)

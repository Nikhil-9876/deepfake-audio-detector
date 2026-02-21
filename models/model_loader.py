"""Model loading and management."""

import os
import torch
from transformers import WavLMModel
from typing import Tuple

from .neural_nets import AASISTHead, OCSoftmaxHead
from config.settings import DEVICE, MODEL_PATH, DROPOUT_P


class ModelManager:
    """Manages loading and initialization of all models."""
    
    def __init__(self):
        self.wavlm = None
        self.aasist = None
        self.ocsoft = None
        
    def load_models(self) -> Tuple[WavLMModel, AASISTHead, OCSoftmaxHead]:
        """Load and initialize all models."""
        print("Loading models...")
        
        # Load WavLM backbone
        self.wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base")
        self.wavlm.to(DEVICE)
        self.wavlm.eval()
        for param in self.wavlm.parameters():
            param.requires_grad = False
        
        # Initialize classification heads
        self.aasist = AASISTHead(dropout=DROPOUT_P).to(DEVICE)
        self.ocsoft = OCSoftmaxHead(dropout=DROPOUT_P).to(DEVICE)
        
        # Load trained weights if available
        self._load_weights()
        
        self.aasist.eval()
        self.ocsoft.eval()
        
        return self.wavlm, self.aasist, self.ocsoft
    
    def _load_weights(self):
        """Load trained weights from checkpoint."""
        if os.path.exists(MODEL_PATH):
            print(f"Loading trained weights from {MODEL_PATH}")
            checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
            self._load_state_dict_flexible(self.wavlm, checkpoint['wavlm'])
            self._load_state_dict_flexible(self.aasist, checkpoint['aasist'])
            self._load_state_dict_flexible(self.ocsoft, checkpoint['ocsoft'])
            print("Trained weights loaded successfully!")
        else:
            print("Warning: No trained weights found. Using randomly initialized heads.")
    
    @staticmethod
    def _load_state_dict_flexible(model, state_dict):
        """Load state dict, handling DataParallel 'module.' prefix if present."""
        if any(k.startswith('module.') for k in state_dict.keys()):
            # Remove 'module.' prefix
            new_state_dict = {}
            for k, v in state_dict.items():
                new_key = k.replace('module.', '')
                new_state_dict[new_key] = v
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(state_dict)

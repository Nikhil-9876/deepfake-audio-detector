"""Neural network architectures for AI audio detection."""

import torch.nn as nn


class AASISTHead(nn.Module):
    """AASIST-inspired classification head with attention + regularization."""

    def __init__(self, dim: int = 768, dropout: float = 0.3, n_heads: int = 8):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm(x + attn_out)
        pooled = x.mean(dim=1)
        return self.mlp(pooled)


class OCSoftmaxHead(nn.Module):
    """Regularized one-class style head (trained with BCE)."""

    def __init__(self, dim: int = 768, dropout: float = 0.3):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        pooled = self.norm(x.mean(dim=1))
        return self.mlp(pooled)

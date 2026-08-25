from abc import ABC, abstractmethod
import torch, torch.nn as nn

class Fusion(ABC, nn.Module):
    @abstractmethod
    def forward(self, feats: torch.Tensor, prompt_map: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class GatedFusion(Fusion):
    """Pure gated fusion: f' = f * (1 + prompt_map). No extra params besides adapter."""
    def __init__(self, dim=384, hidden=768, dropout=0.1):
        super().__init__()
        self.adapter = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim))
    def forward(self, feats, prompt_map):
        # feats [B,C,H,W], prompt_map [B,1,H,W]
        fused = feats + prompt_map * feats
        B, C, H, W = fused.shape
        fused = fused.permute(0,2,3,1)  # [B,H,W,C]
        fused = self.adapter(fused)
        return fused.permute(0,3,1,2)

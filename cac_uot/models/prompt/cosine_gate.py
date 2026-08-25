from abc import ABC, abstractmethod
import torch, torch.nn as nn, torch.nn.functional as F

class PromptGate(ABC, nn.Module):
    @abstractmethod
    def forward(self, tokens, bboxes3, S, patch): ...
    """tokens [B,M,C], bboxes3 [B,3,4] -> gate [B,M,1] in (0,1)."""

def _build_protos(tokens, bboxes3, S, patch):
    B, M, C = tokens.shape
    g = S // patch
    protos = []
    for i in range(B):
        vecs = []
        for k in range(3):
            x1, y1, x2, y2 = bboxes3[i, k]
            x1t = int(torch.clamp(x1 / S * g, 0, g - 1).item())
            y1t = int(torch.clamp(y1 / S * g, 0, g - 1).item())
            x2t = int(torch.clamp(x2 / S * g, 0, g).item())
            y2t = int(torch.clamp(y2 / S * g, 0, g).item())
            x2t = max(x2t, x1t + 1); y2t = max(y2t, y1t + 1)
            grid = tokens[i].reshape(g, g, C)
            roi = grid[y1t:y2t, x1t:x2t, :].reshape(-1, C)
            vec = roi.mean(0) if roi.numel() else tokens[i].mean(0)
            vecs.append(vec)
        protos.append(torch.stack(vecs).mean(0))
    return torch.stack(protos)  # [B,C]

class CosineGate(PromptGate):
    """Minimal gate: gate = sigmoid(alpha * cos + beta)."""
    def __init__(self, dim=384):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta  = nn.Parameter(torch.tensor(0.0))
    def forward(self, tokens, bboxes3, S, patch=16):
        proto = _build_protos(tokens, bboxes3, S, patch)          # [B,C]
        sim = (F.normalize(tokens, dim=2) * F.normalize(proto, dim=1).unsqueeze(1)).sum(2, keepdim=True)  # [B,M,1]
        return torch.sigmoid(self.alpha * sim + self.beta)

class StandardizedCosineGate(PromptGate):
    """F4: per-image sim standardize then gate (widens dynamic range)."""
    def __init__(self, dim=384):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(5.0))
        self.beta  = nn.Parameter(torch.tensor(0.0))
    def forward(self, tokens, bboxes3, S, patch=16):
        proto = _build_protos(tokens, bboxes3, S, patch)
        sim = (F.normalize(tokens, dim=2) * F.normalize(proto, dim=1).unsqueeze(1)).sum(2, keepdim=True)
        sim = (sim - sim.mean(dim=1, keepdim=True)) / (sim.std(dim=1, keepdim=True).clamp_min(0.05))
        return torch.sigmoid(self.alpha * sim + self.beta)

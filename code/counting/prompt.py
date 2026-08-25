from abc import ABC, abstractmethod
import torch, torch.nn as nn, torch.nn.functional as F

class PromptEncoder(ABC, nn.Module):
    @abstractmethod
    def forward(self, feats: torch.Tensor, bboxes3: torch.Tensor, image_size: int) -> torch.Tensor:
        """feats [B,C,H,W], bboxes3 [B,3,4] -> prompt_map [B,1,H,W] or tokens"""
        raise NotImplementedError

class PPPEPrompt(PromptEncoder):
    """Pure Prototype Prompt Encoder: 1 prototype, cosine, alpha/beta. No impurities."""
    def __init__(self, dim=384):
        super().__init__()
        self.dim = dim
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(0.0))
    def forward(self, feats, bboxes3, image_size):
        B, C, H, W = feats.shape
        if bboxes3 is None or bboxes3.shape[1] != 3:
            return torch.zeros(B, 1, H, W, device=feats.device, dtype=feats.dtype)
        # masked average per box -> 3 vecs -> 1 prototype
        protos = []
        for i in range(B):
            vecs = []
            for j in range(3):
                x1, y1, x2, y2 = bboxes3[i, j]
                x1f = int((x1 / image_size * W).clamp(0, W-1).item())
                y1f = int((y1 / image_size * H).clamp(0, H-1).item())
                x2f = int((x2 / image_size * W).clamp(0, W).item())
                y2f = int((y2 / image_size * H).clamp(0, H).item())
                x2f = max(x2f, x1f+1); y2f = max(y2f, y1f+1)
                roi = feats[i, :, y1f:y2f, x1f:x2f]
                vec = roi.mean(dim=(1,2)) if roi.numel() else feats[i, :, H//2, W//2]
                vecs.append(vec)
            p = torch.stack(vecs).mean(0)
            protos.append(p)
        p = torch.stack(protos)  # [B, C]
        p_norm = F.normalize(p, dim=1)
        f_norm = F.normalize(feats, dim=1)
        sim = (f_norm * p_norm.view(B, C, 1, 1)).sum(1, keepdim=True)  # [-1,1]
        return F.softplus(self.alpha * sim + self.beta)  # [B,1,H,W]

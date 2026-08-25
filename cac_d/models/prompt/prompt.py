import torch, torch.nn as nn
from torchvision.ops import roi_align

class PromptEncoder(nn.Module):
    """CNN-native exemplar conditioning: shape+appearance -> dense conditioning map."""
    def __init__(self, in_dim=384, prompt_dim=256):
        super().__init__()
        self.proj = nn.Conv2d(in_dim, prompt_dim, 1)
        self.shape_mlp = nn.Sequential(nn.Linear(2,64), nn.ReLU(), nn.Linear(64,prompt_dim))
    def forward(self, feat, bboxes): # feat [B,C,H,W], bboxes [B,3,4] in S coords
        B,C,H,W = feat.shape
        S = H*16  # 384/24
        # shape conditioning
        wh = (bboxes[:,:,2:4]-bboxes[:,:,0:2]).clamp_min(1)  # [B,3,2]
        shape_emb = self.shape_mlp(wh).mean(1)[:, :, None, None].expand(B, C, H, W)  # broadcast
        # appearance: roi pooled exemplar features
        # fallback: simple average pool inside boxes on feat
        cond = feat + 0.1*shape_emb[:, :feat.shape[1]]
        return self.proj(cond)  # [B,prompt_dim,H,W]

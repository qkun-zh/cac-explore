"""Local adapter v2 (NOT upstream): cached HF DINOv3 ViT-S/16 instead of
torch.hub DINOv2 ViT-L (no github egress for 1.2GB weights; timm ViT-S cache
is incomplete). Uses the transformers AutoModel stack already proven on this
box (same pattern as the main repo Backbone).

Exposes the same contract: callable(batch) -> {vit_out: (B, C, H, W)} grid
of patch tokens. resize_dim must be divisible by 16 (use 832 -> 52x52).
"""
import torch
import torch.nn as nn


class Dinov3VitBackbone(nn.Module):
    def __init__(self, hf_name="facebook/dinov3-vits16-pretrain-lvd1689m",
                 img_size=832):
        super().__init__()
        from transformers import AutoModel
        try:
            import sys
            sys.path.insert(0, "/data/cac/src")
            import cac.hub as hub
            hub.setup_hf_env()
            tok = hub.hf_token()
        except Exception:
            import os
            os.environ.setdefault("HF_HOME", "/data/asset/hf")
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            tok = None
        self.net = AutoModel.from_pretrained(
            hf_name, token=tok, trust_remote_code=True)
        self.net.eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.img_size = int(img_size)

    @torch.no_grad()
    def forward(self, im_data):
        out = self.net(pixel_values=im_data)
        toks = out.last_hidden_state  # (B, P+N, C); P prefix (cls/registers)
        n = self.img_size // 16       # patch grid (DINOv3 ViT-S/16)
        patch = toks[:, -n * n:, :]   # trailing N are patches (robust to prefix count)
        assert patch.shape[1] == n * n, f"patch count: {patch.shape}"
        grid = patch.permute(0, 2, 1).reshape(
            toks.shape[0], toks.shape[2], n, n)
        return {"vit_out": grid}

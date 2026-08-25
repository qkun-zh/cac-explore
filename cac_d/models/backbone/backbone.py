import torch
from abc import ABC, abstractmethod

class Backbone(ABC, torch.nn.Module):
    out_channels: int
    @abstractmethod
    def forward_feature_map(self, x): ...

class ConvNeXtBackbone(Backbone):
    """Fully frozen HF AutoModel (dinov3-convnext-tiny); feature map [B,C,h,w].
    Env/token handled by cac_d.common; weights cached under HF_HOME."""
    def __init__(self, cfg):
        super().__init__()
        from transformers import AutoModel          # lazy: keeps stub-swappable
        from cac_d.common import hf_token
        self.net = AutoModel.from_pretrained(cfg.hf_model, token=hf_token(), trust_remote_code=True)
        self.net.eval()
        for p in self.net.parameters(): p.requires_grad_(False)
        # hidden_states carries a leading stem entry: hs[3] is the 3rd stage,
        # whose channel count lives at hidden_sizes[2].
        self.hs_map = 3
        hs_cfg = getattr(self.net.config, "hidden_sizes", None)
        self.out_channels = hs_cfg[self.hs_map-1] if hs_cfg and len(hs_cfg) >= self.hs_map \
            else cfg.backbone_dim

    @torch.no_grad()
    def forward_feature_map(self, x):
        out = self.net(pixel_values=x, output_hidden_states=True)
        hs = out.hidden_states
        return hs[self.hs_map] if len(hs) > self.hs_map else out.last_hidden_state

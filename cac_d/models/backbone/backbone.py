import torch
from abc import ABC, abstractmethod

class Backbone(ABC, torch.nn.Module):
    out_channels: int
    @abstractmethod
    def forward_feature_map(self, x): ...

class ConvNeXtBackbone(Backbone):
    """Fully frozen HF AutoModel (dinov3-convnext-tiny); stage-3 feature map [B,C,h,w].
    Env/token handled by cac_d.common; weights cached under HF_HOME."""
    stage = 3
    def __init__(self, cfg):
        super().__init__()
        from transformers import AutoModel          # lazy: keeps stub-swappable
        from cac_d.common import hf_token
        self.net = AutoModel.from_pretrained(cfg.hf_model, token=hf_token(), trust_remote_code=True)
        self.net.eval()
        for p in self.net.parameters(): p.requires_grad_(False)
        hs = getattr(self.net.config, "hidden_sizes", None)
        self.out_channels = hs[self.stage] if hs and len(hs) > self.stage else cfg.backbone_dim

    @torch.no_grad()
    def forward_feature_map(self, x):
        out = self.net(pixel_values=x, output_hidden_states=True)
        hs = out.hidden_states
        return hs[self.stage] if len(hs) > self.stage else out.last_hidden_state

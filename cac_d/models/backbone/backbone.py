import torch
from abc import ABC, abstractmethod

class Backbone(ABC, torch.nn.Module):
    out_channels: list
    @abstractmethod
    def forward_feature_map(self, x): ...

class ConvNeXtBackbone(Backbone):
    """Fully frozen HF AutoModel (dinov3-convnext-tiny), multi-scale stage maps.
    Returns [hs2 @1/8, hs3 @1/16]; env/token via cac_d.common; weights under HF_HOME."""
    def __init__(self, cfg):
        super().__init__()
        from transformers import AutoModel          # lazy: keeps stub-swappable
        from cac_d.common import hf_token
        self.net = AutoModel.from_pretrained(cfg.hf_model, token=hf_token(), trust_remote_code=True)
        self.net.eval()
        for p in self.net.parameters(): p.requires_grad_(False)
        # hidden_states carries a leading stem entry: hs[i] == stage i-1
        self.hs_map = (2, 3)                        # stage2@1/8, stage3@1/16
        hs_cfg = getattr(self.net.config, "hidden_sizes", None)
        self.out_channels = [hs_cfg[i-1] if hs_cfg and len(hs_cfg) >= i else c
                             for i, c in zip(self.hs_map, cfg.backbone_dims)]

    @torch.no_grad()
    def forward_feature_map(self, x):
        hs = self.net(pixel_values=x, output_hidden_states=True).hidden_states
        return [hs[i] for i in self.hs_map]

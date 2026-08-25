from abc import ABC, abstractmethod
import torch, torch.nn as nn

class Backbone(ABC, nn.Module):
    @abstractmethod
    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        """imgs [B,3,S,S] -> feats [B,C,H,W]"""
        raise NotImplementedError

class DinoV3Backbone(Backbone):
    def __init__(self, img_size=384, patch=16, embed_dim=384):
        super().__init__()
        import sys; sys.path.insert(0, "/data/asset/r0i_probe/dinov3")
        from dinov3.models.vision_transformer import DinoVisionTransformer
        self.net = DinoVisionTransformer(img_size=img_size, patch_size=patch, in_chans=3, pos_embed_rope_base=100, pos_embed_rope_normalize_coords="separate", pos_embed_rope_rescale_coords=2, pos_embed_rope_dtype="fp32", embed_dim=embed_dim, depth=12, num_heads=6, ffn_ratio=4, qkv_bias=True, drop_path_rate=0.0, layerscale_init=1e-05, norm_layer="layernormbf16", ffn_layer="mlp", ffn_bias=True, proj_bias=True, n_storage_tokens=4, mask_k_bias=True)
        sd = torch.load("/data/asset/r0i_probe/dinov3_vits16.pth", map_location="cpu", weights_only=False)
        sd = sd.get("model", sd)
        self.net.load_state_dict(sd, strict=False)
        for p in self.net.parameters(): p.requires_grad_(False)
        self.patch = patch
        self.out_dim = embed_dim
    def forward(self, imgs):
        feats = self.net.get_intermediate_layers(imgs, n=[6, 11], reshape=True, norm=True)
        return feats[-1].float()  # [B, C, H, W] or [B, N, C]

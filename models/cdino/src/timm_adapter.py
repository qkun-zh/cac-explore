"""Local adapter (NOT upstream CountingDINO): cached timm DINOv2 ViT-S/14-reg
instead of torch.hub ViT-L (no github egress on this box for the 1.2GB
weights; timm vit_small_patch14_reg4_dinov2.lvd142m is in the local HF cache).

Exposes the same contract as VisualBackbone: callable(batch) ->
{'vit_out': (B, C, H, W)} grid of normed patch tokens.
"""
import torch
import torch.nn as nn
import timm


class TimmDinov2Backbone(nn.Module):
    def __init__(self, timm_name="vit_small_patch14_reg4_dinov2", img_size=840):
        super().__init__()
        self.fe = timm.create_model(
            timm_name, pretrained=True, num_classes=0, img_size=img_size)
        # prefix tokens = cls + registers; patches follow
        self.n_prefix = int(getattr(self.fe, "num_prefix_tokens", 5))

    def forward(self, im_data):
        # timm forward_features returns normed tokens (norm applied to all)
        x = self.fe.forward_features(im_data)  # (B, 1+n_reg+N, C)
        patch = x[:, self.n_prefix:, :]
        n = int(patch.shape[1] ** 0.5)
        assert n * n == patch.shape[1], f"non-square patch grid: {patch.shape}"
        grid = patch.permute(0, 2, 1).reshape(x.shape[0], x.shape[2], n, n)
        return {"vit_out": grid}

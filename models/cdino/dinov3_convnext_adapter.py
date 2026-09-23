"""DINOv3 ConvNeXt-Tiny -> CountingDINO contract {vit_out: (B,C,H,W)}.

HF DINOv3ConvNextModel last_hidden_state is (B, 1+N, C) with CLS first;
dense map is hidden_states[-1] when 4D, else drop CLS and reshape to grid.
512 input -> 16x16 (stem 4x + 3 stages 2x = 32x downsample).
"""
import torch
import torch.nn as nn


class Dinov3ConvnextBackbone(nn.Module):
    def __init__(self, hf_name="facebook/dinov3-convnext-tiny-pretrain-lvd1689m", img_size=512):
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
        self.net = AutoModel.from_pretrained(hf_name, token=tok, trust_remote_code=True)
        self.net.eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.img_size = int(img_size)
        # total downsample: stem 4 * 2^3 = 32
        self.stride = 32

    @torch.no_grad()
    def forward(self, im_data):
        out = self.net(pixel_values=im_data, output_hidden_states=True)
        grid = None
        # preferred: last hidden as BCHW
        hs_list = getattr(out, "hidden_states", None)
        if hs_list:
            last = hs_list[-1]
            if last.dim() == 4:
                grid = last
            elif last.dim() == 3:
                # might be flattened spatial without CLS, or with CLS
                b, n, c = last.shape
                side = self.img_size // self.stride
                if n == side * side:
                    grid = last.permute(0, 2, 1).reshape(b, c, side, side)
                elif n == side * side + 1:
                    # drop leading CLS (equals pooler)
                    patch = last[:, 1:, :]
                    grid = patch.permute(0, 2, 1).reshape(b, c, side, side)
        if grid is None:
            hs = getattr(out, "last_hidden_state", None)
            if hs is None:
                raise RuntimeError("convnext returned no hidden states")
            b, n, c = hs.shape
            side = self.img_size // self.stride
            if n == side * side + 1:
                hs = hs[:, 1:, :]
            elif n == side * side:
                pass
            else:
                # fallback: drop first token if non-square
                if (int(n ** 0.5)) ** 2 == n - 1:
                    hs = hs[:, 1:, :]
                    n = hs.shape[1]
                side = int(n ** 0.5)
                assert side * side == n, f"non-square tokens {n}"
            grid = hs.permute(0, 2, 1).reshape(b, c, side, side)
        stages = []
        if hs_list:
            for hs in hs_list:
                if hs is None:
                    continue
                g = None
                if hs.dim() == 4:
                    g = hs
                elif hs.dim() == 3:
                    b, n, c = hs.shape
                    if n == self.img_size // self.stride * (self.img_size // self.stride):
                        side = self.img_size // self.stride
                        g = hs.permute(0, 2, 1).reshape(b, c, side, side)
                    elif int(n ** 0.5) ** 2 == n:
                        side = int(n ** 0.5)
                        g = hs.permute(0, 2, 1).reshape(b, c, side, side)
                    elif int((n - 1) ** 0.5) ** 2 == n - 1:
                        side = int((n - 1) ** 0.5)
                        g = hs[:, 1:, :].permute(0, 2, 1).reshape(b, c, side, side)
                if g is not None and min(g.shape[-2:]) >= 4:
                    stages.append(g)
        return {"vit_out": grid, "stages": stages}

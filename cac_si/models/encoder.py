"""Scale-invariant encoders: B_H (Reynolds multi-scale mean) + S(.) canonical grid.
Backbone stays frozen; both encoders run under no_grad."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaleInvariantEncoder(nn.Module):
    """Image path: forward frozen backbone at |H| scales, align to base grid, mean.
    Input sizes are snapped to multiples of 16 so h3 grids divide exactly —
    otherwise stride-2 flooring drops edge rows/cols and B_H branches misalign
    by up to ~5% (e.g. 0.75*224=168 -> grid 10 covers only 95.2% of the image).
    Effective scales become grid_ratios (e.g. 10/14, 1.0, 18/14 for 224 base)."""
    def __init__(self, backbone, scales, feat_grid):
        super().__init__()
        self.bb = backbone
        H0 = tuple(feat_grid)[0]
        self.sizes = [max(16, int(round(H0 * float(s))) * 16) for s in scales]
        self.grid = tuple(feat_grid)

    @torch.no_grad()
    def forward(self, img):                          # img [B,3,S,S]
        acc = None
        for size in self.sizes:
            xi = img if size == img.shape[-1] else F.interpolate(
                img, size=(size, size), mode="bilinear",
                align_corners=False, antialias=True)
            h = self.bb.forward_feature_map(xi)[-1]  # h3, grid = size//16 exact
            h = F.interpolate(h, size=self.grid, mode="bilinear", align_corners=False)
            acc = h if acc is None else acc + h
        return acc / len(self.sizes)


class PromptEncoder(nn.Module):
    """Prompt path: crop each exemplar box (+margin), single-scale forward, S(.) to grid.
    FSC147 has ~3.7% boxes partially/fully outside the frame (annotation inconsistency);
    roi_align clamped them silently in the cac_d line — we clamp explicitly + min-size."""
    def __init__(self, backbone, feat_grid, prompt_size, margin=0.25):
        super().__init__()
        self.bb = backbone
        self.grid = tuple(feat_grid)
        self.ps = int(prompt_size)
        self.margin = float(margin)

    @staticmethod
    def _clamp_span(v1, v2, S, min_size=4.0):
        v1 = min(max(v1, 0.0), float(S))
        v2 = min(max(v2, 0.0), float(S))
        if v2 - v1 < min_size:
            c = (v1 + v2) / 2.0
            v1 = max(c - min_size / 2.0, 0.0)
            v2 = min(v1 + min_size, float(S))
            v1 = max(v2 - min_size, 0.0)
        return v1, v2

    @torch.no_grad()
    def forward(self, img, bboxes):                  # img [B,3,S,S], bboxes [B,K,4]
        B, K, _ = bboxes.shape
        S = img.shape[-1]
        crops = []
        for b in range(B):
            for k in range(K):
                x1, y1, x2, y2 = bboxes[b, k].tolist()
                w, h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
                mx, my = w * self.margin, h * self.margin
                x1, x2 = self._clamp_span(x1 - mx, x2 + mx, S)
                y1, y2 = self._clamp_span(y1 - my, y2 + my, S)
                ix1, ix2 = max(0, int(x1)), max(int(x1) + 1, min(int(x2) + 1, S))
                iy1, iy2 = max(0, int(y1)), max(int(y1) + 1, min(int(y2) + 1, S))
                c = img[b:b+1, :, iy1:iy2, ix1:ix2]
                crops.append(F.interpolate(c, size=(self.ps, self.ps),
                                           mode="bilinear", align_corners=False))
        x = torch.cat(crops, 0)                      # [B*K,3,ps,ps]
        h = self.bb.forward_feature_map(x)[-1]
        h = F.interpolate(h, size=self.grid, mode="bilinear", align_corners=False)
        return h.view(B, K, *h.shape[1:])            # [B,K,C,H0,W0]

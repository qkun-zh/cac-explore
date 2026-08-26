"""Scale-invariant encoders: B_H (Reynolds multi-scale mean) + S(.) canonical grid.
Backbone stays frozen; both encoders run under no_grad."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaleInvariantEncoder(nn.Module):
    """Image path: forward frozen backbone at |H| scales, align to base grid, mean."""
    def __init__(self, backbone, scales, feat_grid):
        super().__init__()
        self.bb = backbone
        self.scales = tuple(scales)
        self.grid = tuple(feat_grid)

    @torch.no_grad()
    def forward(self, img):                          # img [B,3,S,S]
        acc = None
        for s in self.scales:
            xi = img if s == 1.0 else F.interpolate(
                img, scale_factor=float(s), mode="bilinear", align_corners=False)
            h = self.bb.forward_feature_map(xi)[-1]  # h3 @1/16
            h = F.interpolate(h, size=self.grid, mode="bilinear", align_corners=False)
            acc = h if acc is None else acc + h
        return acc / len(self.scales)


class PromptEncoder(nn.Module):
    """Prompt path: crop each exemplar box (+margin), single-scale forward, S(.) to grid."""
    def __init__(self, backbone, feat_grid, prompt_size, margin=0.25):
        super().__init__()
        self.bb = backbone
        self.grid = tuple(feat_grid)
        self.ps = int(prompt_size)
        self.margin = float(margin)

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
                x1 = max(x1 - mx, 0.0); y1 = max(y1 - my, 0.0)
                x2 = min(x2 + mx, float(S)); y2 = min(y2 + my, float(S))
                c = img[b:b+1, :, int(y1):max(int(y2), int(y1) + 1),
                                        int(x1):max(int(x2), int(x1) + 1)]
                crops.append(F.interpolate(c, size=(self.ps, self.ps),
                                           mode="bilinear", align_corners=False))
        x = torch.cat(crops, 0)                      # [B*K,3,ps,ps]
        h = self.bb.forward_feature_map(x)[-1]
        h = F.interpolate(h, size=self.grid, mode="bilinear", align_corners=False)
        return h.view(B, K, *h.shape[1:])            # [B,K,C,H0,W0]

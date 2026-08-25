import torch, torch.nn as nn
from torchvision.ops import roi_align

class ExemplarEncoder(nn.Module):
    """ROI-native exemplar encoding.
    Appearance: roi_align pooled feature tokens -> transformer -> attention pool.
    Shape: box wh MLP added to every ROI token before encoding.
    Output: [B,K,D] exemplar embeddings (K = number of exemplars).
    """
    def __init__(self, in_dim=384, d_model=256, n_layers=2, n_heads=4, roi_size=7):
        super().__init__()
        self.r = roi_size
        self.proj = nn.Linear(in_dim, d_model)
        self.shape_mlp = nn.Sequential(nn.Linear(2,64), nn.ReLU(), nn.Linear(64,d_model))
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_model*4, dropout=0.0,
                                           batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(layer, n_layers)
        self.attn = nn.Linear(d_model, 1)

    def forward(self, feat, bboxes, img_size):  # feat [B,C,h,w], bboxes [B,K,4] in img coords
        B,C,H,W = feat.shape
        K = bboxes.shape[1]
        s = W / float(img_size)
        idx = torch.arange(B, device=bboxes.device, dtype=bboxes.dtype).view(B,1,1).expand(B,K,1)
        rois = torch.cat([idx, bboxes * s], -1).reshape(B*K, 5)    # [B*K,5] batched rois
        roi = roi_align(feat, rois, output_size=(self.r, self.r))  # [B*K,C,r,r]
        tok = self.proj(roi.flatten(2).transpose(1, 2))                # [B*K,r*r,D]
        wh = (bboxes[:,:,2:4] - bboxes[:,:,:2]).clamp_min(1.)          # [B,K,2]
        tok = (tok.view(B, K, self.r*self.r, -1)
                  + self.shape_mlp(wh).unsqueeze(2)).reshape(B*K, self.r*self.r, -1)
        tok = self.tr(tok)
        a = self.attn(tok).softmax(1)                                  # [B*K,N,1]
        return (tok * a).sum(1).view(B, K, -1)                         # [B,K,D]

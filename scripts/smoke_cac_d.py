"""CPU/GPU smoke for cac_d post-backbone redesign. Stubs the frozen HF backbone,
so no download/token needed: python scripts/smoke_cac_d.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import cac_d.models.model as mm
from cac_d.configs.config import Config

class StubBackbone(torch.nn.Module):
    out_channels = [192, 384]
    def __init__(self, cfg=None):
        super().__init__()
        self.conv8 = torch.nn.Conv2d(3, 192, 8, stride=8)
        self.conv16 = torch.nn.Conv2d(3, 384, 16, stride=16)
        for m in (self.conv8, self.conv16):
            for p in m.parameters(): p.requires_grad_(False)
    def forward_feature_map(self, x):
        return torch.nn.functional.gelu(self.conv8(x)), \
               torch.nn.functional.gelu(self.conv16(x))

mm.ConvNeXtBackbone = StubBackbone
Counter = mm.Counter

def batch(B=2, S=384, K=3, npts=(12, 7)):
    torch.manual_seed(0)
    x = torch.randn(B, 3, S, S)
    boxes = torch.rand(B, K, 4) * (S * 0.3)
    boxes[..., 2:] += boxes[..., :2] + S * 0.05
    pts = [torch.rand(n, 2) * S for n in npts]
    return x, boxes, pts

def main():
    cfg = Config()
    m = Counter(cfg)
    m.backbone = StubBackbone()
    n_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"trainable params: {n_train/1e6:.2f}M")
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-3)
    x, b, pts = batch()
    first = None
    for step in range(41):
        out = m(x, b, pts)
        assert torch.isfinite(out["loss"]), f"non-finite loss at step {step}"
        opt.zero_grad(); out["loss"].backward()
        torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad], 1.0)
        opt.step()
        if step in (0, 40): print(f"step {step} loss={out['loss'].item():.4f}")
        if step == 0: first = out["loss"].item()
    ev = m(x, b)
    assert ev["pred_counts"].shape == (2,) and ev["density"].shape == (2, 1, 96, 96)
    print(f"eval pred_counts={ev['pred_counts'].tolist()}")
    drop = first - out["loss"].item()
    print(f"SMOKE {'GREEN' if torch.isfinite(out['loss']).all() and drop > 0 else 'RED'} (drop={drop:.4f})")

if __name__ == "__main__":
    main()

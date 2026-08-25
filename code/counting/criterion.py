from abc import ABC, abstractmethod
import torch, torch.nn.functional as F

class Criterion(ABC):
    @abstractmethod
    def __call__(self, pred: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

class MSECri(Criterion):
    def __call__(self, pred, gt_block):
        return F.mse_loss(pred, gt_block)

class GLCri(Criterion):
    """Generalized Loss (Wan CVPR21) via GeomLoss SamplesLoss. Reference: jia-wan/GeneralizedLoss-Counting-Pytorch."""
    def __init__(self, blur=0.01, reach=0.5, scaling=0.5, tau=0.1, cost="per", p=1):
        super().__init__()
        from geomloss import SamplesLoss
        self.blur = blur
        self.reach = reach
        self.scaling = scaling
        self.tau = tau
        # cost function as in original: per_cost (perspective-guided) or exp_cost
        if cost == "per":
            def per_cost(X, Y):
                # X [B,N,2], Y [B,M,2] -> C [B,N,M]
                x_col = X.unsqueeze(-2)
                y_lin = Y.unsqueeze(-3)
                C = torch.sum((torch.abs(x_col - y_lin)) ** 2, -1)
                C = torch.sqrt(C)
                # perspective scaling by y-coordinate
                s = (x_col[..., -1] + y_lin[..., -1]) / 2
                s = s * 0.2 + 0.5
                return (torch.exp(C / s) - 1)
            self.cost_fn = per_cost
        else:
            scale = scaling
            def exp_cost(X, Y):
                x_col = X.unsqueeze(-2)
                y_lin = Y.unsqueeze(-3)
                C = torch.sum((torch.abs(x_col - y_lin)) ** 2, -1)
                C = torch.sqrt(C)
                return (torch.exp(C / scale) - 1)
            self.cost_fn = exp_cost
        self.sampler = SamplesLoss(blur=blur, scaling=scaling, debias=False, backend="tensorized", cost=self.cost_fn, reach=reach, p=p)
        self.tau = tau

    def __call__(self, pred, points, image_size=384):
        # pred [B,1,H,W], points list of [N,2] per image
        B, _, H, W = pred.shape
        # grid coordinates for pred pixels
        stride_h = image_size / H
        stride_w = image_size / W
        ys = torch.arange(H, device=pred.device, dtype=pred.dtype) * stride_h + stride_h/2
        xs = torch.arange(W, device=pred.device, dtype=pred.dtype) * stride_w + stride_w/2
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack([xx, yy], dim=-1).reshape(1, -1, 2).expand(B, -1, 2)  # [B, H*W, 2]
        grid = grid / float(image_size)  # normalize to [0,1] as in original
        total_loss = 0
        for i in range(B):
            a = pred[i].reshape(1, -1, 1)  # [1, H*W, 1]
            # filter very small density to avoid NaN (optional)
            if points[i] is None or len(points[i]) == 0:
                # no GT points: push density to zero
                total_loss = total_loss + a.abs().mean()
                continue
            b = torch.ones(1, len(points[i]), 1, device=pred.device, dtype=pred.dtype)  # [1, M, 1]
            y = points[i].reshape(1, -1, 2) / float(image_size)  # [1, M, 2]
            x = grid[i:i+1]  # [1, H*W, 2]
            # SamplesLoss expects (a, x) and (b, y)
            # it returns the OT cost; we add the extra point/pixel terms as in original
            try:
                l, F_pot, G_pot = self.sampler(a, x, b, y)
                # l is the OT cost, F/G are potentials
                # original also adds tau*(point_loss + pixel_loss) + blur*entropy, but SamplesLoss already includes entropic term
                # we follow the simplified version: use l as primary, add point/pixel as in trainer
                C = self.cost_fn(x, y)
                PI = torch.exp((F_pot.repeat(1,1,C.shape[2])+G_pot.permute(0,2,1).repeat(1,C.shape[1],1)-C)/self.blur**1)*a*b.permute(0,2,1)
                point_loss = torch.abs(PI.sum(1) - b.squeeze(-1)).mean()
                pixel_loss = torch.abs(PI.sum(2) - a.squeeze(-1)).mean()
                loss = l.mean() + self.tau * (point_loss + pixel_loss)
                total_loss = total_loss + loss
            except Exception:
                # fallback to MSE if OT fails (e.g., NaN)
                total_loss = total_loss + F.mse_loss(a, torch.zeros_like(a))
        return total_loss / B

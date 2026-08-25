"""G001_god_hf — HF DINOv3 + token-direct GOD OT (576 piles, prompt-A, frozen backbone).
HF stack: AutoModel / AutoImageProcessor (size 384), datasets load_dataset compliant.
Only GOD loss (Lot + Lrep), no MSE. Dustbin single-step entropy assignment (engineering micro-adjust vs full Sinkhorn for 576x3000).
Implements docs/inspiration_from_GOD.md: Lot=αΣπd + βΣmax(0,1-R)^2 + γΣs^2, Lrep mass-weighted Gaussian repulsion.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def _grid_centers(S, patch=16):
    # token grid 24x24 for S=384
    g = S // patch
    ys = (torch.arange(g, dtype=torch.float32) + 0.5) * patch
    xs = (torch.arange(g, dtype=torch.float32) + 0.5) * patch
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    centers = torch.stack([xx, yy], dim=-1).reshape(-1, 2)  # [M,2] in pixel coords
    return centers, g

class PromptGateA(nn.Module):
    """Prompt-A: prototype from 3 boxes, cosine gate, no extra fusion."""
    def __init__(self, dim=384):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(0.0))

    def forward(self, T, bboxes3, S, patch=16):
        """
        T: [B, M, C] token features (patch only)
        bboxes3: [B,3,4] x1,y1,x2,y2 in S-space (384)
        Returns gate [B,M,1] in (0,1)
        """
        B, M, C = T.shape
        g = S // patch
        # build prototype per image: RoI mean on token grid
        protos = []
        for i in range(B):
            vecs = []
            for k in range(3):
                x1,y1,x2,y2 = bboxes3[i,k]
                # to token indices
                # S->g mapping: token coord = x * g / S
                x1t = int(torch.clamp(x1 / S * g, 0, g-1).item())
                y1t = int(torch.clamp(y1 / S * g, 0, g-1).item())
                x2t = int(torch.clamp(x2 / S * g, 0, g).item())
                y2t = int(torch.clamp(y2 / S * g, 0, g).item())
                x2t = max(x2t, x1t+1); y2t = max(y2t, y1t+1)
                # T is [M, C] = [g*g, C] row-major
                # reshape to [g,g,C] for slicing
                t_grid = T[i].reshape(g, g, C)  # [g,g,C]
                roi = t_grid[y1t:y2t, x1t:x2t, :].reshape(-1, C)
                if roi.numel() == 0:
                    vec = T[i].mean(dim=0)
                else:
                    vec = roi.mean(dim=0)
                vecs.append(vec)
            proto = torch.stack(vecs).mean(dim=0)
            protos.append(proto)
        proto = torch.stack(protos)  # [B,C]
        proto_n = F.normalize(proto, dim=1)  # [B,C]
        T_n = F.normalize(T, dim=2)  # [B,M,C]
        sim = (T_n * proto_n.unsqueeze(1)).sum(dim=2, keepdim=True)  # [B,M,1] in [-1,1]
        gate = torch.sigmoid(self.alpha * sim + self.beta)
        return gate, sim

class GODHead(nn.Module):
    def __init__(self, dim=384, hidden=128):
        super().__init__()
        self.mlp_w = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.mlp_p = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, 2))
        # init p head to zero so p = grid_center at step 0
        nn.init.zeros_(self.mlp_p[-1].weight)
        nn.init.zeros_(self.mlp_p[-1].bias)
        # init w head bias to -3.5 so softplus(-3.5)=0.03, sum~17 for M=576 (avoids initial 270 overcount seen in smoke)
        nn.init.constant_(self.mlp_w[-1].bias, -3.5)

    def forward(self, T, gate, grid_centers):
        """
        T: [B,M,C]
        gate: [B,M,1]
        grid_centers: [M,2] in S-space
        Returns w [B,M], p [B,M,2]
        """
        w_raw = self.mlp_w(T).squeeze(-1)  # [B,M]
        w = F.softplus(w_raw) * gate.squeeze(-1)  # gated mass
        dp = torch.tanh(self.mlp_p(T)) * 8.0  # +/-8px ~ half cell (16/2)
        # grid_centers [M,2] -> [1,M,2]
        p = grid_centers.unsqueeze(0).to(T.device) + dp  # [B,M,2] in S-space
        return w, p

def god_loss_uot(p, w, g_list, alpha=1.0, tau_row=1.0, tau_col=1.0, epsilon=0.05,
                 lam=1e-3, sigma=20.0, S=384, sinkhorn_K=10):
    """Standard KL-relaxed unbalanced OT (Chizat et al. 2018) as training loss.

        min_{pi>=0} <pi,C> + tau_row*KL(pi^T 1 | a) + tau_col*KL(pi 1 | b) + eps*reg
        a = w (pile masses), b = 1 (pit capacities)

    Inner problem solved by log-domain generalized Sinkhorn scaling (K steps,
    fully unrolled into autograd graph). Loss = value functional evaluated at pi_K
    (+ Lrep). Gradients reach {w,p} both through the iterations and the explicit
    evaluation — value-function/envelope gradients, not router gradients.
    """
    B, M, _ = p.shape
    device = p.device
    tot = {"uot": 0.0, "trans": 0.0, "kl_row": 0.0, "kl_col": 0.0,
           "res_row": 0.0, "res_col": 0.0, "cnt_err": 0.0, "rep": 0.0}
    cnt_list = []
    r1 = tau_row / (tau_row + epsilon)
    r2 = tau_col / (tau_col + epsilon)
    for b in range(B):
        pb, wb, gb = p[b], w[b], g_list[b]
        # Lrep (mass-weighted Gaussian repulsion) — identical to v6/v7 spec
        pb_n = pb / S
        diff_p = pb_n.unsqueeze(1) - pb_n.unsqueeze(0)
        dist2_p = (diff_p ** 2).sum(-1)
        sig_n = max(sigma, 8.0) / S
        Ker_rep = torch.exp(-dist2_p / (2 * sig_n ** 2 + 1e-12)) * (1 - torch.eye(M, device=device))
        rep = lam * (wb.unsqueeze(1) * wb.unsqueeze(0) * Ker_rep).sum() * 0.5
        tot["rep"] += rep.item()
        if gb is None or gb.numel() == 0:
            cnt_list.append(wb.sum().detach())
            continue
        gb = gb.to(pb.device)
        N = gb.shape[0]
        d2 = ((pb.unsqueeze(1) - gb.unsqueeze(0)) ** 2).sum(-1) / (S * S)   # [M,N]
        # fp32 log-domain generalized Sinkhorn scaling
        with torch.autocast(device_type="cuda", enabled=False):
            lk = (-d2 / epsilon).float()                       # log kernel
            la = torch.log(wb.clamp_min(1e-8)).float()         # log a (=w)
            lb = torch.zeros(N, device=device)                 # log b (=1)
            lu = torch.zeros(M, device=device)
            lv = torch.zeros(N, device=device)
            for _ in range(sinkhorn_K):
                lu = r1 * (la - torch.logsumexp(lk + lv[None, :], dim=1))
                lv = r2 * (lb - torch.logsumexp(lk + lu[:, None], dim=0))
            P = torch.exp(lu[:, None] + lk + lv[None, :])      # [M,N]
        rowsum = P.sum(dim=1)                                  # soft-matched to w
        colsum = P.sum(dim=0)                                  # soft-matched to 1
        trans = alpha * (P * d2).sum()
        kl_row = (rowsum * torch.log(rowsum.clamp_min(1e-8) / wb.clamp_min(1e-8))
                  - rowsum + wb).sum()
        ones = torch.ones_like(colsum)
        kl_col = (colsum * torch.log(colsum.clamp_min(1e-8)) - colsum + ones).sum()
        loss_img = trans + tau_row * kl_row + tau_col * kl_col + rep
        tot["uot"] += loss_img.item()
        tot["trans"] += trans.item()
        tot["kl_row"] += tau_row * kl_row.item()
        tot["kl_col"] += tau_col * kl_col.item()
        tot["res_row"] += (rowsum - wb).abs().sum().item()
        tot["res_col"] += (colsum - 1).abs().sum().item()
        cnt = P.sum()
        tot["cnt_err"] += abs(cnt.item() - N)
        cnt_list.append(cnt.detach())
    total = tot["uot"] / B
    metrics = {"lot": tot["uot"] / B, "rep": tot["rep"] / B, "cnt_err": tot["cnt_err"] / B,
               "trans": tot["trans"] / B, "klr": tot["kl_row"] / B, "klc": tot["kl_col"] / B,
               "resr": tot["res_row"] / B, "resc": tot["res_col"] / B,
               "def": 0.0, "over": 0.0, "sur": 0.0, "ent": 0.0}
    return total, metrics, torch.stack(cnt_list) if cnt_list else torch.zeros(B, device=device)


    """
    p: [B,M,2] in S-space
    w: [B,M]
    g_list: list of [N_i,2] tensors in S-space (variable N)
    Returns total loss, dict metrics
    Dustbin single-step entropy assignment: logits = [-d/ε, 0] -> π = w * softmax
    Then Lot + Lrep + overflow penalty β_over·Σmax(0,R−1)² (restores doc's removed overflow guard;
    needed because single-step row-independent softmax does not inherit balanced-OT no-overflow property).
    """
    B, M, _ = p.shape
    device = p.device
    total_lot = 0.0
    total_rep = 0.0
    total_count_err = 0.0
    total_over = 0.0
    total_trans = 0.0
    total_def = 0.0
    total_sur = 0.0
    total_ent = 0.0
    for b in range(B):
        pb = p[b]  # [M,2]
        wb = w[b]  # [M]
        gb = g_list[b]  # [N,2] or empty
        if gb is None or gb.numel() == 0:
            # no GT pits: all mass should be surplus
            s = wb  # all surplus
            lot = gamma * (s**2).sum()
            # repulsion still
            # compute pairwise repulsion
            diff = pb.unsqueeze(1) - pb.unsqueeze(0)  # [M,M,2]
            dist2 = (diff**2).sum(dim=-1)  # [M,M]
            # Gaussian kernel
            K = torch.exp(-dist2 / (2*sigma**2 + 1e-8))
            # zero diag
            K = K * (1 - torch.eye(M, device=device))
            rep = lam * (wb.unsqueeze(1) * wb.unsqueeze(0) * K).sum() * 0.5
            total_lot = total_lot + lot
            total_rep = total_rep + rep
            continue
        gb = gb.to(device)
        N = gb.shape[0]
        # distance matrix [M,N] squared, normalized by S^2
        # pb [M,2], gb [N,2] -> [M,N]
        diff = pb.unsqueeze(1) - gb.unsqueeze(0)  # [M,N,2]
        d2 = (diff**2).sum(dim=-1) / (S*S)  # normalized squared distance
        # logits for dustbin: [M, N+1], last column = 0 (cost 0)
        logits = torch.cat([-d2 / (epsilon+1e-8), torch.zeros(M,1, device=device)], dim=1)  # [M,N+1]
        # softmax over N+1 per pile
        prob = F.softmax(logits, dim=1)  # [M,N+1]
        pi = wb.unsqueeze(1) * prob[:, :N]  # [M,N]
        s = wb * prob[:, N]  # [M]
        R = pi.sum(dim=0)  # [N]
        # Lot terms
        transport = alpha * (pi * d2).sum()
        deficit = torch.clamp(1 - R, min=0)  # [N]
        lot_def = beta * (deficit**2).sum()
        overflow = torch.clamp(R - 1, min=0)  # [N] pit capacity violated
        lot_over = beta_over * (overflow**2).sum()
        lot_sur = gamma * (s**2).sum()
        ent_raw = ( (pi.clamp_min(1e-8) * torch.log(pi.clamp_min(1e-8))).sum() + (s.clamp_min(1e-8) * torch.log(s.clamp_min(1e-8))).sum() )
        ent = epsilon * ent_raw
        lot = transport + lot_def + lot_over + lot_sur + ent * 0.1  # scale entropy small
        total_trans += transport.item()
        total_def += lot_def.item()
        total_over += lot_over.item()
        total_sur += lot_sur.item()
        total_ent += ent.item() * 0.1
        # Lrep: mass-weighted Gaussian
        # pairwise distance between piles in normalized coords
        pb_n = pb / S  # [M,2] in [0,1]
        diff_p = pb_n.unsqueeze(1) - pb_n.unsqueeze(0)  # [M,M,2]
        dist2_p = (diff_p**2).sum(dim=-1)  # [M,M]
        sigma_n = sigma / S
        K = torch.exp(-dist2_p / (2*sigma_n**2 + 1e-8))
        K = K * (1 - torch.eye(M, device=device))
        rep = lam * (wb.unsqueeze(1) * wb.unsqueeze(0) * K).sum() * 0.5
        total_lot = total_lot + lot
        total_rep = total_rep + rep
        # count metric: predicted count = sum(w - s) = sum pi
        pred_c = pi.sum()
        true_c = float(N)
        total_count_err = total_count_err + abs(pred_c.item() - true_c)
        total_over = total_over + lot_over.item()
    # average over batch
    total = (total_lot + total_rep) / B
    return total, {"lot": total_lot.item()/B, "rep": total_rep.item()/B, "cnt_err": total_count_err/B, "over": total_over/B,
                   "trans": total_trans/B, "def": total_def/B, "sur": total_sur/B, "ent": total_ent/B}

class DinoGODHf(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.S = int(cfg.get("input_size", 384))
        self.patch = int(cfg.get("patch_size", 16))
        self.cfg = cfg
        # HF backbone
        model_name = cfg.get("hf_model", "facebook/dinov3-vits16-pretrain-lvd1689m")
        # Use AutoModel with HF mirror/mirror token handling
        import os
        os.environ.setdefault("HF_HOME", "/data/asset/hf")
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        # token from file if exists
        token = None
        for p in ["/tmp/hf_token.txt", "/data/repo/local/hf_access_token.txt", "/root/.cache/huggingface/token"]:
            if os.path.exists(p):
                try:
                    token = open(p).read().strip()
                    break
                except: pass
        if token is None:
            try:
                token = open(os.path.expanduser("~/cac_explore/local/hf_access_token.txt")).read().strip()
            except: pass
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained(model_name, token=token, trust_remote_code=True)
        # freeze all backbone per spec (fully frozen viable)
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        # check hidden size
        hidden = getattr(self.backbone.config, "hidden_size", 384)
        self.hidden = hidden
        # heads
        self.prompt = PromptGateA(dim=hidden)
        self.head = GODHead(dim=hidden, hidden=128)
        # grid centers buffer
        centers, g = _grid_centers(self.S, self.patch)
        self.register_buffer("grid_centers", centers)
        self.g = g
        # GOD hyperparams
        self.alpha = float(cfg.get("god_alpha", 1.0))
        self.beta = float(cfg.get("god_beta", 0.5))
        self.gamma = float(cfg.get("god_gamma", 0.1))
        self.epsilon = float(cfg.get("god_epsilon", 0.05))
        self.lam = float(cfg.get("god_lambda", 1e-3))
        self.sigma_scale = float(cfg.get("god_sigma_scale", 1.0))
        self.beta_over = float(cfg.get("god_beta_overflow", 0.5))
        # v7 solver: standard KL-relaxed UOT via log-domain Sinkhorn scaling
        self.solver = str(cfg.get("god_solver", "sinkhorn"))
        self.sinkhorn_K = int(cfg.get("sinkhorn_K", 10))
        self.tau_row = float(cfg.get("god_tau_row", 1.0))   # row/supply side (↔ γ)
        self.tau_col = float(cfg.get("god_tau_col", 1.0))   # col/demand side (↔ β)
        # for param_groups
        self.is_frozen = True

    def param_groups(self, base_lr, weight_decay):
        # HF AdamW will be used outside; this is for engine compat
        # backbone frozen -> no params
        rest = [p for n,p in self.named_parameters() if p.requires_grad]
        return [{"params": rest, "lr": base_lr, "weight_decay": weight_decay}]

    def forward(self, imgs, bboxes=None, bboxes3=None, points=None):
        """
        imgs: [B,3,S,S] already processor-normalized (HF processor output pixel_values)
              OR raw [0,1] if processor not used externally. We handle both.
        bboxes/bboxes3: [B,4] / [B,3,4] in S-space
        points: list of [N_i,2] in S-space (for loss) OR None (inference)
        Returns dict with p,w, loss if points given
        """
        B = imgs.shape[0]
        # backbone forward: AutoModel expects pixel_values
        # DINOv3ViTModel forward signature: pixel_values
        out = self.backbone(pixel_values=imgs)
        # last_hidden_state: [B, L, C] L = 1 + 4 + HW
        h = out.last_hidden_state  # [B, L, C]
        # slice patch tokens: skip 1 cls + 4 registers
        num_reg = getattr(self.backbone.config, "num_register_tokens", 4)
        patch_tokens = h[:, 1+num_reg:, :]  # [B, HW, C]
        # ensure HW == g*g (576 for 384)
        # It should be, but if S mismatched, handle via reshape
        # prompt gate
        if bboxes3 is None and bboxes is not None:
            # expand single box to 3
            bboxes3 = bboxes.unsqueeze(1).repeat(1,3,1)
        if bboxes3 is None:
            # no prompt: gate = 1
            gate = torch.ones(B, patch_tokens.shape[1], 1, device=imgs.device)
        else:
            gate, _ = self.prompt(patch_tokens, bboxes3, self.S, self.patch)
        w, p = self.head(patch_tokens, gate, self.grid_centers)
        # w [B,M], p [B,M,2]
        if points is not None:
            # points: list of [N_i,2] or tensor padded? Handle list
            if isinstance(points, torch.Tensor) and points.dim()==3:
                # padded: [B, Nmax,2] with -1 for invalid? Not used
                g_list = [points[i] for i in range(B)]
            elif isinstance(points, list):
                g_list = points
            else:
                g_list = None
            # sigma from median box size
            sigma = 20.0
            if bboxes3 is not None:
                # median box size in pixels
                wh = bboxes3[:,:,2:4] - bboxes3[:,:,0:2]  # [B,3,2]
                wh = wh.clamp_min(1)
                msize = wh.mean().item() if wh.numel() else 20.0
                sigma = msize * self.sigma_scale
                sigma = max(sigma, 8.0)
            if self.solver == "sinkhorn":
                loss, metrics, cnt_open = god_loss_uot(p, w, g_list, alpha=self.alpha,
                    tau_row=self.tau_row, tau_col=self.tau_col, epsilon=self.epsilon,
                    lam=self.lam, sigma=sigma, S=self.S, sinkhorn_K=self.sinkhorn_K)
            else:
                loss, metrics = god_loss_router(p, w, g_list, alpha=self.alpha, beta=self.beta,
                    gamma=self.gamma, epsilon=self.epsilon, lam=self.lam, sigma=sigma, S=self.S,
                    beta_over=self.beta_over)
                cnt_open = None
            # count prediction via transported mass (for MAE)
            # need to recompute pi for metrics: reuse dustbin prob logic quickly
            # For inference we also need counts
            with torch.no_grad():
                if cnt_open is None:
                    # router fallback: recompute transported count
                    pred_counts = []
                    for b in range(B):
                        pb = p[b]; wb = w[b]; gb = g_list[b] if g_list is not None else None
                        if gb is None or gb.numel()==0:
                            pred_c = 0.0
                        else:
                            gb = gb.to(pb.device)
                            N = gb.shape[0]
                            d2 = ((pb.unsqueeze(1)-gb.unsqueeze(0))**2).sum(-1)/(self.S*self.S)
                            logits = torch.cat([-d2/self.epsilon, torch.zeros(pb.shape[0],1, device=pb.device)], dim=1)
                            prob = F.softmax(logits, dim=1)
                            pi = wb.unsqueeze(1)*prob[:,:N]
                            pred_c = pi.sum().item()
                        pred_counts.append(pred_c)
                    pred_counts = torch.tensor(pred_counts, device=imgs.device)
                else:
                    pred_counts = cnt_open
            return {"p": p, "w": w, "loss": loss, "pred_counts": pred_counts,
                    "counts_sumw": w.sum(dim=1).detach(), "metrics": metrics, "gate": gate}
        else:
            # inference without GT: count = sum w * gate? Actually need dustbin prob without GT: we can't compute pi without g. Fallback sum w
            # For eval, we need points to compute pi; if no points, just sum w*gate as count
            # But for FSC147 eval, points not available; we should return sum(w) as count proxy or use w*gate sum
            # The true GOD count is Σ(w - s) which needs GT; without GT we approximate Σw
            pred_counts = w.sum(dim=1)
            return {"p": p, "w": w, "pred_counts": pred_counts, "gate": gate}

def build_model(cfg):
    return DinoGODHf(cfg)

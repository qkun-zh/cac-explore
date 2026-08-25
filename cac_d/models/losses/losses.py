import torch, torch.nn.functional as F

def uot_loss(p, w, points, S=384, eps=0.08, tau=1.0, alpha=1.0, iters=32):
    """Minimal unbalanced OT: entropic assignment of cell masses to GT points
    (with a no-match column) + demand-KL on column sums. `iters` reserved for
    full log-domain Sinkhorn; unused in this minimal form."""
    B = p.shape[0]
    tot = p.new_zeros(())
    cnts = []
    for b in range(B):
        pb, wb, gb = p[b], w[b], points[b]
        if gb.numel() == 0: continue
        gb = gb.to(pb.device); N = gb.shape[0]
        d2 = ((pb.unsqueeze(1) - gb.unsqueeze(0)) ** 2).sum(-1) / (S*S)     # [M,N]
        prob = F.softmax(torch.cat([-d2 / eps, torch.zeros(len(pb), 1, device=pb.device)], 1), 1)
        pi = wb.unsqueeze(1) * prob[:, :N]
        col = pi.sum(0)
        tot = tot + alpha*(pi*d2).sum() + tau*(col*col.clamp_min(1e-8).log() - col + 1).sum()
        cnts.append(pi.sum())
    return (tot/B if B else tot), (torch.stack(cnts) if cnts else torch.zeros(B, device=p.device))

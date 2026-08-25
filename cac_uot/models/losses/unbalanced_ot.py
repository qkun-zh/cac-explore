import torch, torch.nn.functional as F

def unbalanced_ot_loss(p, w, g_list, transport_weight=1.0, supply_tau=1.0, demand_tau=1.0,
                       entropy_reg=0.05, S=384, sinkhorn_iters=10):
    """Standard KL-relaxed unbalanced OT (Chizat). Returns loss tensor + metrics + transported counts.
       p [B,M,2], w [B,M], g_list list[N,2]. Loss = <pi,C> + τ_supply KL_rows + τ_demand KL_cols.
       pi from K-step log-domain generalized Sinkhorn; gradients through K steps.
    """
    B, M, _ = p.shape
    totals = {"uot": 0.0, "trans": 0.0, "klr": 0.0, "klc": 0.0, "resr": 0.0, "resc": 0.0, "cnt_err": 0.0}
    cnts, losses = [], []
    r1, r2 = supply_tau/(supply_tau+entropy_reg), demand_tau/(demand_tau+entropy_reg)
    for b in range(B):
        pb, wb, gb = p[b], w[b], g_list[b]
        if gb is None or gb.numel() == 0:
            losses.append(torch.zeros((), device=p.device))
            cnts.append(wb.sum().detach()); continue
        gb = gb.to(pb.device); N = gb.shape[0]
        d2 = ((pb.unsqueeze(1)-gb.unsqueeze(0))**2).sum(-1)/(S*S)
        with torch.autocast(device_type="cuda", enabled=False):
            lk = (-d2/entropy_reg).float(); la = torch.log(wb.clamp_min(1e-8)).float()
            lb = torch.zeros(N, device=pb.device)
            lu = torch.zeros(M, device=pb.device); lv = torch.zeros(N, device=pb.device)
            for _ in range(sinkhorn_iters):
                lu = r1 * (la - torch.logsumexp(lk + lv[None,:], dim=1))
                lv = r2 * (lb - torch.logsumexp(lk + lu[:,None], dim=0))
            P = torch.exp(lu[:,None] + lk + lv[None,:])
        rowsum, colsum = P.sum(1), P.sum(0)
        trans = transport_weight * (P*d2).sum()
        klr = (rowsum*torch.log(rowsum.clamp_min(1e-8)/wb.clamp_min(1e-8)) - rowsum + wb).sum()
        klc = (colsum*torch.log(colsum.clamp_min(1e-8)) - colsum + 1).sum()
        loss_b = trans + supply_tau*klr + demand_tau*klc
        losses.append(loss_b)
        for k,v in [("trans",trans.item()),("klr",supply_tau*klr.item()),("klc",demand_tau*klc.item()),
                    ("resr",(rowsum-wb).abs().sum().item()),("resc",(colsum-1).abs().sum().item()),
                    ("cnt_err",abs(P.sum().item()-N))]:
            totals[k] += v
        cnts.append(P.sum().detach())
    total = sum(losses)/B
    metrics = {k: v/B for k,v in totals.items()}
    metrics["lot"] = total.item()
    return total, metrics, torch.stack(cnts) if cnts else torch.zeros(B, device=p.device)

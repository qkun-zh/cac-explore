import torch
def repulsion(p, w, lam=1e-3, sigma=20.0, S=384):
    pb_n = p / S
    d2 = ((pb_n.unsqueeze(1)-pb_n.unsqueeze(0))**2).sum(-1)
    sig = max(sigma,8.0)/S
    K = torch.exp(-d2/(2*sig**2+1e-12)) * (1-torch.eye(p.shape[1], device=p.device))
    return lam * (w.unsqueeze(1)*w.unsqueeze(0)*K).sum()*0.5

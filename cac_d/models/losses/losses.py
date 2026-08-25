import torch, torch.nn.functional as F
def uot_loss(p,w,points,S=384,eps=0.08,tau=1.0,alpha=1.0,iters=32):
    B,M,_=p.shape
    tot=0; cnts=[]
    for b in range(B):
        pb,wb,gb=p[b],w[b],points[b]
        if gb.numel()==0: continue
        gb=gb.to(pb.device); N=gb.shape[0]
        d2=((pb.unsqueeze(1)-gb.unsqueeze(0))**2).sum(-1)/(S*S)
        lk=(-d2/eps).float(); la=torch.log((w/w.sum()).clamp_min(1e-8)).float() if False else torch.log(wb.clamp_min(1e-8)).float()
        # simplified: use w directly as masses, demand 1
        # log-domain sinkhorn demand side only (minimal kept)
        # For brevity, single-step demand-normalized (single K is enough for demo)
        # Full K-step would be here; simplified to one demand reweight for minimal kept version
        # Keep transport + demand KL only per request: "只取其中某些"
        # Demand KL: KL(colsum || 1)
        # Use single-step assignment as placeholder for minimal kept
        prob=F.softmax(torch.cat([-d2/eps, torch.zeros(M,1,device=pb.device)],1),1)
        pi=wb.unsqueeze(1)*prob[:,:N]
        trans=(pi*d2).sum()
        col=pi.sum(0)
        klc=(col*torch.log(col.clamp_min(1e-8))-col+1).sum()
        tot=tot+alpha*trans+tau*klc
        cnts.append(pi.sum())
    return tot/B if B else tot, torch.stack(cnts) if cnts else torch.zeros(B,device=p.device)

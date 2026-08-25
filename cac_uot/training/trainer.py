import torch, math, time, json

BUCKETS = [(0,25),(25,75),(75,200),(200,500),(500,float("inf"))]

def _spearman(a,b):
    ra = a.argsort().float(); rb = b.argsort().float()
    # rank transform via argsort twice
    ra = torch.empty_like(a).float().scatter_(0, ra.argsort(), torch.arange(1, len(a)+1, dtype=torch.float32))
    rb = torch.empty_like(b).float().scatter_(0, rb.argsort(), torch.arange(1, len(b)+1, dtype=torch.float32))
    ra -= ra.mean(); rb -= rb.mean()
    return (ra*rb).sum() / (ra.norm()*rb.norm()).clamp_min(1e-12)

def train_one_epoch(model, loader, optimizer, device, cfg, ep, log_every=50):
    model.train(); total=0; nb=0; t0=time.time()
    for batch in loader:
        pv = batch["pixel_values"].to(device)
        b3 = batch["bboxes3"].to(device)
        pts = [p.to(device) for p in batch["points"]]
        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=cfg.amp):
            out = model(pv, b3, pts)
            loss = out["loss"]
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item(); nb+=1
        if nb % log_every == 0:
            w, g, met = out["w"].detach().float(), out["gate"].detach().float(), out["metrics"]
            ws = w.sum(1)
            print(f"ep{ep} it{nb}/{len(loader)} loss={loss.item():.3f} "
                  f"[trans={met.get('trans',0):.2f} klr={met.get('klr',0):.2f} klc={met.get('klc',0):.2f} "
                  f"resr={met.get('resr',0):.1f} resc={met.get('resc',0):.1f} anchor={met.get('anchor',0):.3f} rep={met.get('rep',0):.4f}] "
                  f"w_sum={ws.mean().item():.1f} gate={g.mean().item():.3f} a={model.gate.alpha.item():+.2f} gnorm={gn.item():.1f}", flush=True)
    return total/max(nb,1)

@torch.no_grad()
def evaluate(model, loader, device, ep=0):
    model.eval()
    gts, preds, preds_open, gates, wsums = [], [], [], [], []
    for batch in loader:
        pv = batch["pixel_values"].to(device)
        b3 = batch["bboxes3"].to(device)
        pts = [p.to(device) for p in batch["points"]]
        out = model(pv, b3, pts)
        gts += [p.shape[0] for p in pts]
        preds += out["counts_sumw"].float().cpu().tolist()
        preds_open += out["pred_counts"].float().cpu().tolist()
        gates.append(out["gate"].float().mean().item())
        wsums += out["w"].float().sum(1).cpu().tolist()
    gt=torch.tensor(gts).float(); pred=torch.tensor(preds).float(); po=torch.tensor(preds_open).float()
    err=pred-gt; erro=po-gt
    def bucket_print(err, name):
        print(f" {name}:", flush=True)
        for lo,hi in BUCKETS:
            m=(gt>=lo)&(gt<hi)
            if m.sum()==0: continue
            bm=err[m]
            hi_s="inf" if hi==float("inf") else str(int(hi))
            print(f"  [{int(lo):>3},{hi_s:>4}) n={int(m.sum()):4d} MAE={bm.abs().mean():6.1f} med={bm.median():+6.1f}", flush=True)
    print(f"== Ep{ep:02d} VAL closed MAE={err.abs().mean():.2f} RMSE={err.pow(2).mean().sqrt():.2f} rho={_spearman(gt,pred):.3f} | "
          f"open MAE={erro.abs().mean():.2f} rho={_spearman(gt,po):.3f} gate={sum(gates)/len(gates):.3f}", flush=True)
    bucket_print(err, "closed Σw"); bucket_print(erro, "open OT")
    return err.abs().mean().item(), erro.abs().mean().item()

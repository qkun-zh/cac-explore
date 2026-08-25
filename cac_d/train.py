import os, sys, time, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cac_d.common import setup_hf_env, hf_token
setup_hf_env()
import torch
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel
from cac_d.configs.config import Config
from cac_d.models.model import Counter

def lr_lambda(e):
    c = Config()
    if e < c.warmup_epochs:
        return (e + 1) / c.warmup_epochs
    if e < c.warmup_epochs + c.stable_epochs:
        return 1.0
    t = (e - c.warmup_epochs - c.stable_epochs) / max(1, c.epochs - c.warmup_epochs - c.stable_epochs)
    return c.eta_min_ratio + (1 - c.eta_min_ratio) * 0.5 * (1 + math.cos(math.pi * t))

@torch.no_grad()
def evaluate(m, va, device, cached=False):
    err = n = sq = 0.0
    for batch in va:
        if cached:
            h2 = batch["h2"].to(device, non_blocking=True)
            h3 = batch["h3"].to(device, non_blocking=True)
            e_ = batch["e"].to(device, non_blocking=True)
            bb = batch["bboxes"].to(device, non_blocking=True)
            pred = m(None, bb, h2=h2, h3=h3, e=e_)["pred_counts"].cpu()
        else:
            pv = batch["pixel_values"].to(device, non_blocking=True)
            bb = batch["bboxes"].to(device, non_blocking=True)
            pred = m(pv, bb)["pred_counts"].cpu()
        gt = torch.tensor([len(p) for p in batch["points"]], dtype=torch.float32)
        err += (pred-gt).abs().sum().item()
        sq += ((pred-gt)**2).sum().item()
        n += len(gt)
    return err/n, (sq/n)**0.5

def main():
    cfg = Config()
    import json
    ov = os.environ.get("CAC_D_OVERRIDE")
    if ov:
        for k, v in json.loads(ov).items():
            assert hasattr(cfg, k), f"unknown override {k}"
            setattr(cfg, k, v)
    cached = cfg.use_cached_features
    snap = os.path.join(os.path.dirname(cfg.best_ckpt), f"cfg.json")
    with open(snap, "w") as f:
        json.dump({**cfg.__dict__, "override": ov}, f, indent=1)
    torch.manual_seed(cfg.seed)

    if cached:
        from cac_d.datasets.cached_dataset import CachedDataset, cached_collate
        tr_cache = os.path.join(cfg.cache_dir, "train")
        va_cache = os.path.join(cfg.cache_dir, "val")
        print(f"CACHED mode: train={tr_cache} val={va_cache}")
        tr = DataLoader(CachedDataset(tr_cache), batch_size=cfg.batch_size,
                        shuffle=True, num_workers=cfg.num_workers, pin_memory=True,
                        persistent_workers=cfg.num_workers > 0, collate_fn=cached_collate)
        va = DataLoader(CachedDataset(va_cache), batch_size=cfg.batch_size,
                        shuffle=False, num_workers=cfg.num_workers, pin_memory=True,
                        persistent_workers=cfg.num_workers > 0, collate_fn=cached_collate)
    else:
        from transformers import AutoImageProcessor
        from cac_d.datasets.dataset import FSC147, collate
        proc = AutoImageProcessor.from_pretrained(cfg.hf_model, token=hf_token(),
                                                  trust_remote_code=True,
                                                  size={"height": cfg.image_size, "width": cfg.image_size})
        tr = DataLoader(FSC147("train", cfg.image_size, augment=True,
                               flip_p=cfg.flip_p, color_jitter=cfg.color_jitter),
                        batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
                        pin_memory=True, persistent_workers=cfg.num_workers > 0,
                        collate_fn=lambda b: collate(b, proc))
        va = DataLoader(FSC147("val", cfg.image_size), batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_workers, pin_memory=True,
                        persistent_workers=cfg.num_workers > 0, collate_fn=lambda b: collate(b, proc))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg.amp and device.type == "cuda"
    model = Counter(cfg, cached=cached).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable params: {sum(p.numel() for p in params)/1e6:.2f}M")
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler(enabled=use_amp)
    ema = AveragedModel(model,
                        avg_fn=lambda avg, new, n: cfg.ema_decay*avg + (1-cfg.ema_decay)*new
                        ).to(device)
    best = float("inf")
    for ep in range(1, cfg.epochs+1):
        t0 = time.time(); model.train(); tot = 0.0
        s_den = s_cnt = 0.0
        for batch in tr:
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", torch.float16, enabled=use_amp):
                if cached:
                    h2 = batch["h2"].to(device, non_blocking=True)
                    h3 = batch["h3"].to(device, non_blocking=True)
                    e_ = batch["e"].to(device, non_blocking=True)
                    bb = batch["bboxes"].to(device, non_blocking=True)
                    pts = [p.to(device, non_blocking=True) for p in batch["points"]]
                    out = model(None, bb, pts, h2=h2, h3=h3, e=e_)
                else:
                    pv = batch["pixel_values"].to(device, non_blocking=True)
                    bb = batch["bboxes"].to(device, non_blocking=True)
                    pts = [p.to(device, non_blocking=True) for p in batch["points"]]
                    out = model(pv, bb, pts)
                loss = out["loss"]
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt); scaler.update()
            ema.update_parameters(model); tot += loss.item()
            s_den += out["loss_den"].item()
            s_cnt += out["loss_cnt"].item()
            if ep == 1 and tot == loss.item():
                print(f"  [debug] loss={loss.item():.4f} isnan={torch.isnan(loss).item()} "
                      f"min={loss.detach().min().item():.4f} dtype={loss.dtype}", flush=True)
        sched.step()
        mae_raw, rmse_raw = evaluate(model, va, device, cached=cached)
        ema.eval(); mae_ema, rmse_ema = evaluate(ema, va, device, cached=cached)
        nbatch = len(tr)
        print(f"Ep{ep} loss={tot/nbatch:.3f} den={s_den/nbatch:.4f} cnt={s_cnt/nbatch:.4f} "
              f"[{time.time()-t0:.0f}s] MAE={mae_raw:.2f}/{mae_ema:.2f} RMSE={rmse_raw:.1f}/{rmse_ema:.1f} best={best:.2f}", flush=True)
        if min(mae_raw, mae_ema) < best:
            best = min(mae_raw, mae_ema)
            src = ema if mae_ema <= mae_raw else model
            torch.save(src.state_dict(), cfg.best_ckpt); print(" best", flush=True)

if __name__ == "__main__": main()

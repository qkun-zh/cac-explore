import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cac_d.common import setup_hf_env, hf_token
setup_hf_env()
import torch
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor            # HF processor
from cac_d.configs.config import Config
from cac_d.models.model import Counter
from cac_d.datasets.dataset import FSC147, collate

def main():
    cfg = Config()
    torch.manual_seed(cfg.seed)
    proc = AutoImageProcessor.from_pretrained(cfg.hf_model, token=hf_token(),
                                              trust_remote_code=True,
                                              size={"height": cfg.image_size, "width": cfg.image_size})
    tr = DataLoader(FSC147("train", cfg.image_size), batch_size=cfg.batch_size, shuffle=True,
                    num_workers=cfg.num_workers, pin_memory=True,
                    persistent_workers=cfg.num_workers > 0, collate_fn=lambda b: collate(b, proc))
    va = DataLoader(FSC147("val", cfg.image_size), batch_size=cfg.batch_size, shuffle=False,
                    num_workers=cfg.num_workers, pin_memory=True,
                    persistent_workers=cfg.num_workers > 0, collate_fn=lambda b: collate(b, proc))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg.amp and device.type == "cuda"
    model = Counter(cfg).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)   # torch optimizer
    scaler = torch.amp.GradScaler(enabled=use_amp)
    best = float("inf")
    for ep in range(1, cfg.epochs+1):
        t0 = time.time(); model.train(); tot = 0.0
        for batch in tr:
            pv = batch["pixel_values"].to(device, non_blocking=True)
            bb = batch["bboxes"].to(device, non_blocking=True)
            pts = [p.to(device, non_blocking=True) for p in batch["points"]]
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", torch.float16, enabled=use_amp):
                out = model(pv, bb, pts); loss = out["loss"]
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt); scaler.update(); tot += loss.item()
        model.eval(); err = n = 0.0
        with torch.no_grad():
            for batch in va:
                pv = batch["pixel_values"].to(device, non_blocking=True)
                bb = batch["bboxes"].to(device, non_blocking=True)
                pred = model(pv, bb)["pred_counts"].cpu()
                gt = torch.tensor([len(p) for p in batch["points"]], dtype=torch.float32)
                err += (pred-gt).abs().sum().item(); n += len(gt)
        mae = err/n; dt = time.time()-t0
        print(f"Ep{ep} loss={tot/len(tr):.3f} [{dt:.0f}s] val MAE={mae:.2f} best={best:.2f}", flush=True)
        if mae < best:
            best = mae; torch.save(model.state_dict(), cfg.best_ckpt); print(" best", flush=True)

if __name__ == "__main__": main()

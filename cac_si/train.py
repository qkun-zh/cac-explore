import os, sys, time, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cac_d.common import setup_hf_env, hf_token
setup_hf_env()
import torch
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel
from cac_si.configs.config import Config
from cac_si.models.model import SICounter


def make_lr_lambda(cfg):
    def lr_lambda(e):
        if e < cfg.warmup_epochs:
            return (e + 1) / cfg.warmup_epochs
        if e < cfg.warmup_epochs + cfg.stable_epochs:
            return 1.0
        t = (e - cfg.warmup_epochs - cfg.stable_epochs) / max(1, cfg.epochs - cfg.warmup_epochs - cfg.stable_epochs)
        return cfg.eta_min_ratio + (1 - cfg.eta_min_ratio) * 0.5 * (1 + math.cos(math.pi * t))
    return lr_lambda


@torch.no_grad()
def evaluate(m, va, device):
    err = n = sq = 0.0
    for batch in va:
        pv = batch["pixel_values"].to(device, non_blocking=True)
        bb = batch["bboxes"].to(device, non_blocking=True)
        pred = m(pv, bb)["pred_counts"].cpu()
        gt = torch.tensor([len(p) for p in batch["points"]], dtype=torch.float32)
        err += (pred - gt).abs().sum().item()
        sq += ((pred - gt) ** 2).sum().item()
        n += len(gt)
    return err / n, (sq / n) ** 0.5


def main():
    cfg = Config()
    import json
    ov = os.environ.get("CAC_SI_OVERRIDE")
    if ov:
        for k, v in json.loads(ov).items():
            assert hasattr(cfg, k), f"unknown override {k}"
            setattr(cfg, k, v)
    snap = os.path.join(os.path.dirname(cfg.best_ckpt), "cfg.json")
    with open(snap, "w") as f:
        json.dump({**cfg.__dict__, "override": ov}, f, indent=1, default=str)
    metrics_path = os.path.join(os.path.dirname(cfg.best_ckpt), "metrics.jsonl")
    torch.manual_seed(cfg.seed)

    from transformers import AutoImageProcessor
    from cac_d.datasets.dataset import FSC147, collate
    proc = AutoImageProcessor.from_pretrained(cfg.hf_model, token=hf_token(),
                                              trust_remote_code=True,
                                              size={"height": cfg.image_size,
                                                    "width": cfg.image_size})
    coll = lambda b: collate(b, proc)
    tr = DataLoader(FSC147("train", cfg.image_size, augment=True,
                           flip_p=cfg.flip_p, color_jitter=cfg.color_jitter),
                    batch_size=cfg.batch_size, shuffle=True,
                    num_workers=cfg.num_workers, pin_memory=True,
                    persistent_workers=cfg.num_workers > 0, collate_fn=coll)
    va = DataLoader(FSC147("val", cfg.image_size), batch_size=cfg.batch_size,
                    shuffle=False, num_workers=cfg.num_workers, pin_memory=True,
                    persistent_workers=cfg.num_workers > 0, collate_fn=coll)
    te = DataLoader(FSC147("test", cfg.image_size), batch_size=cfg.batch_size,
                    shuffle=False, num_workers=cfg.num_workers, pin_memory=True,
                    persistent_workers=cfg.num_workers > 0, collate_fn=coll)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg.amp and device.type == "cuda"
    model = SICounter(cfg).to(device)
    total = sum(p.numel() for p in model.parameters()) / 1e6
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"Params: total {total:.2f}M (budget 32M) | trainable {sum(p.numel() for p in params)/1e6:.2f}M")
    assert total <= 32.0, "mission budget exceeded"
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, make_lr_lambda(cfg))
    scaler = torch.amp.GradScaler(enabled=use_amp)
    ema = AveragedModel(model,
                        avg_fn=lambda avg, new, n: cfg.ema_decay * avg + (1 - cfg.ema_decay) * new
                        ).to(device)

    best = float("inf")
    for ep in range(1, cfg.epochs + 1):
        t0 = time.time(); model.train(); tot = 0.0
        s_den = s_cnt = 0.0; gn_sum = 0.0
        for batch in tr:
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", torch.float16, enabled=use_amp):
                pv = batch["pixel_values"].to(device, non_blocking=True)
                bb = batch["bboxes"].to(device, non_blocking=True)
                pts = [p.to(device, non_blocking=True) for p in batch["points"]]
                out = model(pv, bb, pts)
                loss = out["loss"]
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gn_sum += float(torch.nn.utils.clip_grad_norm_(params, 1.0))
            scaler.step(opt); scaler.update()
            ema.update_parameters(model)
            tot += loss.item()
            s_den += out["loss_den"].item()
            s_cnt += out["loss_cnt"].item()
        t_train = time.time() - t0
        sched.step(); lr_now = sched.get_last_lr()[0]
        t_ev = time.time()
        mae_raw, rmse_raw = evaluate(model, va, device)
        ema.eval(); mae_ema, rmse_ema = evaluate(ema, va, device)
        t_eval = time.time() - t_ev
        nbatch = len(tr)
        d_avg, c_avg = s_den / nbatch, s_cnt / nbatch
        tot_avg = (s_den + s_cnt) / nbatch
        pct_den = 100 * d_avg / tot_avg if tot_avg > 0 else 0.0
        rec = {"ep": ep, "t_total": round(t_train + t_eval, 1),
               "t_train": round(t_train, 1), "t_val": round(t_eval, 1),
               "lr": lr_now, "grad_norm": round(gn_sum / nbatch, 3),
               "loss": round(tot_avg, 4), "loss_den": round(d_avg, 5),
               "loss_cnt": round(c_avg, 5), "pct_den": round(pct_den, 1),
               "mae_raw": round(mae_raw, 2), "mae_ema": round(mae_ema, 2),
               "rmse_raw": round(rmse_raw, 1), "rmse_ema": round(rmse_ema, 1)}
        print(f"Ep{ep} [{rec['t_total']}s tr {rec['t_train']} ev {rec['t_val']}] lr={lr_now:.2e} "
              f"g={rec['grad_norm']:.2e} loss={tot_avg:.3e} (den {pct_den:.0f}% | cnt {100-pct_den:.0f}%) "
              f"MAE={mae_raw:.2f}/{mae_ema:.2f} RMSE={rmse_raw:.1f}/{rmse_ema:.1f} best={best:.2f}", flush=True)
        is_best = False
        if min(mae_raw, mae_ema) < best:
            best = min(mae_raw, mae_ema); is_best = True
            src = ema if mae_ema <= mae_raw else model
            torch.save(src.state_dict(), cfg.best_ckpt); print(" best", flush=True)
        rec["best"] = round(best, 2); rec["is_best"] = is_best
        if ep % cfg.test_every == 0:
            t_te = time.time()
            ema.eval(); mae_te, rmse_te = evaluate(ema, te, device)
            mae_te_raw, rmse_te_raw = evaluate(model, te, device)
            rec.update(test_mae_raw=round(mae_te_raw, 2), test_mae_ema=round(mae_te, 2),
                       test_rmse_raw=round(rmse_te_raw, 1), test_rmse_ema=round(rmse_te, 1),
                       t_test=round(time.time() - t_te, 1))
            print(f"  TEST Ep{ep}: MAE={mae_te_raw:.2f}/{mae_te:.2f} "
                  f"RMSE={rmse_te_raw:.1f}/{rmse_te:.1f}", flush=True)
        with open(metrics_path, "a") as f:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()

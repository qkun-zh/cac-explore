"""code/engine/train.py — shared training engine for all nodes.

Contract: a node directory must contain model.py (build_model(cfg)) and config.py (cfg = dict(...)).
Usage:
  python code/engine/train.py --node_dir tree/nodes/N0001_x [--run_dir /data/runs/N0001_x]
  python code/engine/train.py --node_dir ... --smoke          # synthetic-data smoke, no dataset needed
"""
import argparse, importlib.util, json, math, os, sys, time, traceback

import torch
import torch.nn.functional as F


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def make_loaders(cfg, smoke):
    if smoke:
        from torch.utils.data import DataLoader, Dataset

        class Synth(Dataset):
            """Multi-blob density maps on a low-res grid (g=size/8), matching typical density-head outputs [B,1,g,g]."""
            def __init__(self, n, size):
                self.n, self.size = n, size
            def __len__(self):
                return self.n
            def __getitem__(self, i):
                g = torch.Generator().manual_seed(i * 7919)
                img = torch.rand(3, self.size, self.size, generator=g)
                x0, y0 = int(torch.randint(0, self.size // 2, (1,), generator=g)), int(torch.randint(0, self.size // 2, (1,), generator=g))
                x1, y1 = x0 + self.size // 3, y0 + self.size // 3
                gs = self.size // 8
                yy, xx = torch.meshgrid(torch.arange(gs), torch.arange(gs), indexing="ij")
                dens = torch.zeros(gs, gs)
                for _ in range(1 + int(torch.randint(0, 5, (1,), generator=g))):  # 1-5 objects
                    cy, cx = int(torch.randint(0, gs, (1,), generator=g)), int(torch.randint(0, gs, (1,), generator=g))
                    s = 1.0 + 2.0 * float(torch.rand(1, generator=g))
                    blob = torch.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * s * s))
                    dens = dens + blob / blob.sum().clamp_min(1e-6)   # each blob normalized, count ≈ number of objects
                return {"imgs": img, "bboxes": torch.tensor([x0, y0, x1, y1], dtype=torch.float32),
                        "density": dens[None], "counts": dens.sum()}
        s = cfg["input_size"]
        bs = max(2, min(int(cfg.get("batch_size", 8)), 4))
        tr = Synth(16, s); va = Synth(8, s)
        return DataLoader(tr, batch_size=bs, shuffle=True), DataLoader(va, batch_size=bs), bs

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.fsc147 import FSC147Density, collate_density
    from torch.utils.data import DataLoader
    root = cfg.get("data_root", "/data/dataset/FSC147")
    size = int(cfg.get("input_size", 384))
    tr = FSC147Density(root, size, "train")
    va = FSC147Density(root, size, "val")
    bs = int(cfg.get("batch_size", 8))
    nw = int(cfg.get("num_workers", 4))
    return (DataLoader(tr, batch_size=bs, shuffle=True, num_workers=nw, collate_fn=collate_density, drop_last=True, pin_memory=True),
            DataLoader(va, batch_size=bs, shuffle=False, num_workers=nw, collate_fn=collate_density, pin_memory=True), bs)


def evaluate(model, loader, device, seq=False, max_frac=1.0):
    model.eval(); mae = mse = n = 0
    if max_frac < 1.0:
        n_keep = max(1, int(len(loader) * max_frac))
        loader = [b for i, b in zip(range(n_keep), loader)]
    with torch.no_grad():
        for b in loader:
            imgs, bbox, gt = b["imgs"].to(device), b["bboxes"].to(device), b["counts"]
            out = model(imgs, bbox)
            if seq:
                pred = out["logits"].argmax(-1).sum(1).float().cpu()
            else:
                dens = out["density"] if isinstance(out, dict) else out
                pred = dens.flatten(1).sum(1).float().cpu()
            mae += (pred - gt).abs().sum().item()
            mse += ((pred - gt) ** 2).sum().item()
            n += gt.numel()
    model.train()
    return mae / max(n, 1), math.sqrt(mse / max(n, 1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--node_dir", required=True)
    p.add_argument("--run_dir", default=None)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--timeout-min", type=float, default=float(os.environ.get("TAU_MAX_MIN", 30)))
    p.add_argument("--epochs", type=int, default=None)
    args = p.parse_args()

    node_dir = os.path.abspath(args.node_dir)
    run_name = os.path.basename(node_dir)
    run_dir = args.run_dir or os.environ.get("RUN_DIR") or os.path.join("/data/runs" if os.path.isdir("/data") else ".runs", run_name)
    os.makedirs(run_dir, exist_ok=True)
    result_path = os.path.join(run_dir, "result.json")

    def write_result(status, metrics=None, timing=None, diag=None):
        r = {"node": run_name, "status": status,
             "metrics": metrics or {}, "timing": timing or {}, "diagnostics": diag or {},
             "run_dir": run_dir, "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        json.dump(r, open(result_path, "w"), indent=2, ensure_ascii=False)
        print("RESULT " + json.dumps(r, ensure_ascii=False), flush=True)

    t_start = time.time()
    cfg_mod = load_module(os.path.join(node_dir, "config.py"), f"{run_name}_config")
    model_mod = load_module(os.path.join(node_dir, "model.py"), f"{run_name}_model")
    cfg = dict(getattr(cfg_mod, "cfg"))
    cfg.setdefault("smoke", False)
    if args.smoke:
        cfg["smoke"] = True
    epochs = args.epochs or int(cfg.get("epochs", 10))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    use_amp = bool(cfg.get("amp", True)) and device.type == "cuda"
    try:
        model = model_mod.build_model(cfg).to(device)
    except Exception:
        write_result("failed", diag={"traceback": traceback.format_exc()[-2000:]})
        raise SystemExit(1)
    total_p = sum(q.numel() for q in model.parameters()) / 1e6
    print(f"[engine] node={run_name} device={device} params={total_p:.2f}M smoke={cfg['smoke']}", flush=True)
    assert total_p < float(cfg.get("max_params_M", 32)), f"params {total_p:.2f}M over budget"  # mission budget: 32M

    train_loader, val_loader, _ = make_loaders(cfg, cfg["smoke"])
    optim = torch.optim.AdamW(filter(lambda q: q.requires_grad, model.parameters()),
                              lr=float(cfg.get("lr", 1e-3)), weight_decay=float(cfg.get("weight_decay", 1e-4)))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, epochs, eta_min=float(cfg.get("eta_min", 1e-6)))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    w_cnt = float(cfg.get("loss_count_weight", 0.3))
    loss_fn_name = cfg.get("loss_function", "mse")
    huber_delta = float(cfg.get("huber_delta", 5.0))
    seq_mode = str(cfg.get("paradigm", "reg")) == "seq"

    best = float("inf"); timed_out = False; oom = False
    ckpt = os.path.join(run_dir, "best.pth")
    for ep in range(1, epochs + 1):
        if time.time() - t_start > args.timeout_min * 60:
            timed_out = True; break
        model.train(); ls = nb = 0
        try:
            for b in train_loader:
                imgs, bbox, gt_d, gt_c = b["imgs"].to(device), b["bboxes"].to(device), b["density"].to(device), b["counts"].to(device)
                optim.zero_grad()
                with torch.cuda.amp.autocast(enabled=use_amp):
                    if seq_mode:
                        Lg, K = int(cfg.get("seq_grid", 14)), int(cfg.get("seq_vocab", 64))
                        patch = F.adaptive_avg_pool2d(gt_d.float(), (Lg, Lg)) * float((gt_d.shape[-1] // Lg) ** 2)  # SUM-pool: preserve mass
                        targets = patch.flatten(1).round().clamp(0, K - 1).long()
                        out = model(imgs, bbox, targets)
                        loss = F.cross_entropy(out["logits"].reshape(-1, K), targets.view(-1))
                    else:
                        out = model(imgs, bbox)
                        dens = out["density"] if isinstance(out, dict) else out
                        if dens.shape[-2:] != gt_d.shape[-2:]:  # models may emit low-res density; upsample to GT size
                            oh, ow = int(dens.shape[-2]), int(dens.shape[-1])
                            dens = F.interpolate(dens.float(), size=gt_d.shape[-2:], mode="bilinear", align_corners=False)
                            dens = dens * (oh * ow) / float(dens.shape[-2] * dens.shape[-1])  # sum-conserving
                        if bool(cfg.get("tail_reweight", False)):
                            pred_c = dens.float().flatten(1).sum(1)
                            base_i = F.mse_loss(dens.float(), gt_d, reduction="none").mean(dim=(1, 2, 3)) + w_cnt * (pred_c - gt_c).abs()
                            w = 1.0 / torch.clamp(gt_c.float(), min=1.0) ** float(cfg.get("tail_exp", 0.5))
                            w = w / w.mean().clamp_min(1e-8)
                            loss = (base_i * w).mean()
                        elif loss_fn_name == "huber":
                            dens_loss = F.huber_loss(dens.float(), gt_d, delta=huber_delta, reduction="mean")
                        else:
                            dens_loss = F.mse_loss(dens.float(), gt_d)
                        loss = dens_loss + w_cnt * F.l1_loss(dens.float().flatten(1).sum(1), gt_c)
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optim); scaler.update()
                ls += loss.item(); nb += 1
        except torch.cuda.OutOfMemoryError:
            oom = True; break
        sched.step()
        mae, rmse = evaluate(model, val_loader, device, seq=seq_mode,
                             max_frac=float(cfg.get("eval_frac", 1.0)))
        tag = ""
        if mae < best:
            best = mae; tag = " ***BEST"
            torch.save({"epoch": ep, "model": model.state_dict(), "best_mae": best}, ckpt)
        write_result("running", {"mae": mae, "rmse": rmse, "best_mae": best},
                     {"train_seconds": round(time.time() - t_start, 1), "epochs_done": ep},
                     {"oom": oom})
        print(f"E{ep:03d}/{epochs} loss={ls/max(nb,1):.4f} MAE={mae:.3f} RMSE={rmse:.3f} best={best:.3f} "
              f"[{time.time()-t_start:.0f}s]{tag}", flush=True)

    status = "failed" if oom else ("timeout" if timed_out else "success")
    mae, rmse = evaluate(model, val_loader, device, seq=seq_mode)
    write_result(status, {"mae": mae, "rmse": rmse, "best_mae": best},
                 {"train_seconds": round(time.time() - t_start, 1), "epochs_done": epochs if not (timed_out or oom) else "partial"},
                 {"oom": oom, "instability": not math.isfinite(best), "smoke": cfg["smoke"], "params_M": round(total_p, 2)})
    print(f"[engine] done status={status} best={best:.3f}", flush=True)


if __name__ == "__main__":
    main()

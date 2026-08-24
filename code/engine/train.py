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


def _sigmoid_focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    """Element-wise focal loss on raw logits; returns mean over all elements."""
    logits, targets = logits.float(), targets.float()
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * targets + (1 - p) * (1 - targets)
    a_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (a_t * (1 - p_t) ** gamma * bce).mean()


def _detect_targets(gt_d, grid, pool="mass", thresh=0.25):
    """Build point-detection targets from a GT density map [B,1,H,W].

    cls_t [B,1,g,g]: 1 where the cell likely contains an object center.
      pool="mass": expected #objects in cell = avg_density * HW/g^2 > thresh.
        Scale-free (density amplitude ~1/kernel_area, so absolute thresholds fail).
      pool="max"/"avg": raw pooled density > thresh (legacy/simple variants).
      NOTE: approximate fallback — under-labels very large objects whose kernel
      spreads mass over many cells; prefer exact point targets when available.
    reg_t [B,2,g,g]: (dx,dy) pixel offset from cell center to the per-cell density
      centroid (approximates nearest-peak offset; handles multi-object cells gracefully).
    """
    B, _, H, W = gt_d.shape
    if pool == "mass":
        pos_score = F.adaptive_avg_pool2d(gt_d.float(), (grid, grid)) * float(H * W) / float(grid * grid)
    elif pool == "max":
        pos_score = F.adaptive_max_pool2d(gt_d.float(), (grid, grid))
    else:
        pos_score = F.adaptive_avg_pool2d(gt_d.float(), (grid, grid))
    cls_t = (pos_score > thresh).float()
    ys = (torch.arange(H, device=gt_d.device, dtype=gt_d.dtype) + 0.5).view(1, 1, H, 1)
    xs = (torch.arange(W, device=gt_d.device, dtype=gt_d.dtype) + 0.5).view(1, 1, 1, W)
    denom = F.adaptive_avg_pool2d(gt_d, (grid, grid)).clamp_min(1e-8)
    cy = F.adaptive_avg_pool2d(gt_d * ys, (grid, grid)) / denom  # centroid row in px
    cx = F.adaptive_avg_pool2d(gt_d * xs, (grid, grid)) / denom  # centroid col in px
    gy = (torch.arange(grid, device=gt_d.device, dtype=gt_d.dtype) + 0.5).view(1, 1, grid, 1) * (H / grid)
    gx = (torch.arange(grid, device=gt_d.device, dtype=gt_d.dtype) + 0.5).view(1, 1, 1, grid) * (W / grid)
    reg_t = torch.cat([(cx - gx), (cy - gy)], dim=1)  # [B,2,g,g]
    return cls_t, reg_t


def _points_to_targets(points, grid, img_size, device):
    """Exact targets from GT points. points: list of [N_i,2] tensors (x,y in S-space).

    cls_t [B,1,g,g]: 1 where >=1 GT point falls in the cell.
    reg_t [B,2,g,g]: (dx,dy) offset from cell center to a GT point in that cell
    (last point wins on multi-object collisions; cells are 14x14 px at 392/28).
    """
    B = len(points)
    cls_t = torch.zeros(B, 1, grid, grid, device=device)
    reg_t = torch.zeros(B, 2, grid, grid, device=device)
    ps = float(img_size) / grid
    for i, pts in enumerate(points):
        if pts is None or pts.numel() == 0:
            continue
        pts = pts.to(device)
        cols = (pts[:, 0] / ps).long().clamp(0, grid - 1)
        rows = (pts[:, 1] / ps).long().clamp(0, grid - 1)
        cls_t[i, 0, rows, cols] = 1.0
        reg_t[i, 0, rows, cols] = pts[:, 0] - (cols.float() + 0.5) * ps
        reg_t[i, 1, rows, cols] = pts[:, 1] - (rows.float() + 0.5) * ps
    return cls_t, reg_t


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
                pts = []
                for _ in range(1 + int(torch.randint(0, 5, (1,), generator=g))):  # 1-5 objects
                    cy, cx = int(torch.randint(0, gs, (1,), generator=g)), int(torch.randint(0, gs, (1,), generator=g))
                    s = 1.0 + 2.0 * float(torch.rand(1, generator=g))
                    blob = torch.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * s * s))
                    dens = dens + blob / blob.sum().clamp_min(1e-6)   # each blob normalized, count ≈ number of objects
                    pts.append([float(cx + 0.5) * self.size / gs, float(cy + 0.5) * self.size / gs])
                return {"imgs": img, "bboxes": torch.tensor([x0, y0, x1, y1], dtype=torch.float32),
                        "density": dens[None], "counts": dens.sum(),
                        "points": torch.tensor(pts, dtype=torch.float32)}
        s = cfg["input_size"]
        bs = max(2, min(int(cfg.get("batch_size", 8)), 4))

        def collate_synth(batch):
            return {"imgs": torch.stack([b["imgs"] for b in batch]),
                    "bboxes": torch.stack([b["bboxes"] for b in batch]),
                    "density": torch.stack([b["density"] for b in batch]),
                    "counts": torch.stack([b["counts"] for b in batch]),
                    "points": [b["points"] for b in batch]}

        tr = Synth(16, s); va = Synth(8, s)
        return (DataLoader(tr, batch_size=bs, shuffle=True, collate_fn=collate_synth),
                DataLoader(va, batch_size=bs, collate_fn=collate_synth), bs)

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from torch.utils.data import DataLoader
    root = cfg.get("data_root", "/data/dataset/FSC147")
    size = int(cfg.get("input_size", 384))
    detect_mode = str(cfg.get("paradigm", "reg")) == "detect"
    if detect_mode:
        from data.fsc147 import FSC147Detect as DS, collate_detect as collate
    else:
        from data.fsc147 import FSC147Density as DS, collate_density as collate
    tr = DS(root, size, "train", augment=bool(cfg.get("augment", False)))
    va = DS(root, size, "val")
    bs = int(cfg.get("batch_size", 8))
    nw = int(cfg.get("num_workers", 4))
    return (DataLoader(tr, batch_size=bs, shuffle=True, num_workers=nw, collate_fn=collate, drop_last=True, pin_memory=True),
            DataLoader(va, batch_size=bs, shuffle=False, num_workers=nw, collate_fn=collate, pin_memory=True), bs)


def evaluate(model, loader, device, seq=False, ebc=False, detect=False, conf_thr=0.3, max_frac=1.0):
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
            elif ebc:
                probs = F.softmax(out["ebc_logits"], dim=1)  # [B,K,g,g]
                bin_indices = torch.arange(probs.shape[1], device=probs.device, dtype=torch.float32)
                expected = (probs * bin_indices.view(1, -1, 1, 1)).sum(dim=1)  # [B,g,g]
                pred = expected.flatten(1).sum(1).float().cpu()
            elif detect:
                pred = (torch.sigmoid(out["cls_logits"].float()) > conf_thr).sum(dim=(2, 3)).flatten().float().cpu()
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
    dl_hi = None
    if bool(cfg.get("dual_res_eval", False)) and not cfg["smoke"]:  # optional rider: eval@448 (≤10 LOC)
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from torch.utils.data import DataLoader
        from data.fsc147 import FSC147Density as DSH, collate_density as CH
        dl_hi = DataLoader(DSH(str(cfg.get("data_root", "/data/dataset/FSC147")), int(cfg.get("dual_res_size", 448)), "val"), batch_size=4, shuffle=False, num_workers=2, collate_fn=CH, pin_memory=True)
    if hasattr(model, 'param_groups'):
        optim = torch.optim.AdamW(model.param_groups(float(cfg.get("lr", 1e-3)),
                                                    float(cfg.get("weight_decay", 1e-4))))
    else:
        optim = torch.optim.AdamW(filter(lambda q: q.requires_grad, model.parameters()),
                                  lr=float(cfg.get("lr", 1e-3)), weight_decay=float(cfg.get("weight_decay", 1e-4)))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, epochs, eta_min=float(cfg.get("eta_min", 1e-6)))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    w_cnt = float(cfg.get("loss_count_weight", 0.3))
    loss_fn_name = cfg.get("loss_function", "mse")
    huber_delta = float(cfg.get("huber_delta", 5.0))
    paradigm = str(cfg.get("paradigm", "reg"))
    seq_mode = paradigm == "seq"
    ebc_flag = paradigm == "ebc"
    detect_mode = paradigm == "detect"
    ebc_mode = paradigm == "ebc"
    f_alpha = float(cfg.get("focal_alpha", 0.25))
    f_gamma = float(cfg.get("focal_gamma", 2.0))
    conf_thr = float(cfg.get("conf_threshold", 0.3))
    reg_w = float(cfg.get("reg_weight", 1.0))

    best = float("inf"); timed_out = False; oom = False
    swa_s, swa_e = cfg.get("swa_start"), cfg.get("swa_end")
    use_swa = swa_s is not None and swa_e is not None
    if use_swa:
        swa_s, swa_e = int(swa_s), int(swa_e)
    swa_sum, swa_lo, swa_hi = {}, None, None
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
                    elif ebc_mode:
                        out = model(imgs, bbox)
                        logits = out["ebc_logits"]  # [B, num_bins, g, g]
                        g = int(logits.shape[-1])
                        K = int(logits.shape[1])
                        # Build targets: SUM-pool density to grid, scale by area ratio
                        cell_area = float((gt_d.shape[-1] // g) ** 2)
                        patch_counts = F.adaptive_avg_pool2d(gt_d.float(), (g, g)) * cell_area
                        targets = patch_counts.squeeze(1).round().clamp(0, K - 1).long()  # [B, g, g]
                        loss = F.cross_entropy(logits, targets)
                    elif detect_mode:
                        out = model(imgs, bbox)
                        cls_pred, reg_pred = out["cls_logits"], out["reg_offsets"]  # [B,1,g,g] / [B,2,g,g]
                        g = int(cls_pred.shape[-1])
                        pts = b.get("points")
                        if pts is not None:  # exact GT points (preferred)
                            cls_t, reg_t = _points_to_targets(pts, g, int(imgs.shape[-1]), gt_d.device)
                        else:  # density-derived fallback (approximate)
                            cls_t, reg_t = _detect_targets(gt_d, g, pool=str(cfg.get("cls_pool", "mass")),
                                                           thresh=float(cfg.get("cls_threshold", 0.25)))
                        loss = _sigmoid_focal_loss(cls_pred, cls_t, alpha=f_alpha, gamma=f_gamma)
                        pos_mask = cls_t.bool().expand_as(reg_pred)
                        if pos_mask.any():
                            reg_loss = F.l1_loss(reg_pred[pos_mask], reg_t.expand_as(reg_pred)[pos_mask])
                            loss = loss + reg_w * reg_loss
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
                        else:
                            if loss_fn_name == "huber":
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
        if use_swa and swa_s <= ep <= swa_e:  # SWA-lite: CPU accumulator of trainable params
            for n, p in model.named_parameters():
                if p.requires_grad:
                    v = p.detach().float().cpu()
                    swa_sum[n] = v if n not in swa_sum else swa_sum[n] + v
            swa_lo, swa_hi = (ep if swa_lo is None else swa_lo), ep
        mae, rmse = evaluate(model, val_loader, device, seq=seq_mode, ebc=ebc_flag, detect=detect_mode,
                             conf_thr=conf_thr, max_frac=float(cfg.get("eval_frac", 1.0)))
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
    mae, rmse = evaluate(model, val_loader, device, seq=seq_mode, ebc=ebc_flag, detect=detect_mode, conf_thr=conf_thr)
    diag = {"oom": oom, "instability": not math.isfinite(best), "smoke": cfg["smoke"], "params_M": round(total_p, 2)}
    if dl_hi is not None:
        m448, r448 = evaluate(model, dl_hi, device, seq=seq_mode, ebc=ebc_flag, detect=detect_mode, conf_thr=conf_thr)
        diag.update(mae448=round(m448, 4), rmse448=round(r448, 4))
    if use_swa and swa_sum:  # uniform-average window (clamped to epochs actually run)
        cnt = swa_hi - swa_lo + 1
        swa_sd = {n: v / cnt for n, v in swa_sum.items()}
        torch.save({"epoch_window": [swa_lo, swa_hi], "truncated": swa_hi < swa_e, "model": swa_sd},
                   os.path.join(run_dir, "swa.pth"))
        model.load_state_dict(swa_sd, strict=False)
        swa_mae, swa_rmse = evaluate(model, val_loader, device, seq=seq_mode, ebc=ebc_flag, detect=detect_mode, conf_thr=conf_thr)
        diag.update(swa_mae=round(swa_mae, 4), swa_rmse=round(swa_rmse, 4),
                    swa_epoch_window=[swa_lo, swa_hi], swa_truncated=bool(swa_hi < swa_e))
        print(f"[engine] swa window=[{swa_lo},{swa_hi}] MAE={swa_mae:.3f} RMSE={swa_rmse:.3f}", flush=True)
    write_result(status, {"mae": mae, "rmse": rmse, "best_mae": best},
                 {"train_seconds": round(time.time() - t_start, 1), "epochs_done": epochs if not (timed_out or oom) else "partial"}, diag)
    print(f"[engine] done status={status} best={best:.3f}", flush=True)


if __name__ == "__main__":
    main()

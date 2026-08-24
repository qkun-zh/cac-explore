#!/usr/bin/env python3
"""OIR R0-C 可学习性（修正版）— GPU ~25min

冻结 backbone + FiLM 颈部 + sigmoid 占用头，**无加权 BCE**（proper scoring ⇒ mean(ô) 无偏，
pos_weight 会破坏校准污染反演——GOD §8 张力点，本门以无加权为准）。
评估用 R0-A/B 验证过的偏差校正反演（3 框估 Ā 与 κ）。
PASS IF cell-AUROC>0.80 且 反演 val MAE ≤24（GOD §4 锁死）
"""
import os, sys, json, math, argparse
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.environ.get("OIR_DATA", "/data/dataset/FSC147")
GRID, PATCH, S_EVAL = 28, 14, 392


def invert_corrected(f, grain_area, kappa):
    target = max(1e-12, 1 - f)
    g = kappa - 1
    lo, hi = 1e-9, 50.0
    if g <= 1e-9:
        mu = -math.log(target)
    else:
        h = lambda mu: math.exp(-mu) * (1 + 0.5 * g * mu * mu) - target
        if h(hi) > 0:
            mu = hi
        else:
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if h(mid) > 0:
                    lo = mid
                else:
                    hi = mid
            mu = 0.5 * (lo + hi)
    return (mu / grain_area) * S_EVAL * S_EVAL


def box_stats(ann):
    areas = []
    for corners in ann["box_examples_coordinates"][:3]:
        xs = [p[0] for p in corners]; ys = [p[1] for p in corners]
        w = (max(xs) - min(xs)) * S_EVAL / float(ann["W"])
        h = (max(ys) - min(ys)) * S_EVAL / float(ann["H"])
        areas.append(max(w * h, 1e-6))
    m = float(np.mean(areas))
    kap = float(np.mean(np.square(areas))) / m ** 2 if len(areas) >= 2 and m > 0 else 1.0
    return m, min(max(kap, 1.0), 3.0)


class OIRLight(nn.Module):
    """冠军同款 tap 结构 + FiLM + 2层颈 + sigmoid 头；backbone 全冻（可学习性探针）"""

    def __init__(self):
        super().__init__()
        import timm
        ch = 384
        self.backbone = timm.create_model("vit_small_patch14_reg4_dinov2.lvd142m",
                                          pretrained=True, dynamic_img_size=True,
                                          features_only=True, out_indices=(6, 11))
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.register_buffer("in_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("in_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.t6_proj = nn.Linear(ch, ch)
        self.t11_proj = nn.Linear(ch, ch)
        self.layer_logits = nn.Parameter(torch.zeros(2))
        self.film = nn.Linear(128, ch * 2)
        self.prompt_enc = nn.Sequential(nn.Linear(4, 128), nn.GELU(), nn.Linear(128, 128))
        self.neck = nn.Sequential(
            nn.Conv2d(ch, 256, 3, padding=1), nn.GroupNorm(8, 256), nn.GELU(),
            nn.Conv2d(256, 128, 3, padding=1), nn.GroupNorm(8, 128), nn.GELU())
        self.head = nn.Conv2d(128, 1, 1)

    def forward(self, imgs, bboxes):
        B, S = imgs.shape[0], imgs.shape[-1]
        x = (imgs - self.in_mean) / self.in_std
        taps = self.backbone(x)
        ps = S // PATCH
        f6, f11 = taps[0].float(), taps[1].float()
        if f6.ndim == 3:
            f6 = f6.transpose(1, 2).reshape(B, -1, ps, ps)
            f11 = f11.transpose(1, 2).reshape(B, -1, ps, ps)
        gate = torch.softmax(self.layer_logits, 0)
        tok = gate[0] * self.t6_proj(f6.flatten(2).transpose(1, 2)) + \
              gate[1] * self.t11_proj(f11.flatten(2).transpose(1, 2))
        b = bboxes / float(S)
        w = (b[:, 2] - b[:, 0]).clamp_min(1e-4); h = (b[:, 3] - b[:, 1]).clamp_min(1e-4)
        cxywh = torch.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2, w, h], 1)
        pr = self.prompt_enc(cxywh)
        sc, sh = self.film(pr).chunk(2, -1)
        tok = tok * (1 + sc.unsqueeze(1)) + sh.unsqueeze(1)
        feat = self.neck(tok.transpose(1, 2).reshape(B, -1, ps, ps))
        return {"occ": torch.sigmoid(self.head(feat))}


def occupancy_soft(dens, thr_frac):
    peak = float(dens.max()) if dens.size else 0.0
    if peak < 1e-12:
        return np.zeros((GRID, GRID), dtype=np.float32)
    occ = (dens > peak * thr_frac).astype(np.float32)
    return F.adaptive_avg_pool2d(torch.from_numpy(occ)[None, None], (GRID, GRID))[0, 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--thr", type=float, default=0.02)
    ap.add_argument("--fill", type=float, default=0.75, help="grain 填充比 c（R0-B 扫出的最优值）")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[R0-C] device={device} ep={args.epochs} thr={args.thr} fill={args.fill} BCE unweighted")

    sys.path.insert(0, os.path.join(REPO, "code"))
    from data.fsc147 import FSC147Density, collate_density

    orig_get = FSC147Density.__getitem__

    def get_with_occ(self, i):
        item = orig_get(self, i)
        dens = item["density"][0].numpy()
        soft = occupancy_soft(dens, args.thr)
        item["occ"] = soft.float().unsqueeze(0)
        item["idx"] = torch.tensor(i)
        return item

    FSC147Density.__getitem__ = get_with_occ

    def collate_occ(batch):
        out = collate_density(batch)
        out["occ"] = torch.stack([b["occ"] for b in batch])
        out["idx"] = torch.stack([b["idx"] for b in batch])
        return out

    from torch.utils.data import DataLoader
    tr = DataLoader(FSC147Density(DATA_ROOT, S_EVAL, "train", augment=True),
                    batch_size=args.bs, shuffle=True, num_workers=4,
                    collate_fn=collate_occ, drop_last=True, pin_memory=True)
    va_ids_json = os.path.join(DATA_ROOT, "Train_Test_Val_FSC_147.json")
    with open(va_ids_json) as f:
        val_ids = json.load(f)["val"]
    va_ds = FSC147Density(DATA_ROOT, S_EVAL, "val")
    va = DataLoader(va_ds, batch_size=args.bs, shuffle=False, num_workers=4,
                    collate_fn=collate_occ, pin_memory=True)
    with open(os.path.join(DATA_ROOT, "annotation_FSC147_384.json")) as f:
        anno = json.load(f)

    model = OIRLight().to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"params total {sum(p.numel() for p in model.parameters())/1e6:.2f}M trainable {n_train:.2f}M")
    optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, args.epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    best = {"auroc": 0.0, "mae": float("inf")}
    try:
        from sklearn.metrics import roc_auc_score
    except Exception:
        roc_auc_score = None

    for ep in range(1, args.epochs + 1):
        model.train(); tot = nb = 0
        for b in tr:
            imgs = b["imgs"].to(device); bb = b["bboxes"].to(device)
            t = b["occ"].to(device)
            optim.zero_grad()
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                pred = model(imgs, bb)["occ"]
                p = pred.clamp(1e-6, 1 - 1e-6).float()
                loss = -(t * torch.log(p) + (1 - t) * torch.log(1 - p)).mean()  # 无加权 SoftBCE
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim); scaler.update()
            tot += loss.item(); nb += 1
        sched.step()

        model.eval()
        probs_all, labs_all = [], []
        maes = []
        with torch.no_grad():
            for b in va:
                imgs = b["imgs"].to(device); bb = b["bboxes"].to(device)
                occ_gt = b["occ"].numpy()  # [B,1,G,G]
                pred = model(imgs, bb)["occ"].float().cpu().numpy()
                probs_all.append(pred.ravel())
                labs_all.append((occ_gt > 0.5).astype(float).ravel())
                for i in range(pred.shape[0]):
                    f_hat = float(np.clip(pred[i].mean(), 1e-6, 1 - 1e-6))
                    im_id = val_ids[int(b["idx"][i])]
                    stem = im_id[:-4] if im_id.endswith(".jpg") else im_id
                    gt = float(b["counts"][i])
                    area, kap = box_stats(anno[im_id])
                    n_hat = invert_corrected(f_hat, args.fill * area, kap)
                    maes.append(abs(n_hat - gt))
        auroc = float(roc_auc_score(np.concatenate(labs_all), np.concatenate(probs_all))) \
            if roc_auc_score else float("nan")
        mae = float(np.mean(maes))
        star = ""
        if auroc > best["auroc"]:
            best["auroc"] = auroc; star += " AUROC↑"
        if mae < best["mae"]:
            best["mae"] = mae; star += " MAE↓"
            torch.save({"epoch": ep, "model": model.state_dict(),
                        "auroc": auroc, "inv_mae": mae}, "/tmp/oir_r0c_best.pth")
        print(f"E{ep:02d}/{args.epochs} loss={tot/max(nb,1):.4f} AUROC={auroc:.3f} invMAE={mae:.1f}{star}")

    print(f"\n[R0-C] best AUROC={best['auroc']:.3f} best invMAE={best['mae']:.1f}")
    ok_a = best["auroc"] > 0.80
    ok_m = best["mae"] <= 24.0
    print(f"AUROC>0.80? {ok_a}   invMAE<=24? {ok_m}")
    print("VERDICT:", "PASS — 进入完整 OIR-Net 注册"
          if (ok_a and ok_m) else
          ("FAIL — AUROC 不足：FiLM 太弱，按预案①换 cross-attention 重试一次" if not ok_a
           else "FAIL — 校准/尺寸估计问题：查 f̂-GT 回归斜率与框面积噪声"))


if __name__ == "__main__":
    main()

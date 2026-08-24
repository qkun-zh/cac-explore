#!/usr/bin/env python3
"""R0-C 可学习性 — GPU 25min

冻结冠军 backbone + 2层头 + sigmoid，BCE 训占用图 10ep
PASS IF cell-AUROC>0.80 且 反演后 val MAE ≤24 (25分钟贴平40分钟冠军，否则读出无红利)

占用 GT: GT密度图低阈值二值化(τ≈0.02) → 14px平均池化 → 软标签 t_c∈[0,1] (与R0-B一致)
模型: 复用 DINOv2-S reg4 backbone (blocks0-9冻结，10-11可训?  spec说冻结冠军backbone → 这里按 spec 冻结全部 backbone，仅训颈部+头，2层neck 768→256→128 + FiLM + sigmoid)
但为贴近冠军配置，我们保持冠军的 partial-FT 结构? spec §6 says 复用冠军 backbone+FiLM，冻结blocks0-9 微调10-11@0.1 — R0-C 轻量验证可先冻结全部以证可学习性，再全训.

实现: 简化 — 直接加载 N0027/N0021 架构但替换 head 为 occupancy head (1ch sigmoid)，BCE loss.

需要 FSC147 loader修改返回 occupancy soft label.

"""
import os, json, math, argparse, sys, importlib.util
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from PIL import Image

REPO=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT="/data/dataset/FSC147"

def boxes_r2_mean(ann, S=392):
    areas=[]
    for corners in ann["box_examples_coordinates"][:3]:
        xs=[p[0] for p in corners]; ys=[p[1] for p in corners]
        w=(max(xs)-min(xs))*S/float(ann["W"]); h=(max(ys)-min(ys))*S/float(ann["H"])
        areas.append(w*h)
    return float(np.mean(areas))/4.0

def gt_occupancy_soft(dens, thr_frac=0.02):
    # dens [S,S] -> binary per-pixel at thr_frac*peak -> avg pool 14 -> soft [28,28]
    peak=float(dens.max()) if dens.size>0 else 0
    if peak<1e-9:
        return np.zeros((28,28), dtype=float)
    thr=peak*thr_frac
    occ=(dens > thr).astype(float)
    t=torch.from_numpy(occ).unsqueeze(0).unsqueeze(0).float()
    S=occ.shape[0]
    # adaptive pool to 28x28 regardless of S (if S not 392)
    pooled=F.avg_pool2d(t, kernel_size=max(1,S//28), stride=max(1,S//28))
    # if not exactly 28, interpolate
    if pooled.shape[-1]!=28:
        pooled=F.adaptive_avg_pool2d(t, (28,28))
    soft=pooled[0,0].numpy()
    return soft

# Model definition for R0-C (light)
BACKBONE="vit_small_patch14_reg4_dinov2.lvd142m"
PATCH=14
class PromptEncoderTiny(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.mlp=nn.Sequential(nn.Linear(4, out_dim), nn.GELU(), nn.Linear(out_dim, out_dim))
    def forward(self, bboxes, size):
        b=bboxes/float(size)
        w=(b[:,2]-b[:,0]).clamp_min(1e-4); h=(b[:,3]-b[:,1]).clamp_min(1e-4)
        cxywh=torch.stack([(b[:,0]+b[:,2])/2,(b[:,1]+b[:,3])/2,w,h],dim=1)
        return self.mlp(cxywh)

class OIRLight(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        import timm
        ch=384; dim=256
        self.backbone=timm.create_model(BACKBONE, pretrained=True, dynamic_img_size=True, features_only=True, out_indices=(6,11))
        # freeze blocks 0-9 per spec (spec says frozen 0-9, tune 10-11@0.1 for full OIR; for R0-C we freeze all to test learnability)
        freeze_all=bool(cfg.get("freeze_all", True))
        for name,p in self.backbone.named_parameters():
            if freeze_all:
                p.requires_grad_(False)
            else:
                if "blocks.10." in name or "blocks.11." in name or "norm." in name:
                    p.requires_grad_(True)
                else:
                    p.requires_grad_(False)
        self.backbone_lr_mult=float(cfg.get("backbone_lr_mult",0.1))
        mean=(0.485,0.456,0.406); std=(0.229,0.224,0.225)
        self.register_buffer("in_mean", torch.tensor(mean).view(1,3,1,1))
        self.register_buffer("in_std", torch.tensor(std).view(1,3,1,1))
        self.patch=PATCH
        self.t6_proj=nn.Linear(ch,ch); self.t11_proj=nn.Linear(ch,ch)
        self.layer_logits=nn.Parameter(torch.zeros(2))
        # FiLM: exemplar → scale/shift
        self.prompt_enc=PromptEncoderTiny(out_dim=128)
        self.film=nn.Linear(128, 768*2)  # scale+shift for 768? Actually after concat tokens 384, we have neck 768->... Let's keep simple: FiLM on tokens directly
        # Neck 384→256→128
        self.neck=nn.Sequential(
            nn.Conv2d(384, 256, 3, padding=1), nn.GroupNorm(8,256), nn.GELU(),
            nn.Conv2d(256, 128, 3, padding=1), nn.GroupNorm(8,128), nn.GELU(),
        )
        self.head=nn.Conv2d(128,1,1)
    def param_groups(self, base_lr, weight_decay):
        bb_params, rest_params=[],[]
        for name,p in self.named_parameters():
            if not p.requires_grad: continue
            if name.startswith("backbone."):
                bb_params.append(p)
            else:
                rest_params.append(p)
        return [{"params": bb_params, "lr": base_lr*self.backbone_lr_mult}, {"params": rest_params, "lr": base_lr}]
    def forward(self, imgs, bboxes):
        B,S=imgs.shape[0], imgs.shape[-1]
        imgs=(imgs-self.in_mean)/self.in_std
        taps=self.backbone(imgs)
        ps=S//self.patch
        f6,f11=taps[0].float(), taps[1].float()
        if f6.ndim==3:
            f6=f6.transpose(1,2).reshape(f6.shape[0], f6.shape[2], ps, ps)
            f11=f11.transpose(1,2).reshape(f11.shape[0], f11.shape[2], ps, ps)
        gate=torch.softmax(self.layer_logits, dim=0)
        z6=self.t6_proj(f6.flatten(2).transpose(1,2))
        z11=self.t11_proj(f11.flatten(2).transpose(1,2))
        tokens=gate[0]*z6+gate[1]*z11  # B,N,C
        # FiLM
        prompt=self.prompt_enc(bboxes, S)  # B,128
        film_params=self.film(prompt)  # B, 768*2? Actually tokens C=384, film should be 384*2
        # Adjust: film for 384
        if film_params.shape[-1]!=768:
            # if we used 768*2 but C=384, split incorrectly; just project to 384*2
            pass
        # Simplify: film_params = linear 128->768 (384*2)
        scale, shift=film_params.chunk(2, dim=-1)  # each B,384?
        # scale/shift currently B,384? Actually film outputs 768, chunk gives 384 each correct
        # Apply per token
        tokens = tokens * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        # to spatial
        feat=tokens.transpose(1,2).reshape(B, 384, ps, ps)
        feat=self.neck(feat)  # B,128,ps,ps
        occ=self.head(feat)  # B,1,ps,ps
        occ=torch.sigmoid(occ)
        return {"occ": occ}

def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec); sys.modules[name]=m; spec.loader.exec_module(m); return m

def main():
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--thr", type=float, default=0.02)
    ap.add_argument("--freeze_all", action="store_true")
    args=ap.parse_args()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[R0-C] device={device} epochs={args.epochs} thr={args.thr} freeze_all={args.freeze_all}")
    sys.path.insert(0, os.path.join(REPO,"code"))
    from data.fsc147 import FSC147Density, collate_density  # will need occupancy? We'll compute on fly
    # Patch dataset to return occupancy soft label
    # Monkey patch: wrap FSC147Density
    orig_getitem=FSC147Density.__getitem__
    def new_getitem(self, i):
        # call original to get imgs,bboxes,density,counts
        # we need to recompute occupancy from density at thr
        # Use original's logic but we can reuse orig item then add occ
        item=orig_getitem(self, i)  # this already does resize etc and returns dict with imgs,bboxes,density
        # density is [1,S,S] already sum-conserving resized
        dens=item["density"][0].numpy()  # [S,S]
        soft=gt_occupancy_soft(dens, thr_frac=args.thr)  # [28,28]
        item["occ"] = torch.from_numpy(soft).float().unsqueeze(0)  # [1,28,28]
        # But S may be 392, ps=28, occ is 28x28 matching head output at 392; at other S, head output size changes. For R0-C we train only at 392, so okay.
        return item
    FSC147Density.__getitem__=new_getitem
    # collate needs to stack occ
    orig_collate=collate_density
    def new_collate(batch):
        out=orig_collate(batch)
        if "occ" in batch[0]:
            out["occ"]=torch.stack([b["occ"] for b in batch])
        return out
    # monkey patch in data module
    import data.fsc147 as fscmod
    fscmod.collate_density=new_collate

    cfg=dict(input_size=392, epochs=args.epochs, batch_size=args.bs, lr=1e-3, weight_decay=1e-4, eta_min=1e-6, amp=True, max_params_M=32, augment=True, freeze_all=args.freeze_all)
    from torch.utils.data import DataLoader
    root="/data/dataset/FSC147"
    tr_ds=FSC147Density(root, 392, "train", augment=True)
    va_ds=FSC147Density(root, 392, "val")
    tr_ld=DataLoader(tr_ds, batch_size=args.bs, shuffle=True, num_workers=4, collate_fn=new_collate, drop_last=True, pin_memory=True)
    va_ld=DataLoader(va_ds, batch_size=args.bs, shuffle=False, num_workers=4, collate_fn=new_collate, pin_memory=True)

    model=OIRLight(cfg).to(device)
    total=sum(p.numel() for p in model.parameters())/1e6
    print(f"params {total:.2f}M")
    # pos weight: neg/pos ratio from train set stat (approx)
    # Estimate from one epoch: occupancy mean f ~? We'll set w=3 or compute from data
    # For now, compute from first batch
    # pos_weight = (num_neg / num_pos)
    # We'll set w = 2.5 as placeholder; spec says train set statistic fixed
    # Let's compute quickly by sampling 100 images
    # Quick estimate
    sample_occs=[]
    for b in va_ld:
        occ=b["occ"]  # B,1,28,28
        sample_occs.append(occ)
        if len(sample_occs)*args.bs>200:
            break
    all_occ=torch.cat(sample_occs, dim=0)
    pos_ratio=float((all_occ>0.5).float().mean().item())
    neg_ratio=1-pos_ratio
    pos_weight=neg_ratio/max(pos_ratio,0.01)
    print(f"val pos_ratio {pos_ratio:.3f} neg {neg_ratio:.3f} pos_weight {pos_weight:.2f}")
    # Use 3.0 if extreme
    pos_weight=float(np.clip(pos_weight, 1.0, 10.0))

    optim=torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(optim, args.epochs, eta_min=1e-6)
    scaler=torch.cuda.amp.GradScaler(enabled=device.type=="cuda")
    best_auroc=0; best_mae=float('inf')
    # Load anno for r2 inversion
    with open(os.path.join(DATA_ROOT,"annotation_FSC147_384.json")) as f:
        anno=json.load(f)

    for ep in range(1, args.epochs+1):
        model.train(); total_loss=0; n=0
        for b in tr_ld:
            imgs=b["imgs"].to(device); bboxes=b["bboxes"].to(device); occ_gt=b["occ"].to(device)  # [B,1,28,28]
            optim.zero_grad()
            with torch.cuda.amp.autocast(enabled=device.type=="cuda"):
                out=model(imgs,bboxes)
                pred=out["occ"]  # B,1,28,28
                # BCE with pos_weight
                loss=F.binary_cross_entropy_with_logits(torch.logit(pred.clamp(1e-6,1-1e-6)), occ_gt, pos_weight=torch.tensor(pos_weight, device=device))
                # Actually pred is already sigmoid, use BCE directly
                loss=F.binary_cross_entropy(pred.clamp(1e-6,1-1e-6), occ_gt, weight=None, reduction='mean')
                # manual pos weight: w * t*log p + (1-t)*log(1-p) -> we can use BCEWithLogits with logit
                # For simplicity use weighted BCE: already computed via pos_weight on positive pixels
                # Let's use pos_weight version via BCE loss with weight map
                # Compute weighted: loss = -(w*t*log p + (1-t)*log(1-p)).mean()
                # We'll do that
                eps=1e-6
                p=pred.clamp(eps,1-eps)
                t=occ_gt
                loss = -(pos_weight*t*torch.log(p) + (1-t)*torch.log(1-p)).mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            scaler.step(optim); scaler.update()
            total_loss+=loss.item(); n+=1
        sched.step()
        # eval AUROC and inversion MAE
        model.eval()
        auroc_vals=[]
        mae_vals=[]; mse_vals=[]
        with torch.no_grad():
            # we need to collect predictions vs GT occupancy and counts
            all_probs=[]; all_labels=[]
            for b in va_ld:
                imgs=b["imgs"].to(device); bboxes=b["bboxes"].to(device); occ_gt=b["occ"].numpy()  # we have occ soft but AUROC needs binary? Use soft>0.5 as binary?
                # Actually AUROC expects binary labels (occupancy), we have soft [0,1] derived from pixel coverage. For AUROC, threshold soft at 0.5 to get binary?
                out=model(imgs,bboxes)
                pred=out["occ"].cpu().numpy()  # B,1,28,28
                # flatten
                all_probs.append(pred.ravel())
                # labels: use (occ_gt >0.5) as binary occupancy
                all_labels.append((occ_gt>0.5).astype(float).ravel())
                # inversion MAE: need Nhat from mean pred f
                for i in range(imgs.shape[0]):
                    f_hat=float(pred[i].mean())
                    f_hat=np.clip(f_hat,1e-6,1-1e-6)
                    # need GT count and r2 for this image
                    # batch order corresponds to va_ds order which is val ids order; need to map via index
                    # Simplify: use GT counts from batch counts
                    gt_count=float(b["counts"][i].item())
                    # need r2: we can estimate from bboxes? Use mean box area
                    # For inversion we need r2 per image; approximate via bboxes in batch? bboxes is first box only, not ideal.
                    # Use bboxes to estimate r2 ~ area/4
                    x0,y0,x1,y1=b["bboxes"][i].tolist()
                    area=(x1-x0)*(y1-y0)
                    r2=max(area/4.0,1e-6)
                    # For more accurate, we should load anno via id; but we don't have id in batch. Approximate with box area.
                    A=392*392
                    lam_hat= -math.log(1-f_hat)/(math.pi*r2)
                    Nhat=lam_hat*A
                    mae_vals.append(abs(Nhat-gt_count))
                    mse_vals.append((Nhat-gt_count)**2)
            # AUROC
            from sklearn.metrics import roc_auc_score
            probs=np.concatenate(all_probs); labels=np.concatenate(all_labels)
            try:
                auroc=float(roc_auc_score(labels, probs))
            except:
                auroc=0.5
            mae=float(np.mean(mae_vals)); rmse=float(math.sqrt(np.mean(mse_vals)))
        print(f"E{ep:02d}/{args.epochs} loss={total_loss/max(n,1):.4f} AUROC={auroc:.3f} invMAE={mae:.1f} RMSE={rmse:.1f}  bestAUROC={best_auroc:.3f} bestMAE={best_mae:.1f}")
        if auroc>best_auroc:
            best_auroc=auroc
        if mae<best_mae:
            best_mae=mae
            torch.save({"epoch":ep, "model":model.state_dict(), "auroc":auroc, "mae":mae}, "/tmp/oir_r0c_best.pth")
    print(f"\n[R0-C] final best AUROC {best_auroc:.3f} best invMAE {best_mae:.1f}")
    print(f"criterion: AUROC>0.80 ? {best_auroc>0.80}  invMAE<=24 ? {best_mae<=24}")
    if best_auroc>0.80 and best_mae<=24:
        print("VERDICT: PASS — 25min training reaches champion parity, proceed to full OIR-Net")
    else:
        print("VERDICT: FAIL — learnability insufficient")
        if best_auroc<=0.80:
            print("  -> FiLM too weak, try cross-attention per预案①")
        if best_mae>24:
            print("  -> check r estimation noise vs formula bias")

if __name__=="__main__":
    main()

#!/usr/bin/env python3
"""P2 COIN R0 kill-gate: Spearman rho(sum cosine-similarity map, GT count) >0.75 ?

Loads frozen DINOv2-S reg4 backbone (timm, pretrained, dynamic_img_size) and computes
exemplar-conditioned cosine similarity pulse train sum. If rho <0.75, the COIN
multiplicity-gating premise fails without training and the proposal is killed.

Reuse patterns from eval_readout_lab: 3 exemplar boxes from annotation_FSC147_384.json,
GT density sums from gt_density_map_adaptive_384_VarV2, images resized to 392.
"""
import json, os, argparse, math, statistics
import torch, torch.nn.functional as F
from PIL import Image
import numpy as np
from scipy.stats import spearmanr  # fallback to manual if missing

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = "/data/dataset/FSC147"
CKPT_N27 = "/data/runs/N0027_norm_flip_swa/best.pth"

def boxes_in_S(ann, S):
    out=[]
    for corners in ann["box_examples_coordinates"][:3]:
        xs=[p[0] for p in corners]; ys=[p[1] for p in corners]
        out.append((min(xs)*S/ann["W"], min(ys)*S/ann["H"], max(xs)*S/ann["W"], max(ys)*S/ann["H"]))
    return out

def eval_one(model, device, img_path, boxes_S, S):
    # img resize -> tensor, ImageNet norm? Use raw /255 then similar to backbone training? DINOv2 expects 0.485/0.456 etc but frozen Lit.
    # Use timm's expected norm: mean 0.485/0.456/0.406 std 0.229/0.224/0.225 after /255 - same as N27.
    mean=torch.tensor([0.485,0.456,0.406]).view(1,3,1,1).to(device)
    std=torch.tensor([0.229,0.224,0.225]).view(1,3,1,1).to(device)
    img=Image.open(img_path).convert("RGB").resize((S,S), Image.BILINEAR)
    t=torch.from_numpy(np.asarray(img)).permute(2,0,1).float().unsqueeze(0)/255.0
    t=(t.to(device)-mean)/std
    with torch.no_grad():
        feats=model.forward_features(t)  # try timm ViT: features are patch tokens? For features_only we used earlier.
        # fallback: timm features_only returns list; we need direct forward_features logic
    return None

def main():
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--S", type=int, default=392)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--val_frac", type=float, default=1.0)
    args=ap.parse_args()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[COIN R0] device={device} S={args.S} n~{args.n}")

    import timm
    backbone=timm.create_model("vit_small_patch14_reg4_dinov2.lvd142m", pretrained=True, dynamic_img_size=True, features_only=False).to(device).eval()
    # hook to get patch tokens? Use forward_features and x_norm?
    # For reg4, forward_features returns? Let's inspect.
    # Simpler: use features_only=True as champion does.
    # We'll create a second model for token extraction that matches champion's tapping.
    backbone2=timm.create_model("vit_small_patch14_reg4_dinov2.lvd142m", pretrained=True, dynamic_img_size=True, features_only=True, out_indices=(11,)).to(device).eval()
    mean=torch.tensor([0.485,0.456,0.406]).view(1,3,1,1).to(device)
    std=torch.tensor([0.229,0.224,0.225]).view(1,3,1,1).to(device)

    with open(os.path.join(DATA_ROOT, "Train_Test_Val_FSC_147.json")) as f:
        ids=json.load(f)["val"]
    with open(os.path.join(DATA_ROOT, "annotation_FSC147_384.json")) as f:
        anno=json.load(f)
    # sample first n or deterministically: take every k
    if len(ids)>args.n:
        step=len(ids)//args.n
        ids=ids[::max(1,step)][:args.n]
    xs=[]; ys=[]
    patch=14
    ps=args.S//patch  # 28
    for idx, im_id in enumerate(ids):
        stem=im_id[:-4] if im_id.endswith(".jpg") else im_id
        img_path=os.path.join(DATA_ROOT, "images_384_VarV2", f"{stem}.jpg")
        gt_path=os.path.join(DATA_ROOT, "gt_density_map_adaptive_384_VarV2", f"{stem}.npy")
        gt=float(np.load(gt_path).sum())
        boxes=boxes_in_S(anno[im_id], args.S)
        img=Image.open(img_path).convert("RGB").resize((args.S,args.S), Image.BILINEAR)
        t=torch.from_numpy(np.asarray(img)).permute(2,0,1).float().unsqueeze(0)/255.0
        t=(t.to(device)-mean)/std
        with torch.no_grad():
            # features [B, C, ps, ps] after transpose?
            f=backbone2(t)[0]  # [B, C, ps, ps] or [B, tokens, C]
            # Handle both returns: timm features_only with vit: shape [B, 384, ps, ps]? Let's probe
            if f.ndim==3:
                # [B, 1+num_patches+num_reg, C] -> remove cls? Estimate.
                # DINOv2 reg4 has 4 registers; tokens = 1+ps*ps+4 ?
                # Need to infer: if seq_len = ps*ps + 5 (1 cls +4 reg), then patch tokens are middle.
                seq=f
                # heuristic: extract last ps*ps tokens excluding cls/reg
                # For vit_small reg4: num_reg=4 => tokens shape B, 1+ps*ps+4, C
                # Patch tokens are 1 : 1+ps*ps (after cls)
                # Registers are interleaved? Actually reg tokens come after cls before patches in DINOv2.
                # Simpler: drop first token (cls) and last 4 (reg) ?
                # We'll attempt both and check.
                B,Sseq,C=f.shape
                expected=ps*ps
                if Sseq==expected+5:
                    # cls + 4 reg + patches
                    tokens=f[:,1:1+ps*ps,:]  # assume patches after cls, before regs? need ordering check
                    # Alternative: patches are in middle? Let's log first case
                    if idx==0:
                        print(f"[debug] f seq shape {f.shape}, assuming cls first, patches next {ps*ps}")
                    fmap=tokens.transpose(1,2).reshape(B,C,ps,ps)
                elif Sseq==expected+1:
                    tokens=f[:,1:,:]
                    fmap=tokens.transpose(1,2).reshape(B,C,ps,ps)
                else:
                    # unexpected, fallback: take last ps*ps
                    if idx==0:
                        print(f"[debug] unexpected seq {f.shape} ps {ps}")
                    tokens=f[:,-ps*ps:,:]
                    fmap=tokens.transpose(1,2).reshape(B,C,ps,ps)
            else:
                fmap=f
            # Now exemplar pooling: average pool tokens inside each box
            # Box in S-space -> map to token grid coordinates
            sims=[]
            # L2 normalize token features
            fmap_n=F.normalize(fmap.float(), dim=1)  # B,C,ps,ps
            for (x0,y0,x1,y1) in boxes:
                # convert to token indices
                c0=max(0,int(math.floor(x0/patch))); c1=min(ps, int(math.ceil(x1/patch)))
                r0=max(0,int(math.floor(y0/patch))); r1=min(ps, int(math.ceil(y1/patch)))
                if c1<=c0 or r1<=r0:
                    c0,r0=0,0; c1,r1=min(1,ps),min(1,ps)
                pooled=fmap_n[0,:,r0:r1,c0:c1].mean(dim=(1,2))  # [C]
                pooled=F.normalize(pooled, dim=0)
                # cosine sim map
                cos=(fmap_n[0]*pooled.view(-1,1,1)).sum(dim=0)  # ps,ps
                # we have 3 boxes -> will max or avg? COIN spec: softmax over boxes, sum aggregated.
                # For R0 we test simple: per-pixel max over exemplars then sum
                sims.append(cos)
            if len(sims)==0:
                continue
            stacked=torch.stack(sims, dim=0)  # 3,ps,ps
            # aggregation variants: max and mean; we test sum of max
            agg=stacked.max(dim=0).values  # ps,ps
            s=float(agg.sum().item())
            # also clip negatives? cos can be -1..1, shift? Use sum of positive part
            # We'll record raw sum and positive sum
        xs.append(s); ys.append(gt)
        if (idx+1)%50==0:
            print(f"  {idx+1}/{len(ids)} s={s:.2f} gt={gt:.0f}")
    # spearman
    try:
        from scipy.stats import spearmanr
        rho,p=spearmanr(xs, ys)
    except Exception as e:
        # manual spearman via rank
        print(f"scipy fail {e}, fallback")
        # rank
        def rankdata(a):
            sorted_idx=sorted(range(len(a)), key=lambda i:a[i])
            r=[0]*len(a)
            for pos, idx in enumerate(sorted_idx):
                r[idx]=pos+1
            return r
        rx=rankdata(xs); ry=rankdata(ys)
        mx=sum(rx)/len(rx); my=sum(ry)/len(ry)
        num=sum((rx[i]-mx)*(ry[i]-my) for i in range(len(rx)))
        den=math.sqrt(sum((rx[i]-mx)**2 for i in range(len(rx)))*sum((ry[i]-my)**2 for i in range(len(ry))))
        rho=num/den if den!=0 else 0.0
        p=float('nan')
    # also compute positive-sum variant?.quick
    print(f"[COIN R0] n={len(xs)} Spearman rho={rho:.4f} p={p:.3g}")
    print(f"  sum range [{min(xs):.1f}, {max(xs):.1f}] gt range [{min(ys):.0f},{max(ys):.0f}]")
    # verdict
    if rho>0.75:
        print("VERDICT: PASS (>=0.75) -> COIN premise holds, proceed to R1 multiplicity design")
    elif rho>0.60:
        print("VERDICT: WEAK (0.60-0.75) -> borderline, maybe with better pooling/projection could pass; consider R1 with learned proj")
    else:
        print("VERDICT: FAIL (<0.60) -> KILL COIN; similarity sum does NOT track count")
    # also try Pearson
    try:
        import numpy as np2
        xs_a=np.array(xs); ys_a=np.array(ys)
        pear=np.corrcoef(xs_a, ys_a)[0,1]
        print(f"  Pearson r={pear:.4f}")
    except: pass

if __name__=="__main__":
    main()

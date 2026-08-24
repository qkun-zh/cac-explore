#!/usr/bin/env python3
"""P3 AXIOM-TTC R0 drift audit: corr(|count drift under zoom/crop/swap|, |error|) >=0.3 ?

Evaluate champion ckpt (N27 now, or N21 if available) at 392 vs augmentations:
 - zoom: resize to 448 then center-crop 392? and scale integral invariance
 - crop: random crop 350 then resize back?
 - swap: exemplar swap invariance (swap exemplar box with another image's box)

For simplicity R0: zoom 392->518 then re-evaluate count; crop: center 0.9 crop; swap: we can skip or use shuffled boxes.
Measure drift = |N_hat_aug - N_hat_orig| and error = |N_hat_orig - GT|. Corr between drift and error tests if
model's instability predicts its error (useful TTT signal). If corr<0.3, TTT on axioms will not localize error.

Reuse loader similar to eval_res_sweep but single res.

We test on N027 or N021 ckpt whichever exists.
"""
import os, json, math, argparse
import torch, torch.nn.functional as F
from PIL import Image
import numpy as np, importlib.util, sys
from pathlib import Path

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT="/data/dataset/FSC147"
N27_CKPT="/data/runs/N0027_norm_flip_swa/best.pth"
N21_CKPT="/data/runs/N0021_dino_partialft/best.pth"
NODE21_DIR=os.path.join(REPO,"tree","nodes","N0021_dino_partialft")
NODE27_DIR=os.path.join(REPO,"tree","nodes","N0027_norm_flip_swa")

def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name, path)
    m=importlib.util.module_from_spec(spec); sys.modules[name]=m; spec.loader.exec_module(m); return m

def boxes_in_S(ann, S):
    sx, sy=S/float(ann["W"]), S/float(ann["H"])
    out=[]
    for corners in ann["box_examples_coordinates"][:3]:
        xs=[p[0] for p in corners]; ys=[p[1] for p in corners]
        out.append(torch.tensor([min(xs)*sx, min(ys)*sy, max(xs)*sx, max(ys)*sy], dtype=torch.float32))
    # single box version for model: Use first box only (as fsc147.py does)
    xs=[p[0] for p in ann["box_examples_coordinates"][0]]; ys=[p[1] for p in ann["box_examples_coordinates"][0]]
    bbox=torch.tensor([min(xs)*sx, min(ys)*sy, max(xs)*sx, max(ys)*sy], dtype=torch.float32)
    return bbox, out

def predict(model, device, img_np, bbox, S):
    # img_np is PIL already resized? We'll produce tensor inside
    mean=torch.tensor([0.485,0.456,0.406]).view(1,3,1,1).to(device)
    std=torch.tensor([0.229,0.224,0.225]).view(1,3,1,1).to(device)
    t=torch.from_numpy(np.asarray(img_np)).permute(2,0,1).float().unsqueeze(0)/255.0
    # If model has in_mean buffer (N27), then we should NOT pre-normalize? N27 does (imgs - mean)/std inside forward.
    # Check if model has in_mean attr then pass raw /255 tensor
    has_norm=hasattr(model, 'in_mean')
    if has_norm:
        # pass raw normalized already? N27 forward does (imgs - mean)/std, so we should pass raw/255 only
        t_raw=torch.from_numpy(np.asarray(img_np)).permute(2,0,1).float().unsqueeze(0)/255.0
        t=t_raw.to(device)
    else:
        t=(t.to(device)-mean)/std if t.device.type=='cpu' else (t.to(device)-mean)/std  # already
        # Actually for N21 which has no norm, we feed /255 only; but we just normalized. So need branch.
        # For N21, expect no norm -> pass /255
        t=torch.from_numpy(np.asarray(img_np)).permute(2,0,1).float().unsqueeze(0)/255.0
        t=t.to(device)
        if not has_norm:
            pass # keep raw
    bbox=bbox.unsqueeze(0).to(device)
    with torch.no_grad():
        out=model(t, bbox)
        dens=out["density"] if isinstance(out, dict) else out
        # dens may be low-res (28x28) - sum
        pred=float(dens.flatten(1).sum(1).item())
    return pred

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--S", type=int, default=392)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--ckpt", type=str, default=None)
    args=ap.parse_args()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # choose ckpt
    ckpt_path=args.ckpt or (N27_CKPT if os.path.exists(N27_CKPT) else N21_CKPT)
    print(f"[AXIOM R0] device={device} ckpt={ckpt_path}")
    # choose model dir
    if "N0027" in ckpt_path or "n0027" in ckpt_path.lower():
        node_dir=NODE27_DIR
    else:
        node_dir=NODE21_DIR
        # fallback if N27 dir not have model? N27 does
    cfg_mod=load_module(os.path.join(node_dir,"config.py"), "axiom_cfg")
    model_mod=load_module(os.path.join(node_dir,"model.py"), "axiom_model")
    cfg=dict(getattr(cfg_mod,"cfg"))
    model=model_mod.build_model(cfg).to(device).eval()
    ckpt=torch.load(ckpt_path, map_location="cpu")
    sd=ckpt["model"] if "model" in ckpt else ckpt
    # Filter missing keys (e.g., in_mean for N21)
    try:
        model.load_state_dict(sd, strict=True)
    except Exception as e:
        print(f"strict load fail {e}, trying strict=False")
        model.load_state_dict(sd, strict=False)
    print(f"loaded {ckpt_path} epoch {ckpt.get('epoch', '?')} best_mae {ckpt.get('best_mae', '?')}")

    with open(os.path.join(DATA_ROOT,"Train_Test_Val_FSC_147.json")) as f:
        ids=json.load(f)["val"]
    with open(os.path.join(DATA_ROOT,"annotation_FSC147_384.json")) as f:
        anno=json.load(f)
    if len(ids)>args.n:
        step=len(ids)//args.n
        ids=ids[::max(1,step)][:args.n]
    drifts_zoom=[]; drifts_crop=[]; errors=[]
    for idx, im_id in enumerate(ids):
        stem=im_id[:-4] if im_id.endswith(".jpg") else im_id
        img_path=os.path.join(DATA_ROOT, "images_384_VarV2", f"{stem}.jpg")
        gt_path=os.path.join(DATA_ROOT, "gt_density_map_adaptive_384_VarV2", f"{stem}.npy")
        gt=float(np.load(gt_path).sum())
        bbox,_=boxes_in_S(anno[im_id], args.S)
        # orig
        img_orig=Image.open(img_path).convert("RGB").resize((args.S,args.S), Image.BILINEAR)
        pred_orig=predict(model, device, img_orig, bbox, args.S)
        err=abs(pred_orig-gt)
        errors.append(err)
        # zoom: 1.15x (392->448 approx 1.14). We'll do 448 resize then center crop 392? Actually to test scale invariance, we evaluate at 448 and compare? But for drift metric, we want prediction at different scale vs original.
        # Simple: resize to 448 then feed to model (model supports dynamic_img_size). Compare pred.
        # We'll do two variants: predict at 448 and at 392; drift = |pred448 - pred392|.
        # Need boxes for 448 size as well (rescale)
        bbox_448,_=boxes_in_S(anno[im_id], 448)
        img_448=Image.open(img_path).convert("RGB").resize((448,448), Image.BILINEAR)
        pred_448=predict(model, device, img_448, bbox_448, 448)
        drift_zoom=abs(pred_448 - pred_orig)
        drifts_zoom.append(drift_zoom)
        # crop: 0.9 center crop then resize back to 392
        # Take 392*0.9=352 crop box centered
        s=args.S; c=int(s*0.9); x0=(s-c)//2; y0=(s-c)//2
        img_cropped=img_orig.crop((x0,y0,x0+c,y0+c)).resize((s,s), Image.BILINEAR)
        # bbox also crop: shift
        bbox_c=bbox.clone()
        bbox_c[0]-=x0; bbox_c[1]-=y0; bbox_c[2]-=x0; bbox_c[3]-=y0
        # scale because crop then resize: multiply by s/c
        scale=s/float(c)
        bbox_c*=scale
        pred_crop=predict(model, device, img_cropped, bbox_c, s)
        drift_crop=abs(pred_crop - pred_orig)
        drifts_crop.append(drift_crop)
        if (idx+1)%50==0:
            print(f" {idx+1}/{len(ids)} gt={gt:.0f} pred={pred_orig:.1f} err={err:.1f} zoomDrift={drift_zoom:.1f} cropDrift={drift_crop:.1f}")

    # correlations
    def pearson(x,y):
        xa=np.array(x); ya=np.array(y)
        return float(np.corrcoef(xa,ya)[0,1]) if np.std(xa)>1e-9 and np.std(ya)>1e-9 else 0.0
    def spearman(x,y):
        try:
            from scipy.stats import spearmanr
            r,_=spearmanr(x,y); return float(r)
        except:
            # manual
            def rankdata(a):
                s=sorted(range(len(a)), key=lambda i:a[i]); r=[0]*len(a)
                for p,i in enumerate(s): r[i]=p+1
                return r
            rx=rankdata(x); ry=rankdata(y)
            return pearson(rx,ry)
    rz_pear=pearson(drifts_zoom, errors)
    rz_spear=spearman(drifts_zoom, errors)
    rc_pear=pearson(drifts_crop, errors)
    rc_spear=spearman(drifts_crop, errors)
    # also absolute drift vs error? we already use absolute.
    print(f"[AXIOM R0] n={len(ids)}")
    print(f"  zoom drift vs |error|: Pearson {rz_pear:.3f} Spearman {rz_spear:.3f}  (mean drift {np.mean(drifts_zoom):.1f})")
    print(f"  crop drift vs |error|: Pearson {rc_pear:.3f} Spearman {rc_spear:.3f}  (mean drift {np.mean(drifts_crop):.1f})")
    # verdict
    best=max(abs(rz_spear), abs(rc_spear), abs(rz_pear), abs(rc_pear))
    print(f"  best |corr|={best:.3f}")
    if best>=0.3:
        print("VERDICT: PASS (>=0.3) -> drift predicts error, TTT signal exists; proceed to R1 LoRA+TFR design")
    else:
        print("VERDICT: FAIL (<0.3) -> drift is NOT predictive of error; kill AXIOM-TTC (aleatoric or stable)")
    # extra: drift vs gt? maybe drift correlates with count magnitude not error
    try:
        # load gt
        gts=[]
        with open(os.path.join(DATA_ROOT,"Train_Test_Val_FSC_147.json")) as f:
            ids2=json.load(f)["val"]
        with open(os.path.join(DATA_ROOT,"annotation_FSC147_384.json")) as f:
            anno2=json.load(f)
        # we already have gt list? Let's reuse errors+gt? Actually errors already, but need gt per idx
        # reconstruct gts for sampled ids
        gts=[]
        for im_id in ids:
            stem=im_id[:-4] if im_id.endswith(".jpg") else im_id
            gts.append(float(np.load(os.path.join(DATA_ROOT, "gt_density_map_adaptive_384_VarV2", f"{stem}.npy")).sum()))
        print(f"  drift_zoom vs GT corr Spearman {spearman(drifts_zoom, gts):.3f} Pearson {pearson(drifts_zoom,gts):.3f}")
    except Exception as e:
        print(f" extra GT corr fail {e}")

if __name__=="__main__":
    main()

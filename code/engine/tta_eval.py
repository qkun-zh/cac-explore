"""TTA evaluation for FSC147 crowd counting.
Usage: python code/engine/tta_eval.py --run_dir /data/runs/N0036_gca_ddca
"""
import os, sys, math, argparse, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_model(run_dir, node_dir, device):
    ckpt = torch.load(os.path.join(run_dir, "best.pth"), map_location=device, weights_only=False)
    import importlib.util
    spec = importlib.util.spec_from_file_location("model_mod", os.path.join(node_dir, "model.py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    cfg_spec = importlib.util.spec_from_file_location("cfg_mod", os.path.join(node_dir, "config.py"))
    cfg_mod = importlib.util.module_from_spec(cfg_spec); cfg_spec.loader.exec_module(cfg_mod)
    model = mod.build_model(cfg_mod.cfg)
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)
    return model, cfg_mod.cfg

def make_loader(cfg, root="/data/dataset/FSC147"):
    from torch.utils.data import DataLoader
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
    from fsc147 import FSC147Density as DS, collate_density as collate
    va = DS(root, int(cfg.get("input_size", 384)), "val")
    return DataLoader(va, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate, pin_memory=True)

def pad_to(imgs, target_size):
    """Pad images to target_size (center crop if larger, pad if smaller)."""
    B, C, H, W = imgs.shape
    T = target_size
    if H > T or W > T:
        # center crop
        h0 = (H - T) // 2; w0 = (W - T) // 2
        imgs = imgs[:, :, h0:h0+T, w0:w0+T]
    elif H < T or W < T:
        # pad with zeros
        pad_h = T - H; pad_w = T - W
        imgs = F.pad(imgs, (pad_w//2, pad_w - pad_w//2, pad_h//2, pad_h - pad_h//2))
    return imgs

def scale_bbox(bbox, img_h, img_w, target_size):
    """Scale bboxes so that the image content maps to target_size."""
    # After pad_to, the image is target_size x target_size
    # We need to map original coords to padded coords
    # Actually, pad_to center-crops or pads. Let's compute the offset.
    if img_h > target_size or img_w > target_size:
        h0 = (img_h - target_size) // 2; w0 = (img_w - target_size) // 2
        bbox = bbox.clone()
        bbox[..., 0] = bbox[..., 0] - w0
        bbox[..., 2] = bbox[..., 2] - w0
        bbox[..., 1] = bbox[..., 1] - h0
        bbox[..., 3] = bbox[..., 3] - h0
    # clamp
    bbox = bbox.clamp(min=0, max=target_size)
    return bbox

@torch.no_grad()
def tta_eval(model, loader, device, scales=(0.8, 1.0, 1.2), use_flips=True):
    model.eval()
    S = 384  # model expects this size
    mae = mse = n = 0
    for b in loader:
        imgs, bbox, gt = b["imgs"].to(device), b["bboxes"].to(device), b["counts"]
        b3 = b.get("bboxes3")
        if b3 is not None: b3 = b3.to(device)
        B, C, H, W = imgs.shape
        preds = []
        for s in scales:
            # resize then pad to S
            sh = max(32, int(H * s))
            sw = max(32, int(W * s))
            img_s = F.interpolate(imgs, (sh, sw), mode="bilinear", align_corners=False)
            img_s = pad_to(img_s, S)
            # scale and pad bboxes
            rh, rw = sh / H, sw / W
            bbox_s = bbox.clone()
            bbox_s[..., 0] *= rw; bbox_s[..., 2] *= rw
            bbox_s[..., 1] *= rh; bbox_s[..., 3] *= rh
            bbox_s = scale_bbox(bbox_s, sh, sw, S)
            b3_s = None
            if b3 is not None:
                b3_s = b3.clone()
                b3_s[..., 0] *= rw; b3_s[..., 2] *= rw
                b3_s[..., 1] *= rh; b3_s[..., 3] *= rh
                b3_s = scale_bbox(b3_s, sh, sw, S)
            out = model(img_s, bbox_s, b3_s)
            pred = out["density"].flatten(1).sum(1)
            preds.append(pred)
            if use_flips:
                img_f = torch.flip(img_s, [-1])
                bbox_f = bbox_s.clone()
                x0, x2 = bbox_f[..., 0].clone(), bbox_f[..., 2].clone()
                bbox_f[..., 0] = S - x2; bbox_f[..., 2] = S - x0
                b3_f = None
                if b3_s is not None:
                    b3_f = b3_s.clone()
                    x0_3, x2_3 = b3_f[..., 0].clone(), b3_f[..., 2].clone()
                    b3_f[..., 0] = S - x2_3; b3_f[..., 2] = S - x0_3
                out_f = model(img_f, bbox_f, b3_f)
                preds.append(out_f["density"].flatten(1).sum(1))
        pred_avg = torch.stack(preds).mean(0).cpu()
        mae += (pred_avg - gt).abs().sum().item()
        mse += ((pred_avg - gt) ** 2).sum().item()
        n += gt.numel()
    return mae / max(n, 1), math.sqrt(mse / max(n, 1))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--node_dir", default=None)
    p.add_argument("--scales", default="0.8,1.0,1.2")
    p.add_argument("--flips", action="store_true", default=True)
    p.add_argument("--no-flips", dest="flips", action="store_false")
    a = p.parse_args()
    if a.node_dir is None:
        a.node_dir = a.run_dir.replace("/data/runs/", "/data/repo/tree/nodes/").replace("runs/", "tree/nodes/")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scales = tuple(float(x) for x in a.scales.split(","))
    model, cfg = load_model(a.run_dir, a.node_dir, device)
    loader = make_loader(cfg)
    mae0, rmse0 = tta_eval(model, loader, device, scales=(1.0,), use_flips=False)
    print(f"[baseline] MAE={mae0:.3f} RMSE={rmse0:.3f}")
    mae, rmse = tta_eval(model, loader, device, scales=scales, use_flips=a.flips)
    print(f"[TTA] scales={scales} flips={a.flips} MAE={mae:.3f} RMSE={rmse:.3f}")

if __name__ == "__main__":
    main()

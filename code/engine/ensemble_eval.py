"""Ensemble evaluation: average predictions from multiple checkpoints."""
import os, sys, math, argparse, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_model(node_dir, device):
    import importlib.util
    spec = importlib.util.spec_from_file_location("model_mod", os.path.join(node_dir, "model.py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    cfg_spec = importlib.util.spec_from_file_location("cfg_mod", os.path.join(node_dir, "config.py"))
    cfg_mod = importlib.util.module_from_spec(cfg_spec); cfg_spec.loader.exec_module(cfg_mod)
    return mod, cfg_mod.cfg

def make_loader(cfg, root="/data/dataset/FSC147"):
    from torch.utils.data import DataLoader
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
    from fsc147 import FSC147Density as DS, collate_density as collate
    va = DS(root, int(cfg.get("input_size", 384)), "val")
    return DataLoader(va, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate, pin_memory=True)

@torch.no_grad()
def eval_ensemble(models, loader, device):
    for m in models:
        m.eval()
    mae = mse = n = 0
    for b in loader:
        imgs, bbox, gt = b["imgs"].to(device), b["bboxes"].to(device), b["counts"]
        b3 = b.get("bboxes3")
        if b3 is not None: b3 = b3.to(device)
        preds = []
        for m in models:
            out = m(imgs, bbox, b3)
            pred = out["density"].flatten(1).sum(1).cpu()
            preds.append(pred)
        pred_avg = torch.stack(preds).mean(0)
        mae += (pred_avg - gt).abs().sum().item()
        mse += ((pred_avg - gt) ** 2).sum().item()
        n += gt.numel()
    return mae / max(n, 1), math.sqrt(mse / max(n, 1))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--node_dir", default=None)
    p.add_argument("--checkpoints", default=None, help="comma-separated epoch numbers, e.g. 10,15,20,25,best")
    a = p.parse_args()
    if a.node_dir is None:
        a.node_dir = a.run_dir.replace("/data/runs/", "/data/repo/tree/nodes/").replace("runs/", "tree/nodes/")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mod, cfg = load_model(a.node_dir, device)
    loader = make_loader(cfg)
    ckpts = a.checkpoints.split(",") if a.checkpoints else ["best"]
    models = []
    for c in ckpts:
        if c == "best":
            path = os.path.join(a.run_dir, "best.pth")
        else:
            path = os.path.join(a.run_dir, f"ep{c.zfill(3)}.pth")
        if not os.path.exists(path):
            print(f"SKIP {path} (not found)")
            continue
        ckpt = torch.load(path, map_location=device, weights_only=False)
        m = mod.build_model(cfg)
        m.load_state_dict(ckpt["model"])
        m.eval().to(device)
        models.append(m)
        print(f"Loaded {path} (epoch={ckpt.get('epoch','?')})")
    print(f"\nEnsemble {len(models)} models")
    mae, rmse = eval_ensemble(models, loader, device)
    print(f"Ensemble MAE={mae:.3f} RMSE={rmse:.3f}")

if __name__ == "__main__":
    main()

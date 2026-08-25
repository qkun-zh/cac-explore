"""HF GOD training — uses HF datasets + HF processor + HF AutoModel + HF AdamW, only GOD loss.
Example usage per user spec:
  hf auth login
  ds = load_dataset("isentropic/FSC147")
  processor = AutoImageProcessor.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m")
  model = AutoModel.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m")
Here wrapped into build_model for engine compat.
"""
import os, sys
sys.path.insert(0, "/data/repo")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if "__file__" in globals() else "/data/repo")
import json
import math
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# HF stack imports per spec
HF_MODEL = "facebook/dinov3-vits16-pretrain-lvd1689m"
HF_DATASET = "isentropic/FSC147"

def get_hf_objects(token=None, size=384):
    from transformers import AutoImageProcessor, AutoModel
    # token handling
    if token is None:
        for p in ["/tmp/hf_token.txt", "/root/.cache/huggingface/token"]:
            if os.path.exists(p):
                token = open(p).read().strip()
                break
    os.environ.setdefault("HF_HOME", "/data/asset/hf")
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    processor = AutoImageProcessor.from_pretrained(HF_MODEL, token=token, trust_remote_code=True, size={"height": size, "width": size})
    # ensure 384
    # processor.size is SizeDict, override
    try:
        processor.size = {"height": size, "width": size}
    except: pass
    return processor

class FSC147HF(Dataset):
    """Real HF-stack dataset: load_dataset(isentropic/FSC147, refs/convert/parquet) +
    Train_Test_Val_FSC_147.json / annotation_FSC147_384.json via hf_hub_download(repo_type='dataset').
    Rows carry 'path' (e.g. 1050.jpg) so ids join with annotations. Processor does resize+norm in collate."""
    def __init__(self, split="train", processor=None, size=384):
        self.S = size
        self.processor = processor
        token = None
        for p in ["/tmp/hf_token.txt", "/root/.cache/huggingface/token"]:
            if os.path.exists(p):
                token = open(p).read().strip(); break
        os.environ.setdefault("HF_HOME", "/data/asset/hf")
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        from datasets import load_dataset
        from datasets import Image as HFImage
        from huggingface_hub import hf_hub_download
        full = load_dataset(HF_DATASET, revision="refs/convert/parquet", token=token)["train"]
        raw = full.cast_column("image", HFImage(decode=False))
        paths = [os.path.basename(raw[i]["image"]["path"]) for i in range(len(raw))]  # metadata-only pass
        p_splits = hf_hub_download(HF_DATASET, "Train_Test_Val_FSC_147.json", repo_type="dataset", token=token)
        p_anno = hf_hub_download(HF_DATASET, "annotation_FSC147_384.json", repo_type="dataset", token=token)
        split_ids = set(json.load(open(p_splits))[split])
        self.anno = json.load(open(p_anno))
        self.rows = [i for i, pth in enumerate(paths) if pth in split_ids]
        self.ids = [paths[i] for i in self.rows]
        self.ds = full
        print(f"[FSC147HF] split={split} rows={len(self.rows)} (from {len(paths)} total)", flush=True)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        im_id = self.ids[i]              # e.g. 1050.jpg
        stem = im_id[:-4]
        img = self.ds[row]["image"]      # decoded PIL, variable aspect
        S = self.S
        ann = self.anno[im_id]
        sx, sy = S / float(ann["W"]), S / float(ann["H"])
        bboxes3 = []
        for corners in ann["box_examples_coordinates"][:3]:
            xs = [p[0] for p in corners]; ys = [p[1] for p in corners]
            bboxes3.append([min(xs)*sx, min(ys)*sy, max(xs)*sx, max(ys)*sy])
        while len(bboxes3) < 3:
            bboxes3.append(bboxes3[-1])
        bboxes3 = torch.tensor(bboxes3, dtype=torch.float32)
        pts = torch.tensor([[p[0]*sx, p[1]*sy] for p in ann["points"]], dtype=torch.float32)
        return {"image": img, "bboxes3": bboxes3, "points": pts, "id": stem}

def collate_hf(batch, processor):
    images = [b["image"] for b in batch]
    # processor handles resize to 384, rescale, normalize
    inputs = processor(images=images, return_tensors="pt")
    pixel_values = inputs["pixel_values"]  # [B,3,384,384] already normalized
    bboxes3 = torch.stack([b["bboxes3"] for b in batch])  # [B,3,4]
    points = [b["points"] for b in batch]  # list [N,2]
    return {"pixel_values": pixel_values, "bboxes3": bboxes3, "points": points}

BUCKETS = [(0, 25), (25, 75), (75, 200), (200, 500), (500, float("inf"))]

def _rankdata(x):
    """Average-rank transform (ties -> mean rank), torch only."""
    order = x.argsort()
    ranks = torch.empty_like(x)
    sx = x[order]
    i = 0
    r = torch.arange(1, x.numel() + 1, dtype=x.dtype)
    while i < x.numel():
        j = i
        while j + 1 < x.numel() and sx[j + 1] == sx[i]:
            j += 1
        avg = r[i:j + 1].mean()
        ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks

def _spearman(a, b):
    ra, rb = _rankdata(a), _rankdata(b)
    ra = ra - ra.mean(); rb = rb - rb.mean()
    denom = (ra.norm() * rb.norm()).clamp_min(1e-12)
    return (ra * rb).sum() / denom

def train_one_epoch(model, loader, optimizer, device, cfg, ep=0, log_every=50):
    model.train()
    total_loss = 0
    nb = 0
    import time as _t
    t0 = _t.time()
    for batch in loader:
        pixel_values = batch["pixel_values"].to(device)
        bboxes3 = batch["bboxes3"].to(device)
        points = [p.to(device) for p in batch["points"]]  # move GT points to device
        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=cfg.get("amp", True)):
            out = model(pixel_values, bboxes3=bboxes3, points=points)
            loss = out["loss"]
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item(); nb += 1
        if nb % log_every == 0:
            w = out["w"].detach().float()
            ws = w.sum(dim=1)
            g = out["gate"].detach().float()
            met = out["metrics"]
            lr_now = optimizer.param_groups[0]["lr"]
            # detailed sub-losses: solver-aware (sinkhorn: trans/klr/klc/resr/resc; router: def/over/sur)
            print(f"ep{ep} it{nb}/{len(loader)} loss={loss.item():.3f} "
                  f"[lot={met['lot']:.3f} (trans={met.get('trans',0):.3f} klr={met.get('klr',0):.3f} klc={met.get('klc',0):.3f} "
                  f"resr={met.get('resr',0):.2f} resc={met.get('resc',0):.2f}) "
                  f"rep={met['rep']:.4f} cnt_err={met['cnt_err']:.2f}] "
                  f"w_sum(mean/p10/p90)={ws.mean().item():.1f}/{torch.quantile(ws,0.1).item():.1f}/{torch.quantile(ws,0.9).item():.1f} "
                  f"w(min/max)={w.min().item():.3f}/{w.max().item():.3f} "
                  f"gate_mean={g.mean().item():.3f} gate(p10/p90)={torch.quantile(g.flatten(),0.1).item():.3f}/{torch.quantile(g.flatten(),0.9).item():.3f} "
                  f"a={model.prompt.alpha.item():+.3f} b={model.prompt.beta.item():+.3f} "
                  f"gnorm={gn.item():.2f} lr={lr_now:.1e} [{_t.time()-t0:.0f}s]", flush=True)
    return total_loss / max(len(loader),1)

@torch.no_grad()
def evaluate(model, loader, device, ep=0):
    model.eval()
    gts, preds, preds_open = [], [], []   # closed-book Σw / open-book transported
    gates, wsums = [], []
    for batch in loader:
        pixel_values = batch["pixel_values"].to(device)
        bboxes3 = batch["bboxes3"].to(device)
        points = [p.to(device) for p in batch["points"]]
        out = model(pixel_values, bboxes3=bboxes3, points=points)
        pred_open = out["pred_counts"].float().cpu()      # uses GT pits (upper bound)
        pred_closed = out["counts_sumw"].float().cpu()    # GT-free: Σw (deployable)
        gt = torch.tensor([p.shape[0] for p in points], dtype=torch.float32)
        gts.append(gt); preds.append(pred_closed); preds_open.append(pred_open)
        gates.append(out["gate"].float().mean().item())
        wsums += out["w"].float().sum(dim=1).cpu().tolist()
    gt = torch.cat(gts); pred = torch.cat(preds); pred_o = torch.cat(preds_open)
    err = pred - gt; err_o = pred_o - gt
    mae = err.abs().mean().item(); rmse = err.pow(2).mean().sqrt().item()
    mae_o = err_o.abs().mean().item(); rmse_o = err_o.pow(2).mean().sqrt().item()
    bias = err.mean().item()
    rho = _spearman(gt, pred).item(); rho_o = _spearman(gt, pred_o).item()
    ws = torch.tensor(wsums)
    print(f"== Ep{ep:02d} VAL n={gt.numel()} "
          f"[closed-book Σw] MAE={mae:.3f} RMSE={rmse:.3f} ρ={rho:.3f} | "
          f"[open-book OT] MAE={mae_o:.3f} RMSE={rmse_o:.3f} ρ={rho_o:.3f} | gap={mae_o-mae:+.3f}", flush=True)
    print(f"   overall bias(median signed)={bias:+.2f}/{err.median().item():+.2f} "
          f"w_sum(mean/p10/p90)={ws.mean().item():.1f}/{torch.quantile(ws,0.1).item():.1f}/{torch.quantile(ws,0.9).item():.1f} "
          f"gate_mean={(sum(gates)/len(gates)):.3f} a={model.prompt.alpha.item():+.3f} b={model.prompt.beta.item():+.3f}", flush=True)
    for lo, hi in BUCKETS:
        m = (gt >= lo) & (gt < hi)
        if m.sum() == 0:
            continue
        bm = err[m]; bo = err_o[m]
        hi_lab = "inf" if hi == float("inf") else str(int(hi))
        print(f"   bucket[{int(lo):>3},{hi_lab:>4}) n={int(m.sum()):4d} "
              f"cMAE={bm.abs().mean().item():8.2f}(med{bm.median().item():+8.1f}) "
              f"oMAE={bo.abs().mean().item():8.2f}(med{bo.median().item():+8.1f}) "
              f"under%={(bo < 0).float().mean().item()*100:5.1f}", flush=True)
    hist = {"ep": ep, "mae": mae, "rmse": rmse, "mae_open": mae_o, "bias": bias,
            "spearman": rho, "spearman_open": rho_o,
            "gate": sum(gates)/len(gates),
            "alpha": model.prompt.alpha.item(), "beta": model.prompt.beta.item()}
    with open("/tmp/god_hist.jsonl", "a") as f:
        f.write(json.dumps(hist) + "\n")
    return mae, rmse

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--size", type=int, default=384)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}")

    # HF processor
    processor = get_hf_objects(size=args.size)
    print(f"processor size {processor.size}")

    # datasets per spec: real load_dataset path (parquet branch, cached)
    print("HF dataset: load_dataset(isentropic/FSC147, refs/convert/parquet)", flush=True)
    train_set = FSC147HF(split="train", processor=processor, size=args.size)
    val_set = FSC147HF(split="val", processor=processor, size=args.size)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2, collate_fn=lambda b: collate_hf(b, processor))
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=lambda b: collate_hf(b, processor))
    print(f"train {len(train_set)} val {len(val_set)}")

    # model via HF AutoModel
    from tree.nodes.G001_god_hf.model import build_model
    cfg = dict(input_size=args.size, patch_size=16, hf_model=HF_MODEL, god_alpha=1.0, god_beta=0.5, god_gamma=0.1, god_epsilon=0.05, god_lambda=1e-3)
    model = build_model(cfg).to(device)
    print(f"model params {sum(p.numel() for p in model.parameters())/1e6:.2f}M, trainable {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M")

    # torch AdamW (transformers 5.x no longer ships its own)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.05, betas=(0.9, 0.999), eps=1e-8)
    print(f"optimizer torch AdamW lr={args.lr}")

    best = float("inf")
    for ep in range(1, args.epochs+1):
        loss = train_one_epoch(model, train_loader, optimizer, device, cfg, ep=ep)
        mae, rmse = evaluate(model, val_loader, device, ep=ep)
        print(f"Ep {ep:02d} loss={loss:.4f} val MAE={mae:.3f} RMSE={rmse:.3f} best={best:.3f}")
        if mae < best:
            best = mae
            torch.save(model.state_dict(), f"/tmp/god_hf_best.pth")
            print(f"  *** best saved")

if __name__ == "__main__":
    main()

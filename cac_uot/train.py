"""cac_uot entry — HF-only, standard UOT as loss (K-step Sinkhorn)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor
from cac_uot.configs.uot_config import UOTConfig
from cac_uot.models.counter import UOTCounter
from cac_uot.data.fsc147 import FSC147HF, collate_hf
from cac_uot.training.trainer import train_one_epoch, evaluate

def main():
    cfg = UOTConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device} cfg {cfg}")
    # HF processor
    tok = None
    for p in ["/tmp/hf_token.txt", "/root/.cache/huggingface/token"]:
        if os.path.exists(p): token=open(p).read().strip(); tok=token; break
    os.environ.setdefault("HF_HOME","/data/asset/hf"); os.environ.setdefault("HF_ENDPOINT","https://hf-mirror.com")
    processor = AutoImageProcessor.from_pretrained(cfg.hf_processor, token=tok, trust_remote_code=True, size={"height":cfg.image_size,"width":cfg.image_size})
    # datasets (HF)
    train_set = FSC147HF(split="train", processor=processor, size=cfg.image_size)
    val_set   = FSC147HF(split="val", processor=processor, size=cfg.image_size)
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=2, collate_fn=lambda b: collate_hf(b, processor))
    val_loader   = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False, num_workers=2, collate_fn=lambda b: collate_hf(b, processor))
    # model
    model = UOTCounter(cfg).to(device)
    print(f"params {sum(p.numel() for p in model.parameters())/1e6:.2f}M trainable {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr, weight_decay=cfg.weight_decay)
    best=float("inf")
    for ep in range(1, cfg.epochs+1):
        loss = train_one_epoch(model, train_loader, opt, device, cfg, ep)
        mae_c, mae_o = evaluate(model, val_loader, device, ep)
        print(f"Ep{ep:02d} loss={loss:.3f} closed={mae_c:.2f} open={mae_o:.2f} best={best:.2f}")
        if mae_c < best:
            best=mae_c; torch.save(model.state_dict(), "/tmp/cac_uot_best.pth")
            print(" *** best")

if __name__ == "__main__":
    main()

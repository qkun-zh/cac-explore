import os, torch
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor
from configs.config import Config
from models.model import Counter
from datasets.dataset import FSC147, collate
def main():
    cfg=Config()
    tok=open("/tmp/hf_token.txt").read().strip() if os.path.exists("/tmp/hf_token.txt") else None
    proc=AutoImageProcessor.from_pretrained(cfg.hf_model, token=tok, trust_remote_code=True, size={"height":cfg.image_size,"width":cfg.image_size})
    train=FSC147("train",None,cfg.image_size); val=FSC147("val",None,cfg.image_size)
    tr=DataLoader(train,batch_size=cfg.batch_size,shuffle=True,num_workers=2,collate_fn=lambda b: collate(b,proc))
    va=DataLoader(val,batch_size=cfg.batch_size,shuffle=False,num_workers=2,collate_fn=lambda b: collate(b,proc))
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=Counter(cfg).to(device)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=cfg.lr,weight_decay=cfg.weight_decay)
    best=float("inf")
    for ep in range(1,cfg.epochs+1):
        model.train(); tot=0
        for batch in tr:
            pv=batch["pixel_values"].to(device); b=batch["bboxes"].to(device); pts=[p.to(device) for p in batch["points"]]
            opt.zero_grad()
            with torch.autocast("cuda", torch.float16, enabled=cfg.amp):
                out=model(pv,b,pts); loss=out["loss"]
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); tot+=loss.item()
        print(f"Ep{ep} loss={tot/len(tr):.3f}")
        # val closed is density sum
        model.eval(); err=0; n=0
        with torch.no_grad():
            for batch in va:
                pv=batch["pixel_values"].to(device); b=batch["bboxes"].to(device); pts=batch["points"]
                out=model(pv,b); pred=out["pred_counts"].cpu(); gt=torch.tensor([len(p) for p in pts],dtype=torch.float32)
                err+=(pred-gt).abs().sum().item(); n+=len(gt)
        mae=err/n; print(f" val MAE={mae:.2f} best={best:.2f}")
        if mae<best: best=mae; torch.save(model.state_dict(),"/tmp/cac_d_best.pth"); print(" best")
if __name__=="__main__": main()

import os, json, torch
from torch.utils.data import Dataset
from PIL import Image
class FSC147(Dataset):
    def __init__(self, split, processor, size=384):
        from datasets import load_dataset
        from huggingface_hub import hf_hub_download
        tok=open("/tmp/hf_token.txt").read().strip() if os.path.exists("/tmp/hf_token.txt") else None
        ds=__import__("datasets").load_dataset("isentropic/FSC147", revision="refs/convert/parquet", token=tok)["train"]
        raw=ds.cast_column("image", __import__("datasets").Image(decode=False))
        paths=[os.path.basename(raw[i]["image"]["path"]) for i in range(len(raw))]
        pj=hf_hub_download("isentropic/FSC147","Train_Test_Val_FSC_147.json",repo_type="dataset",token=tok)
        pa=hf_hub_download("isentropic/FSC147","annotation_FSC147_384.json",repo_type="dataset",token=tok)
        ids=set(json.load(open(pj))[split]); self.anno=json.load(open(pa))
        self.rows=[i for i,p in enumerate(paths) if p in ids]; self.ids=[paths[i] for i in self.rows]; self.ds=__import__("datasets").load_dataset("isentropic/FSC147", revision="refs/convert/parquet", token=tok)["train"]
        self.processor=processor; self.S=size
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]; im_id=self.ids[i]
        img=self.ds[r]["image"]
        ann=self.anno[im_id]; S=self.S
        sx,sy=S/float(ann["W"]),S/float(ann["H"])
        bboxes=torch.tensor([[min(p[0] for p in c)*sx, min(p[1] for p in c)*sy, max(p[0] for p in c)*sx, max(p[1] for p in c)*sy] for c in ann["box_examples_coordinates"][:3]],dtype=torch.float32)
        pts=torch.tensor([[p[0]*sx,p[1]*sy] for p in ann["points"]],dtype=torch.float32)
        return {"image":img,"bboxes":bboxes,"points":pts}
def collate(batch, proc):
    imgs=[b["image"] for b in batch]
    pix=proc(images=imgs, return_tensors="pt")["pixel_values"]
    return {"pixel_values":pix, "bboxes":torch.stack([b["bboxes"] for b in batch]), "points":[b["points"] for b in batch]}

"""FSC147 via HF datasets (parquet branch) + processor. Self-contained, no local mirror."""
import os, json, torch
from torch.utils.data import Dataset
from PIL import Image

HF_DATASET = "isentropic/FSC147"

class FSC147HF(Dataset):
    """HF-only FSC147. Rows carry path (e.g. 1050.jpg) for id join with annotations."""
    def __init__(self, split="train", processor=None, size=384):
        self.S = size
        self.processor = processor
        token = None
        for p in ["/tmp/hf_token.txt", "/root/.cache/huggingface/token"]:
            if os.path.exists(p):
                token = open(p).read().strip(); break
        os.environ.setdefault("HF_HOME", "/data/asset/hf")
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        from datasets import load_dataset, Image as HFImage
        from huggingface_hub import hf_hub_download
        full = load_dataset(HF_DATASET, revision="refs/convert/parquet", token=token)["train"]
        raw = full.cast_column("image", HFImage(decode=False))
        paths = [os.path.basename(raw[i]["image"]["path"]) for i in range(len(raw))]
        p_splits = hf_hub_download(HF_DATASET, "Train_Test_Val_FSC_147.json", repo_type="dataset", token=token)
        p_anno = hf_hub_download(HF_DATASET, "annotation_FSC147_384.json", repo_type="dataset", token=token)
        split_ids = set(json.load(open(p_splits))[split])
        self.anno = json.load(open(p_anno))
        self.rows = [i for i, pth in enumerate(paths) if pth in split_ids]
        self.ids = [paths[i] for i in self.rows]
        self.ds = full
        print(f"[FSC147HF] split={split} rows={len(self.rows)} (from {len(paths)} total)")

    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        row = self.rows[i]
        im_id = self.ids[i]
        img = self.ds[row]["image"]
        ann = self.anno[im_id]
        S = self.S
        sx, sy = S / float(ann["W"]), S / float(ann["H"])
        bboxes = []
        for corners in ann["box_examples_coordinates"][:3]:
            xs=[p[0] for p in corners]; ys=[p[1] for p in corners]
            bboxes.append([min(xs)*sx, min(ys)*sy, max(xs)*sx, max(ys)*sy])
        while len(bboxes)<3: bboxes.append(bboxes[-1])
        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        points = torch.tensor([[p[0]*sx, p[1]*sy] for p in ann["points"]], dtype=torch.float32)
        return {"image": img, "bboxes3": bboxes, "points": points, "id": im_id[:-4]}

def collate_hf(batch, processor):
    images = [b["image"] for b in batch]
    inputs = processor(images=images, return_tensors="pt")
    return {"pixel_values": inputs["pixel_values"],
            "bboxes3": torch.stack([b["bboxes3"] for b in batch]),
            "points": [b["points"] for b in batch]}

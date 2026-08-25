import json, os
import torch
from torch.utils.data import Dataset
from datasets import load_dataset, Image          # HF datasets
from huggingface_hub import hf_hub_download       # HF hub (annotation files)

REPO = "isentropic/FSC147"
REV = "refs/convert/parquet"
SPLIT_FILE = "Train_Test_Val_FSC_147.json"
ANNO_FILE = "annotation_FSC147_384.json"
_cache = {}

def _load_images(token):
    """Single shared HF load: decoded image dataset + basename path list."""
    if "img" not in _cache:
        ds = load_dataset(REPO, revision=REV, token=token)["train"]
        raw = ds.cast_column("image", Image(decode=False))
        _cache["img"] = ds
        _cache["paths"] = [os.path.basename(r["path"]) for r in raw.data["image"].to_pylist()]
    return _cache["img"], _cache["paths"]

def _load_json(fname, token):
    if fname not in _cache:
        _cache[fname] = json.load(open(hf_hub_download(REPO, fname, repo_type="dataset", token=token)))
    return _cache[fname]

class FSC147(Dataset):
    """FSC-147 via HF; exemplar boxes/points pre-scaled to size x size at init."""
    def __init__(self, split, size=384):
        from cac_d.common import hf_token
        tok = hf_token()
        img, paths = _load_images(tok)
        anno = _load_json(ANNO_FILE, tok)
        ids = set(_load_json(SPLIT_FILE, tok)[split])
        self.size = size
        self.index = []
        for row, im_id in enumerate(paths):
            if im_id not in ids: continue
            ann = anno[im_id]
            sx, sy = size / float(ann["W"]), size / float(ann["H"])
            boxes = torch.tensor(
                [[min(x for x, _ in c) * sx, min(y for _, y in c) * sy,
                  max(x for x, _ in c) * sx, max(y for _, y in c) * sy]
                 for c in ann["box_examples_coordinates"][:3]], dtype=torch.float32).reshape(-1, 4)
            pts = torch.tensor([[x * sx, y * sy] for x, y in ann["points"]],
                               dtype=torch.float32).reshape(-1, 2)
            self.index.append((row, boxes, pts))

    def __len__(self): return len(self.index)

    def __getitem__(self, i):
        row, boxes, pts = self.index[i]
        return {"image": _cache["img"][row]["image"], "bboxes": boxes, "points": pts}

def collate(batch, proc):
    pix = proc(images=[b["image"] for b in batch], return_tensors="pt")["pixel_values"]
    return {"pixel_values": pix,
            "bboxes": torch.stack([b["bboxes"] for b in batch]),
            "points": [b["points"] for b in batch]}

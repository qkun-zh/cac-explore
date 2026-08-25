import json, os
import torch
from PIL import Image as PILImage
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter
from datasets import load_dataset, Image          # HF datasets
from huggingface_hub import hf_hub_download       # HF hub (annotation files)

REPO = "isentropic/FSC147"
REV = "refs/convert/parquet"
SPLIT_FILE = "Train_Test_Val_FSC_147.json"
ANNO_FILE = "annotation_FSC147_384.json"
_cache = {}

def _load_images(token):
    """Single shared HF load: decoded image dataset + basename path list.
    Paths are read from the parquet struct column directly (no byte copies)."""
    if "img" not in _cache:
        ds = load_dataset(REPO, revision=REV, token=token)["train"]
        raw = ds.cast_column("image", Image(decode=False)).data["image"]
        chunks = raw.chunks if hasattr(raw, "chunks") else [raw]
        _cache["paths"] = [os.path.basename(p) for c in chunks
                           for p in c.field("path").to_pylist()]
        _cache["img"] = ds
    return _cache["img"], _cache["paths"]

def _load_json(fname, token):
    if fname not in _cache:
        _cache[fname] = json.load(open(hf_hub_download(REPO, fname, repo_type="dataset", token=token)))
    return _cache[fname]

class FSC147(Dataset):
    """FSC-147 via HF; exemplar boxes/points pre-scaled to size x size at init.
    augment=True applies h-flip (coords mirrored identically -> no misalignment)
    and color jitter (label-invariant)."""
    def __init__(self, split, size=384, augment=False, flip_p=0.5, color_jitter=True):
        from cac_d.common import hf_token
        tok = hf_token()
        img, paths = _load_images(tok)
        anno = _load_json(ANNO_FILE, tok)
        ids = set(_load_json(SPLIT_FILE, tok)[split])
        self.size = size; self.augment = augment; self.flip_p = flip_p
        self.jitter = ColorJitter(0.4, 0.4, 0.4, 0.1) if (augment and color_jitter) else None
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
        img = _cache["img"][row]["image"]
        if self.augment and torch.rand(()) < self.flip_p:
            img = img.transpose(PILImage.Transpose.FLIP_LEFT_RIGHT)
            S = float(self.size)
            boxes = boxes.clone()
            boxes[:, [0, 2]] = torch.stack([S - boxes[:, 2], S - boxes[:, 0]], 1)
            pts = pts.clone(); pts[:, 0] = S - pts[:, 0]
        if self.jitter is not None:
            img = self.jitter(img)
        return {"image": img, "bboxes": boxes, "points": pts}

def collate(batch, proc):
    pix = proc(images=[b["image"] for b in batch], return_tensors="pt")["pixel_values"]
    return {"pixel_values": pix,
            "bboxes": torch.stack([b["bboxes"] for b in batch]),
            "points": [b["points"] for b in batch]}

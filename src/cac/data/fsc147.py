"""FSC147 (VarV2 protocol) class-agnostic counting dataset.

Server layout /data/dataset/FSC147/:
  images_384_VarV2/<id>.jpg                    # variable-aspect images, long side ~384
  gt_density_map_adaptive_384_VarV2/<id>.npy   # precomputed adaptive density maps (image-sized)
  annotation_FSC147_384.json                   # {"<id>.jpg": {"W","H","box_examples_coordinates"(orig coords)}}
  Train_Test_Val_FSC_147.json                  # {"train": [...], "val": [...], "test": [...]}

Output contract: imgs [3,S,S] / bboxes [4] in S-space / density [1,S,S] / counts scalar.
Density resizing is sum-preserving (count unchanged).
"""
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def _clamp_box(b: torch.Tensor, S: int, min_side: float = 2.0) -> torch.Tensor:
    """Clamp a box into [0,S]^2; degenerate boxes are re-expanded around their
    centroid so exemplar RoIs never collapse to zero-area patches."""
    x0, y0, x1, y1 = [float(v) for v in b]
    x0, x1 = sorted((min(max(x0, 0.0), float(S)), min(max(x1, 0.0), float(S))))
    y0, y1 = sorted((min(max(y0, 0.0), float(S)), min(max(y1, 0.0), float(S))))
    if x1 - x0 < min_side or y1 - y0 < min_side:
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        half = min_side / 2.0
        x0, x1 = max(0.0, cx - half), min(float(S), cx + half)
        y0, y1 = max(0.0, cy - half), min(float(S), cy + half)
    return torch.tensor([x0, y0, x1, y1], dtype=torch.float32)


def _augment_scale_flip(img, dens, bbox, bboxes3, S: int, count0: float):
    """Canonical train-time augmentation (protocol v2): random scale in
    [0.6, 1.25] with center placement back to S, then random horizontal flip.

    Draws come from the per-worker seeded RNG (deterministic at the fixed
    seed). Image and density are transformed jointly; boxes follow the same
    affine, so exemplar RoIs stay aligned with the augmented density. The
    count label is recomputed from the final density (any border-crop mass is
    removed, keeping loss and target physically consistent)."""
    if torch.rand(1).item() < 0.5:  # random horizontal flip
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        dens = dens.flip(-1)
        bbox = torch.tensor([S - bbox[2], bbox[1], S - bbox[0], bbox[3]])
        bboxes3 = torch.stack([torch.tensor([S - b[2], b[1], S - b[0], b[3]]) for b in bboxes3])

    s = 0.6 + 0.65 * torch.rand(1).item()  # random scale in [0.6, 1.25]
    sS = max(2, int(round(S * s)))
    scale = sS / float(S)  # true resize ratio (rounding-exact, not `s`)
    if sS != S:
        img = img.resize((sS, sS), Image.BILINEAR)
        dens = F.interpolate(dens[None, None], size=(sS, sS), mode="bilinear", align_corners=False)[0, 0]
        dens = dens * (count0 / dens.sum().clamp_min(1e-8))  # sum-conserving resize
        if sS >= S:  # center crop
            off = (sS - S) // 2
            img = img.crop((off, off, off + S, off + S))
            dens = dens[off:off + S, off:off + S]

            def place(c: float) -> float:
                return c - off
        else:  # center pad with zeros (black image / zero density)
            canvas = Image.new("RGB", (S, S), (0, 0, 0))
            canvas.paste(img, ((S - sS) // 2, (S - sS) // 2))
            img = canvas
            padded = torch.zeros(S, S, dtype=dens.dtype)
            padded[(S - sS) // 2:(S + sS) // 2, (S - sS) // 2:(S + sS) // 2] = dens
            dens = padded

            def place(c: float) -> float:
                return c + (S - sS) / 2.0

        bbox = _clamp_box(torch.tensor([place(b * scale) for b in bbox]), S)
        bboxes3 = torch.stack([_clamp_box(torch.tensor([place(b * scale) for b in b3]), S) for b3 in bboxes3])
    return img, dens, bbox, bboxes3


class FSC147Density(torch.utils.data.Dataset):
    def __init__(self, root, img_size=384, split="train", augment=False):
        self.root, self.size, self.split = root, int(img_size), split
        self.augment = augment and split == "train"
        with open(os.path.join(root, "Train_Test_Val_FSC_147.json")) as f:
            self.ids = list(json.load(f)[split])
        with open(os.path.join(root, "annotation_FSC147_384.json")) as f:
            self.anno = json.load(f)
        self.img_dir = os.path.join(root, "images_384_VarV2")
        self.den_dir = os.path.join(root, "gt_density_map_adaptive_384_VarV2")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        im_id = self.ids[i]
        stem = im_id[:-4] if im_id.endswith(".jpg") else im_id
        img = Image.open(os.path.join(self.img_dir, f"{stem}.jpg")).convert("RGB")
        dens = torch.from_numpy(np.load(os.path.join(self.den_dir, f"{stem}.npy"))).float()
        if dens.dim() == 3:
            dens = dens[0]
        count0 = float(dens.sum())

        S = self.size
        img = img.resize((S, S), Image.BILINEAR)
        dens = F.interpolate(dens[None, None], size=(S, S), mode="bilinear", align_corners=False)[0, 0]
        dens = dens * (count0 / dens.sum().clamp_min(1e-8))  # sum-conserving

        ann = self.anno[im_id]
        corners = ann["box_examples_coordinates"][0]
        xs = [p[0] for p in corners]
        ys = [p[1] for p in corners]
        sx, sy = S / float(ann["W"]), S / float(ann["H"])
        bbox = torch.tensor([min(xs) * sx, min(ys) * sy, max(xs) * sx, max(ys) * sy], dtype=torch.float32)
        bboxes3 = []
        for corners3 in ann["box_examples_coordinates"][:3]:
            xs3 = [p[0] for p in corners3]
            ys3 = [p[1] for p in corners3]
            bboxes3.append([min(xs3) * sx, min(ys3) * sy, max(xs3) * sx, max(ys3) * sy])
        while len(bboxes3) < 3:
            bboxes3.append(bboxes3[-1])
        bboxes3 = torch.tensor(bboxes3, dtype=torch.float32)

        if self.augment:
            # Deterministic per-worker: worker_init_fn reseeds RNG at start, so
            # the draws below are reproducible at `seed` (canonical protocol v2).
            img, dens, bbox, bboxes3 = _augment_scale_flip(img, dens, bbox, bboxes3, S, count0)

        return {"imgs": torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float() / 255.0,
                "bboxes": bbox, "bboxes3": bboxes3, "density": dens[None],
                "counts": dens.sum(), "ids": im_id}


def collate_density(batch):
    out = {"imgs": torch.stack([b["imgs"] for b in batch]),
           "bboxes": torch.stack([b["bboxes"] for b in batch]),
           "density": torch.stack([b["density"] for b in batch]),
           "counts": torch.stack([b["counts"] for b in batch])}
    if "bboxes3" in batch[0]:
        out["bboxes3"] = torch.stack([b["bboxes3"] for b in batch])
    if "ids" in batch[0]:
        out["ids"] = [b["ids"] for b in batch]
    return out
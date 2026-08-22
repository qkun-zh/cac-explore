"""FSC147 类无关计数数据集。布局见 docs/research_direction.md。"""
import json
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class FSC147Density(torch.utils.data.Dataset):
    def __init__(self, root, img_size=384, split="train", sigma=None, augment=False):
        self.root, self.size, self.split, self.augment = root, int(img_size), split, augment and split == "train"
        with open(os.path.join(root, "Train_Test_Val_FSC147.pkl"), "rb") as f:
            splits = pickle.load(f)
        self.ids = list(splits[split])
        with open(os.path.join(root, "FSC147_anno.json")) as f:
            self.anno = json.load(f)
        self.sigma = sigma or max(2.0, self.size / 96.0)

    def __len__(self):
        return len(self.ids)

    def _density(self, dots_hw, h, w):
        m = np.zeros((h, w), dtype=np.float32)
        for y, x in dots_hw:
            yi, xi = int(min(max(y, 0), h - 1)), int(min(max(x, 0), w - 1))
            m[yi, xi] = 1.0
        try:
            from scipy.ndimage import gaussian_filter
            return torch.from_numpy(gaussian_filter(m, self.sigma))
        except ImportError:
            t = torch.from_numpy(m)[None, None]
            return F.max_pool2d(t, 3, 1, 1)[0]

    def __getitem__(self, i):
        im_id = self.ids[i]
        img_dir = os.path.join(self.root, "images_384var")
        path = os.path.join(img_dir, f"{im_id}.jpg")
        if not os.path.exists(path):
            alt = os.path.join(self.root, "images_384", f"{im_id}.jpg")
            path = alt if os.path.exists(alt) else path
        img = Image.open(path).convert("RGB")
        W0, H0 = img.size
        img = img.resize((self.size, self.size), Image.BILINEAR)
        sx, sy = self.size / W0, self.size / H0
        box = self.anno[im_id]["box"]  # [x1,y1,x2,y2] 原始坐标
        bbox = torch.tensor([box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy], dtype=torch.float32)
        pts = np.load(os.path.join(self.root, "ground_truth", f"{im_id}.npy"))  # [N,(y,x)] 或 (x,y)
        pts = np.asarray(pts, dtype=np.float32)
        if pts.ndim == 2 and pts.shape[-1] == 2:
            if pts[:, 0].max() <= H0 + 1 and pts[:, 1].max() > H0 + 1:  # 列序为 (y,x)
                px, py = pts[:, 1], pts[:, 0]
            else:  # 列序为 (x,y)
                px, py = pts[:, 0], pts[:, 1]
        else:
            px = py = np.zeros(len(pts) if pts.ndim == 1 else 0)
        dots_hw = np.stack([py * sy, px * sx], 1)
        dens = self._density(dots_hw, self.size, self.size)
        if self.augment:
            if torch.rand(1).item() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                dens = dens.flip(-1)
                bbox = torch.tensor([self.size - bbox[2], bbox[1], self.size - bbox[0], bbox[3]])
        return {"imgs": torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float() / 255.0,
                "bboxes": bbox, "density": dens, "counts": dens.sum()}


def collate_density(batch):
    return {"imgs": torch.stack([b["imgs"] for b in batch]),
            "bboxes": torch.stack([b["bboxes"] for b in batch]),
            "density": torch.stack([b["density"] if b["density"].dim() == 3 else b["density"][None] for b in batch]),
            "counts": torch.stack([b["counts"] for b in batch])}

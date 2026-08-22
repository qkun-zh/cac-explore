"""FSC147 (VarV2 协议) 类无关计数数据集。

服务器布局 /data/dataset/FSC147/：
  images_384_VarV2/<id>.jpg                 # 已缩放至长边~384的可变尺寸图
  gt_density_map_adaptive_384_VarV2/<id>.npy  # 预计算自适应密度图（与图像同尺寸）
  annotation_FSC147_384.json                # {"<id>": {"box":[x1,y1,x2,y2], ...}}
  Train_Test_Val_FSC_147.json               # {"train": [...], "val": [...], "test": [...]}

输出契约：imgs [3,S,S] / bboxes [4](S坐标系) / density [1,S,S] / counts 标量。
密度重采样后按总和守恒缩放，保证计数严格不变。
"""
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


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
        img = Image.open(os.path.join(self.img_dir, f"{im_id}.jpg")).convert("RGB")
        W0, H0 = img.size
        dens = torch.from_numpy(np.load(os.path.join(self.den_dir, f"{im_id}.npy"))).float()
        if dens.dim() == 3:
            dens = dens[0]
        count0 = float(dens.sum())

        S = self.size
        sx, sy = S / W0, S / H0
        img = img.resize((S, S), Image.BILINEAR)
        dens = F.interpolate(dens[None, None], size=(S, S), mode="bilinear", align_corners=False)[0, 0]
        dens = dens * (count0 / dens.sum().clamp_min(1e-8))  # 总和守恒 → 计数不变

        box = self.anno[im_id]["box"]
        bbox = torch.tensor([box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy], dtype=torch.float32)

        if self.augment and torch.rand(1).item() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            dens = dens.flip(-1)
            bbox = torch.tensor([S - bbox[2], bbox[1], S - bbox[0], bbox[3]])

        return {"imgs": torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float() / 255.0,
                "bboxes": bbox, "density": dens[None], "counts": dens.sum()}


def collate_density(batch):
    return {"imgs": torch.stack([b["imgs"] for b in batch]),
            "bboxes": torch.stack([b["bboxes"] for b in batch]),
            "density": torch.stack([b["density"] for b in batch]),
            "counts": torch.stack([b["counts"] for b in batch])}

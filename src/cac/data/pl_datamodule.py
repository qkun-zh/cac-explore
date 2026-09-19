"""FSC147DataModule — Lightning data module (seeded loaders + smoke dataset)."""
from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader

from cac.data.fsc147 import FSC147Density, collate_density


class _SmokeDataset(torch.utils.data.Dataset):
    """Synthetic multi-blob density maps for smoke tests (no real dataset needed)."""
    def __init__(self, n: int, size: int):
        self.n, self.size = n, size

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        g = torch.Generator().manual_seed(i * 7919)
        size = self.size
        img = torch.rand(3, size, size, generator=g)
        x0, y0 = int(torch.randint(0, size // 2, (1,), generator=g)), int(torch.randint(0, size // 2, (1,), generator=g))
        x1, y1 = x0 + size // 3, y0 + size // 3
        gs = size // 8
        yy, xx = torch.meshgrid(torch.arange(gs), torch.arange(gs), indexing="ij")
        dens = torch.zeros(gs, gs)
        for _ in range(1 + int(torch.randint(0, 5, (1,), generator=g))):
            cy, cx = int(torch.randint(0, gs, (1,), generator=g)), int(torch.randint(0, gs, (1,), generator=g))
            s = 1.0 + 2.0 * float(torch.rand(1, generator=g))
            blob = torch.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * s * s))
            dens = dens + blob / blob.sum().clamp_min(1e-6)
        bboxes3 = torch.stack([torch.tensor([x0, y0, x1, y1], dtype=torch.float32)] * 3)
        return {"imgs": img, "bboxes": torch.tensor([x0, y0, x1, y1], dtype=torch.float32),
                "bboxes3": bboxes3, "density": dens[None], "counts": dens.sum()}


def _collate_smoke(batch):
    out = {"imgs": torch.stack([b["imgs"] for b in batch]),
           "bboxes": torch.stack([b["bboxes"] for b in batch]),
           "density": torch.stack([b["density"] for b in batch]),
           "counts": torch.stack([b["counts"] for b in batch])}
    out["bboxes3"] = torch.stack([b["bboxes3"] for b in batch])
    return out


def worker_init_fn(seed: int):
    def _init(worker_id: int) -> None:
        random.seed(seed + worker_id)
        np.random.seed(seed + worker_id)
        torch.manual_seed(seed + worker_id)
    return _init


class FSC147DataModule(LightningDataModule):
    def __init__(self, cfg: dict[str, Any], data_root: str = "/data/dataset/FSC147",
                 img_size: int = 384, batch_size: int = 16, num_workers: int = 4,
                 augment: bool = False, seed: int | None = None, smoke: bool = False):
        super().__init__()
        self.cfg = cfg
        self.data_root = data_root
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augment = augment
        self.seed = seed
        self.smoke = smoke

    def _dl_kw(self) -> dict[str, Any]:
        if self.smoke:
            return dict(batch_size=max(2, min(int(self.batch_size), 4)), collate_fn=_collate_smoke)
        kw: dict[str, Any] = dict(batch_size=self.batch_size, num_workers=self.num_workers,
                                  collate_fn=collate_density, pin_memory=True)
        if self.seed is not None:
            gen = torch.Generator()
            gen.manual_seed(int(self.seed))
            kw["generator"] = gen
            kw["worker_init_fn"] = worker_init_fn(int(self.seed))
        return kw

    def setup(self, stage: str | None = None) -> None:
        if self.smoke:
            self.train_ds = _SmokeDataset(16, self.img_size)
            self.val_ds = _SmokeDataset(8, self.img_size)
        else:
            self.train_ds = FSC147Density(self.data_root, self.img_size, "train", augment=self.augment)
            self.val_ds = FSC147Density(self.data_root, self.img_size, "val", augment=False)

    def train_dataloader(self) -> DataLoader:
        kw = self._dl_kw()
        if self.smoke:
            return DataLoader(self.train_ds, shuffle=True, **kw)
        return DataLoader(self.train_ds, shuffle=True, drop_last=True, **kw)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_ds, shuffle=False, **self._dl_kw())
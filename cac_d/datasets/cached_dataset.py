"""Cached dataset: loads precomputed backbone features (.pt files).
No backbone forward pass needed during training → saves ~6GB VRAM + ~60s/epoch.

Expected .pt structure per sample:
  h2: [192,48,48]  (stage2 features)
  h3: [384,24,24]  (stage3 features)
  e:  [K,256]      (exemplar embeddings, K=3)
  bboxes: [3,4]    (exemplar bboxes, image coords 0..384)
  points: [N,2]    (GT point annotations, image coords 0..384)
"""
import os
from pathlib import Path
from torch.utils.data import Dataset


class CachedDataset(Dataset):
    def __init__(self, cache_dir):
        self.files = sorted(Path(cache_dir).glob("*.pt"))
        if len(self.files) == 0:
            raise FileNotFoundError(f"No .pt files in {cache_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        return torch.load(self.files[i], weights_only=True)


import torch

def cached_collate(batch):
    """Collate for CachedDataset: stacks h2, h3, e, bboxes; keeps points as list."""
    return {
        "h2": torch.stack([b["h2"] for b in batch]),
        "h3": torch.stack([b["h3"] for b in batch]),
        "e":  torch.stack([b["e"]  for b in batch]),
        "bboxes": torch.stack([b["bboxes"] for b in batch]),
        "points": [b["points"] for b in batch],
    }

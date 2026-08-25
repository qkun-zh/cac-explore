from abc import ABC, abstractmethod
import torch, torch.nn as nn, torch.nn.functional as F

class DensityHead(ABC, nn.Module):
    @abstractmethod
    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class SoftplusHead(DensityHead):
    def __init__(self, in_dim=384, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(in_dim, hidden, 1), nn.GELU(), nn.Dropout(0.1), nn.Conv2d(hidden, 1, 1))
    def forward(self, feats):
        return F.softplus(self.net(feats))

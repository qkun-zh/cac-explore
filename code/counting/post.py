import torch

class SumPost:
    def __call__(self, density: torch.Tensor) -> torch.Tensor:
        return density.sum(dim=(1,2,3))

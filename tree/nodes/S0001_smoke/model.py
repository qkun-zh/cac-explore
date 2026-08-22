import torch
import torch.nn.functional as F


class TinyDensityNet(torch.nn.Module):
    """最小密度网络：验证契约用。forward(imgs[B,3,S,S], bboxes[B,4]) -> {'density':[B,1,S/8,S/8]}"""

    def __init__(self, width=16):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, width, 3, 2, 1), torch.nn.BatchNorm2d(width), torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(width, width, 3, 2, 1), torch.nn.BatchNorm2d(width), torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(width, width, 3, 2, 1), torch.nn.BatchNorm2d(width), torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(width, 1, 3, 1, 1),
        )

    def forward(self, imgs, bboxes):  # noqa: ARG002 bboxes 为契约占位
        return {"density": F.softplus(self.features(imgs))}


def build_model(cfg):
    return TinyDensityNet(width=16)

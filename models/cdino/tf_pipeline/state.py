"""Runtime device and optional secondary-backbone registry."""
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Attempt #19: second frozen ConvNeXt-T for dual-feat mean
MODEL2 = None
# Attempt #20: aux DINOv3-ViT-S/16 for cross-arch density mean
XD_MODEL = None
XD_TRANSFORM = None
XD_MAP_KEYS = ['vit_out']


def set_model2(m):
    global MODEL2
    MODEL2 = m


def set_xd(model, transform, map_keys=None):
    global XD_MODEL, XD_TRANSFORM, XD_MAP_KEYS
    XD_MODEL = model
    XD_TRANSFORM = transform
    if map_keys is not None:
        XD_MAP_KEYS = map_keys

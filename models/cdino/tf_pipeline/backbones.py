"""Backbone + transform construction for all supported model_name values."""
import re
import copy as _copy

import torch
import timm
import torchvision.transforms as T

from src.model import VisualBackbone
from . import state as _state
from .state import device, set_model2


def load_backbone(args):
    # === LOCAL ADAPTER (not upstream): cached timm DINOv2 ViT-S for boxes without github egress ===
    if args.model_name == 'timm_vits14_reg':
        from src.timm_adapter import TimmDinov2Backbone
        resize_dim = 840
        model = TimmDinov2Backbone().to(device).eval()
        transform = T.Compose([
            T.Resize((resize_dim, resize_dim), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
    elif args.model_name.startswith('dinov3_convnext'):
        # local: DINOv3 ConvNeXt-Tiny (conv family)
        from dinov3_convnext_adapter import Dinov3ConvnextBackbone
        resize_dim = args.input_size
        hf_name = 'facebook/dinov3-convnext-tiny-pretrain-lvd1689m'
        model = Dinov3ConvnextBackbone(hf_name=hf_name, img_size=resize_dim).to(device).eval()
        # Attempt #19: second identical frozen backbone (~28M; dual total ~56M)
        
        if args.dual_convnext:
            set_model2(Dinov3ConvnextBackbone(hf_name=hf_name, img_size=resize_dim).to(device).eval())
            print(f"DUAL backbone loaded {_state.MODEL2}", flush=True)
        transform = T.Compose([
            T.Resize((resize_dim, resize_dim), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
    elif args.model_name.startswith('dinov3_'):
        from src.dinov3_adapter import Dinov3VitBackbone
        resize_dim = 832
        hf_map = {
            'dinov3_vits16': 'facebook/dinov3-vits16-pretrain-lvd1689m',
            'dinov3_vitb16': 'facebook/dinov3-vitb16-pretrain-lvd1689m',
            'dinov3_vitl16': 'facebook/dinov3-vitl16-pretrain-lvd1689m',
        }
        hf_name = hf_map[args.model_name]
        model = Dinov3VitBackbone(hf_name=hf_name, img_size=resize_dim).to(device).eval()
        transform = T.Compose([
            T.Resize((resize_dim, resize_dim), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
    elif 'mae' in args.model_name or 'clip' in args.model_name or 'sam' in args.model_name:
        match = re.search(r'patch(\d+)', args.model_name)
        patch_size = int(match.group(1))
        resize_dim = patch_size * 60
        model = VisualBackbone(args.model_name, img_size=resize_dim).to(device).eval()
        data_config = timm.data.resolve_model_data_config(model)
        data_config['input_size'] = (3, resize_dim, resize_dim)
        transform = timm.data.create_transform(**data_config, is_training=False)
    else:
        if 'dinov2' in args.model_name:
            resize_dim = 840
        else:
            resize_dim = 480
        model = VisualBackbone(args.model_name, img_size=resize_dim).to(device).eval()
        transform = T.Compose([
            T.Resize((resize_dim, resize_dim), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    return model, transform, resize_dim

from dataclasses import dataclass

@dataclass
class Config:
    image_size: int = 384
    hf_model: str = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m"
    backbone_dims: tuple = (96, 192, 384, 768)
    embed_dim: int = 256
    exemplar_layers: int = 2
    roi_size: int = 7
    d_fine: int = 128
    cond_dim: int = 64
    gauss_sigma: float = 1.5
    density_weight: float = 1.0
    cnt_weight: float = 1.0
    density_loss: str = "mse"
    gauss_knn: int = 3
    sigma_beta: float = 1.0
    sigma_min: float = 1.0
    sigma_max: float = 8.0
    batch_size: int = 16
    epochs: int = 40
    lr: float = 1e-3
    weight_decay: float = 0.05
    amp: bool = True
    num_workers: int = 8
    seed: int = 0
    warmup_epochs: int = 2
    stable_epochs: int = 10
    test_every: int = 4
    eta_min_ratio: float = 0.1
    ema_decay: float = 0.999
    flip_p: float = 0.5
    color_jitter: bool = False
    best_ckpt: str = "/data/runs/N0033_salf_moe/best.pth"
    use_queue: bool = False
    queue_capacity: int = 32
    queue_m: int = 2
    use_cached_features: bool = False
    cache_dir: str = "/data/cache/fsc147_features"
    balance_weight: float = 0.01

def build_config():
    return Config()

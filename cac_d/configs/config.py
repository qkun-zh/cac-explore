from dataclasses import dataclass

@dataclass
class Config:
    # data (HF dataset isentropic/FSC147; HF model+processor dinov3-convnext-tiny)
    image_size: int = 384
    hf_model: str = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m"
    backbone_dims: tuple = (192, 384)       # stage2@1/8, stage3@1/16
    # exemplar bank
    embed_dim: int = 256
    exemplar_layers: int = 2
    roi_size: int = 7
    # fine grid / matching
    d_fine: int = 128
    cond_dim: int = 64
    gauss_sigma: float = 1.5
    # heads
    # density & count calibration
    density_weight: float = 1.0
    cnt_weight: float = 1.0
    # training (torch.optim.AdamW) — schedule: warmup 2 + stable 10 + cosine 20 = 32
    batch_size: int = 32
    epochs: int = 32
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
    # augmentation (geometric synced with boxes/points)
    flip_p: float = 0.5
    color_jitter: bool = False
    best_ckpt: str = "/tmp/cac_d_best.pth"
    # precomputed feature cache (skip backbone during training)
    use_cached_features: bool = False
    cache_dir: str = "/data/cache/fsc147_features"       # true-384px reference cache
    # FAST-EXPERIMENT profile (224px: ~30s/ep, fits 3-4 concurrent runs on RTX3060).
    # Reduced spatial detail (grid 56x56) — NOT comparable with the 384px baseline.
    # Usage: CAC_D_OVERRIDE='{"cache_dir":"/data/cache/fsc147_features_224","image_size":224}'
    # NOTE: image_size must be overridden together with cache_dir (coords & sigma scale).

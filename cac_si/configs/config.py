from dataclasses import dataclass

@dataclass
class Config:
    # data
    image_size: int = 384            # base resolution (224 = fast lane)
    hf_model: str = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m"
    backbone_dims: tuple = (192, 384)
    prompt_size: int = 112
    prompt_margin: float = 0.25
    # scale-invariant encoder (nominal; sizes snapped to /16 multiples internally)
    scales: tuple = (0.75, 1.0, 1.25)
    # condenser (cross-attention)
    d_sim: int = 256
    n_heads: int = 4
    ff: int = 512
    cond_dim: int = 64
    # INR decoder
    inr_hidden: int = 128
    inr_layers: int = 4
    fourier_freqs: tuple = (1, 2, 4, 8)
    inr_sigma: float = 0.02          # GT gaussian sigma, normalized coords
    n_samples: int = 256             # random x per training step
    fg_sampling: float = 0.0         # fraction of samples drawn near GT points (0=uniform)
    pos_enc: bool = False            # 2D sincos positional encoding on attention tokens
    quad_grid: int = 32              # count quadrature grid (train)
    eval_grid: int = 64              # count quadrature grid (eval)
    # loss weights
    density_weight: float = 1.0
    cnt_weight: float = 1.0
    uncertainty_weight: bool = False   # Kendall&Gal learned balancing of L_den/L_cnt
    # training — schedule: warmup 2 + stable 10 + cosine 20 = 32
    batch_size: int = 16
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
    flip_p: float = 0.5
    color_jitter: bool = False
    best_ckpt: str = "/tmp/cac_si_best.pth"

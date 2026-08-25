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
    gauss_sigma: float = 1.5                # cells at 96x96 grid
    sim_weight: float = 0.25
    # heads
    pile_hidden: int = 128
    uot_topk: int = 2048
    # UOT minimal: transport + demand KL
    transport_weight: float = 0.5
    entropy_reg: float = 0.08
    demand_tau: float = 1.0
    sinkhorn_iters: int = 32
    # density & count calibration
    density_weight: float = 1.0
    cnt_weight: float = 1.0
    # training (torch.optim.AdamW)
    batch_size: int = 32
    epochs: int = 42
    lr: float = 1e-3
    weight_decay: float = 0.05
    amp: bool = True
    num_workers: int = 8
    seed: int = 0
    warmup_epochs: int = 2
    stable_epochs: int = 10                 # constant 1.0× after warmup
    eta_min_ratio: float = 0.1              # cosine floor = lr * ratio (0.05太激进→末段5e-5步长过小)
    ema_decay: float = 0.999                # per-step shadow weights
    # augmentation (geometric synced with boxes/points)
    flip_p: float = 0.5
    color_jitter: bool = False
    best_ckpt: str = "/tmp/cac_d_best.pth"

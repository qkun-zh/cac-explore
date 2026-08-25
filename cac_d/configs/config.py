from dataclasses import dataclass

@dataclass
class Config:
    # data
    image_size: int = 384
    hf_model: str = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m"
    hf_processor: str = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m"
    # prompt
    prompt_hidden: int = 256
    # heads
    pile_hidden: int = 128
    density_hidden: int = 128
    # UOT minimal: transport + demand KL
    transport_weight: float = 1.0
    entropy_reg: float = 0.08
    demand_tau: float = 1.0
    sinkhorn_iters: int = 32
    # density
    density_weight: float = 1.0
    # training
    batch_size: int = 8
    epochs: int = 40
    lr: float = 1e-3
    weight_decay: float = 0.05

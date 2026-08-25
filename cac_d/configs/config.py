from dataclasses import dataclass

@dataclass
class Config:
    # data
    image_size: int = 384
    hf_model: str = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m"
    hf_processor: str = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m"
    # backbone feature
    backbone_dim: int = 384
    # post-backbone redesign
    embed_dim: int = 256
    exemplar_layers: int = 2
    roi_size: int = 7
    # heads
    pile_hidden: int = 128
    density_hidden: int = 128
    # count consistency between pile and density branches
    consist_weight: float = 0.5
    # UOT minimal: transport + demand KL
    transport_weight: float = 1.0
    entropy_reg: float = 0.08
    demand_tau: float = 1.0
    sinkhorn_iters: int = 32
    # density
    density_weight: float = 1.0
    # training
    batch_size: int = 32
    epochs: int = 40
    lr: float = 2e-3
    weight_decay: float = 0.05
    amp: bool = True
    num_workers: int = 8

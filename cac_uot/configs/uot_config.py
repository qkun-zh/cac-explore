from dataclasses import dataclass

@dataclass
class UOTConfig:
    # image / patch
    image_size: int = 384
    patch_size: int = 16

    # HF backbone / processor
    hf_model: str = "facebook/dinov3-vits16-pretrain-lvd1689m"
    hf_processor: str = "facebook/dinov3-vits16-pretrain-lvd1689m"

    # pile predictor
    hidden_dim: int = 384
    head_hidden: int = 128

    # prompt: LOCA OPE (shape+appearance queries, iterative cross-attn adaptation)
    ope_emb_dim: int = 256
    ope_kernel_dim: int = 3                       # prototype kernel size s×s
    ope_iters: int = 3                            # L adaptation steps
    ope_heads: int = 8
    ope_reduction: int = 16                       # S / feature_map_size

    # unbalanced OT  (KL-relaxed, log-domain Sinkhorn)
    transport_weight: float = 1.0                 # α
    entropy_reg: float = 0.08                     # ε
    supply_tau: float = 0.5                       # τ_supply (row, surplus regularizer)
    demand_tau: float = 1.0                       # τ_demand (col, ↔ β)
    sinkhorn_iters: int = 10                      # K
    repulsion_weight: float = 1e-3                # λ
    repulsion_sigma_scale: float = 1.0            # σ

    # P1 direct count-mass supervision |Σw − N|
    count_mass_weight: float = 1.0

    # UW: Kendall & Gal learnable loss balancing (off by default)
    use_uw: bool = False
    uw_init: float = 0.0

    # training
    batch_size: int = 8
    epochs: int = 40
    lr: float = 1e-3
    weight_decay: float = 0.05
    amp: bool = True

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

    # unbalanced OT  (KL-relaxed, log-domain Sinkhorn)
    transport_weight: float = 1.0                 # α
    entropy_reg: float = 0.15                     # ε (raised: widen support radius, fix dead-zone)
    supply_tau: float = 1.0                       # τ_supply (row,  ↔ γ)
    demand_tau: float = 1.0                       # τ_demand (col, ↔ β)
    sinkhorn_iters: int = 10                      # K
    repulsion_weight: float = 1e-3                # λ
    repulsion_sigma_scale: float = 1.0            # σ

    # count-mass auxiliary: |Σw − N| direct supervision (P1)
    count_mass_weight: float = 1.0

    # three fixes  F1 / F3 / F4  — off; current experiment = cnt_mass(P1) + eps widen(P2) only
    box_anchor_weight: float = 0.0                # F1: off (failed single test)
    loss_normalize: str = "none"                  # F3: off
    use_standardized_gate: bool = False           # F4: off

    # training
    batch_size: int = 8
    epochs: int = 40
    lr: float = 1e-3
    weight_decay: float = 0.05
    amp: bool = True

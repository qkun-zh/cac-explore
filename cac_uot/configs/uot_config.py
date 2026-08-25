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
    entropy_reg: float = 0.05                     # ε
    supply_tau: float = 1.0                       # τ_supply (row,  ↔ γ)
    demand_tau: float = 1.0                       # τ_demand (col, ↔ β)
    sinkhorn_iters: int = 10                      # K
    repulsion_weight: float = 1e-3                # λ
    repulsion_sigma_scale: float = 1.0            # σ

    # three fixes  F1 / F3 / F4  (on per user request)
    box_anchor_weight: float = 1.0                # F1: exemplar box mass anchor
    loss_normalize: str = "demand_size"           # F3: loss / N
    use_standardized_gate: bool = True            # F4: standardized gate

    # training
    batch_size: int = 8
    epochs: int = 40
    lr: float = 1e-3
    weight_decay: float = 0.05
    amp: bool = True

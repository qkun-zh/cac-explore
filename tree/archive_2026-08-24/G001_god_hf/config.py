cfg = dict(
    # HF stack
    hf_model="facebook/dinov3-vits16-pretrain-lvd1689m",
    hf_processor="facebook/dinov3-vits16-pretrain-lvd1689m",
    input_size=384,
    patch_size=16,
    # GOD v7 solver: standard KL-relaxed unbalanced OT, log-domain Sinkhorn
    god_solver="sinkhorn",
    sinkhorn_K=10,
    god_tau_row=1.0,   # supply-side KL strength (↔ γ)
    god_tau_col=1.0,   # demand-side KL strength (↔ β)
    god_alpha=1.0,
    god_epsilon=0.05,
    god_lambda=1e-3,
    god_sigma_scale=1.0,
    # prompt-A
    gate_alpha_init=1.0,
    gate_beta_init=0.0,
    # training
    batch_size=8,
    epochs=40,
    lr=1e-3,
    weight_decay=0.05,
    amp=True,
    max_params_M=32,
    data_root="/data/dataset/FSC147",
    smoke=False,
)

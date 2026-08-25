cfg = dict(
    # HF stack
    hf_model="facebook/dinov3-vits16-pretrain-lvd1689m",
    hf_processor="facebook/dinov3-vits16-pretrain-lvd1689m",
    input_size=384,
    patch_size=16,
    # GOD loss
    god_alpha=1.0,
    god_beta=0.5,
    god_gamma=0.1,
    god_epsilon=0.05,  # entropy
    god_lambda=1e-3,
    god_sigma_scale=1.0,  # σ = median_box * scale / 384
    # prompt-A
    gate_alpha_init=1.0,
    gate_beta_init=0.0,
    # training
    batch_size=4,
    epochs=40,
    lr=1e-3,
    weight_decay=0.05,
    amp=True,
    max_params_M=32,
    # engine compat
    use_hf_dataset=True,  # load_dataset("isentropic/FSC147")
    data_root="/data/dataset/FSC147",
    # for old engine smoke
    smoke=False,
)

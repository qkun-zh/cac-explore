cfg = dict(
    input_size=384,
    epochs=20,
    batch_size=8,
    lr=1e-3,
    weight_decay=1e-4,
    eta_min=1e-6,
    amp=True,
    max_params_M=32,
    loss_count_weight=0.3,
    num_workers=4,
)

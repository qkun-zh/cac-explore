cfg = dict(
    exp_name="S0001_smoke",
    epochs=2,
    batch_size=4,
    lr=1e-3,
    weight_decay=1e-4,
    input_size=128,
    num_classes=1,
    max_params_M=0.5,
    loss_count_weight=0.3,
    data_root="/data/dataset/FSC147",
    amp=True,
)

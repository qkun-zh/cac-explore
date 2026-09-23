# CountingDINO ablation queue (after N0030 full-flag lands)

GPU: RTX 3060 **12GB** — inference only, no training. One job at a time.
Log prefix `/data/repro/logs/`.

| # | model | flags | bar / purpose | status |
|---|---|---|---|---|
| 1 | dinov3_vitl16 | DEI+filter+ellipse (full) | H0028 ≤38.696 | **DONE 32.531 SUPPORTED** N0030 |
| 2 | dinov3_vits16 | full + cosine | H0030 ≤39.696 (N0033) | **QUEUED** queue_cosine.sh after TF |
| 3 | dinov3_vitb16 | full (DEI+filter+ellipse) | mid backbone S↔L | next after #2 |
| 4 | dinov3_vitl16 | full + normalize_features | paper normalize path | if #1 fails |
| 5 | dinov3_vitl16 | full + DEI2 | multiscale DEI | if #1 < 39.7 |

Common args:
```
cd /data/cdino_run && HF_HOME=/data/asset/hf HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
/data/miniconda/envs/cac/bin/python convolutional_counting.py \
  --model_name <NAME> --split val \
  --divide_et_impera <T/F> --filter_background <T/F> \
  --ellipse_normalization <T/F> --ellipse_kernel_cleaning <T/F> \
  --img_dir /data/dataset/FSC147/images_384_VarV2 \
  --density_map_dir /data/dataset/FSC147/gt_density_map_adaptive_384_VarV2 \
  --annotation /data/dataset/FSC147/annotation_FSC147_384.json \
  --splits /data/dataset/FSC147/Train_Test_Val_FSC_147.json \
  --log_file results/results.csv --no_skip false
```

## TF baselines (separate queue, after CDINO series or when GPU free)

| method | weights | notes |
|---|---|---|
| TFCounter | SAM **vit_b** only | `/data/repro/run_tfc.sh` |
| TF-CAC | SAM vit_b + DINOv2-vits14 | `/data/repro/run_tfcac.sh` |

**Do NOT use SAM vit_h** (VRAM / size gate).

Reference anchors:
- seed local: vits16 full **39.696**
- paper: DINOv2-L **25.48** val / 20.93 test
- external N0015 (other tree): 19.343 / 19.066
- inference cal: 18.424 / 19.606 (H0029)
- TF-CAC paper (test): 12.26 MAE

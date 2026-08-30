# Research Direction (human-specified)

## Direction Statement

On the FSC147 class-agnostic counting benchmark, discover lightweight, accurate, transferable CAC architectures via hypothesis-driven multi-agent evolutionary search.

**Target**: ≤32M parameters, MAE ≤ 10 on FSC147 **test**.

## Task Definition

- Input: RGB image + exemplar box(es) indicating the target category
- Output: density map whose integral is the predicted count
- Metrics: MAE (primary), RMSE (secondary); official FSC147 train/val/test splits

## Dataset Layout (server `/data/dataset/FSC147`, VarV2 protocol)

```
FSC147/
├── images_384_VarV2/<id>.jpg             # variable-aspect images, long side ~384
├── gt_density_map_adaptive_384_VarV2/<id>.npy  # precomputed adaptive density maps (same size as image)
├── annotation_FSC147_384.json            # {"<id>.jpg": {"W","H" (original size), "box_examples_coordinates": [[[x,y]×4]×3], ...}}
└── Train_Test_Val_FSC_147.json           # {"train":[...], "val":[...], "test":[...]} — 6146 images / 147 categories
```

Loader `code/data/fsc147.py` implements this layout; density resampling is sum-preserving (counts strictly unchanged). Exemplar boxes are given in ORIGINAL image coordinates and are scaled by `S/W`, `S/H` from the annotation's W/H. JSON ids carry a `.jpg` suffix; the loader strips it.
Source: official package uploaded by the user via scp; same content as the HF mirror `isentropic/FSC147`.

## Constraints

- **Backbone**: pretrained from HF Hub or timm (`timm.create_model(name, pretrained=True)` / AutoModel); may be frozen or *partially fine-tuned* with differential LR (validated: DINOv2-S reg4 top blocks 10-11 @ lr×0.1 beats frozen 21.53→20.44, N0021; 2026-08-30 user directive explicitly allows mid-layer FT on DINOv3-ConvNeXt-Tiny stages 1-2 @ lr×0.1 and exploration of intermediate (hs 2/3) vs final (hs 4) readout). Backbone choice is itself an architectural decision recorded in idea.md; full-FT is refuted (EBC 48.4 collapse) — keep unfrozen scope narrow (middle layers only) + lr low. Intermediate features are hypothesized to be more suitable for counting than final-layer outputs (compare hs_map (2,3) vs (3,4)).
- **Parameter budget**: ≤32M TOTAL including the backbone (memory footprint counts). Engine asserts `max_params_M` (default 32) over all params
- Parameter budget and training-time cap live in each node's `config.py`; default wall clock ≤30 minutes
- Single RTX 3060 12GB, AMP mixed precision
- Architecture must expose `build_model(cfg)` taking `[B,3,H,W]` + exemplar bboxes; engine optimizes only `requires_grad` params

**Server notes for pretrained weights**: run_node.sh exports `HF_HOME=/data/asset/hf` (persistent) and `HF_ENDPOINT=https://hf-mirror.com` (reachable mirror). `timm` + `huggingface_hub` are installed in env `cac`. First run of a node downloads backbone weights — do it once inside tmux, cache persists in /data.

## Baseline Reference (from earlier cv_study work)

DViT-Light: 392×392 input, 48-cell grid density head, MSE + 0.3·L1 count loss, AdamW 1e-3, cosine schedule, 150 epochs. The root lineage may start from a similar recipe or from scratch — the Idea Agent decides.

Note: S0001_smoke (0.01M params, 27s for 2 epochs) gives a throughput reference: full-budget nodes can afford roughly 50–100× more compute per run within τ_max=30min.

# TF baseline staging (2026-09-22)

## VRAM gate
RTX 3060 12GB. **SAM-vit_b only** for TF methods. No training.

## Assets
| path | purpose |
|---|---|
| `/data/asset/TFCounter/` | TFCounter code + `pretrain/sam_vit_b_01ec64.pth` |
| `/data/asset/TF-CAC/` | TF-CAC code, `segment_anything -> shi_segment_anything`, `cac_data -> FSC147` |
| `/data/asset/pretrain/` | shared weights: sam_vit_b, dinov2_vits14 |
| `/data/asset/torch_hub_checkpoints/` | dinov2 for torch.hub |
| `/root/.cache/torch/hub/facebookresearch_dinov2_main` | pre-cloned dinov2 |

## Launch (GPU free only; chained after CDINO close)
- `/data/repro/run_tfc.sh` → TFCounter val
- `/data/repro/run_tfcac.sh` → TF-CAC val (vit_b + dinov2-vits14)
- chains: `chain_close.sh` (N0030) → `chain2_tfc.sh` → `chain3_tfcac.sh`

## Paper anchors (do not mix with val without noting split)
- TF-CAC paper FSC-147 **test** MAE 12.26 (box)
- CountingDINO paper DINOv2-L test 20.93 / val often reported separately
- Our seed CDINO vits16 full **val 39.696**


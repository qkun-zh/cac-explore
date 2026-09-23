#!/bin/bash
# Run a graph model node on the server: champion flags + your overrides.
# Usage: run_variant.sh <NODE_ID> [--extra_flag value ...]
# Example: run_variant.sh N0030 --mass_sharpen True
# Logs: /data/repro/logs/<NODE_ID>_sub286.log
set -u
NODE=${1:?usage: run_variant.sh <NODE_ID> [flags...]}; shift || true
export HF_HOME=/data/asset/hf HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
PY=/data/miniconda/envs/cac/bin/python
cd "$(dirname "$0")/cdino" || exit 1
LOG=/data/repro/logs/${NODE}_sub286.log
$PY convolutional_counting.py \
  --model_name dinov3_convnext_tiny \
  --split val \
  --subset_file /data/repro/val_subset286.json \
  --divide_et_impera True \
  --divide_et_impera_twice True \
  --filter_background True \
  --filter_thresh_scale 0.5 \
  --mass_weight_exemplar True \
  --ellipse_normalization True \
  --ellipse_kernel_cleaning True \
  --count_readout density \
  --adaptive_dei True \
  --adaptive_dei_thresh 1.5 \
  --adaptive_dei_depth 1 \
  --adaptive_dei_max 8 \
  --img_dir /data/dataset/FSC147/images_384_VarV2 \
  --density_map_dir /data/dataset/FSC147/gt_density_map_adaptive_384_VarV2 \
  --annotation /data/dataset/FSC147/annotation_FSC147_384.json \
  --splits /data/dataset/FSC147/Train_Test_Val_FSC_147.json \
  --log_file "results/${NODE}_sub286.csv" \
  --no_skip true \
  "$@" \
  > "$LOG" 2>&1
rc=$?
echo "rc=$rc log=$LOG"
grep -E "MAE|ratio_sum|Traceback|AssertionError|out of memory" "$LOG" | tail -20 || true
exit $rc

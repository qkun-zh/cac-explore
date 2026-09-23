#!/bin/bash
# Run a graph model node on the server: champion flags + ONE override flag.
# Usage: run_variant.sh <NODE_ID> [--flag value]
# Example: run_variant.sh N0031 --num_exemplars 8
# Logs: /data/repro/logs/<NODE_ID>_sub286.log
set -u
NODE=${1:?usage: run_variant.sh <NODE_ID> [--flag value]}; shift || true

# --- single-flag guard: at most one --flag in "$@" ---
n_override=0
for a in "$@"; do
  case "$a" in
    --*) n_override=$((n_override + 1)) ;;
  esac
done
if [ "$n_override" -gt 1 ]; then
  echo "FORBIDDEN: only one override flag allowed (got $n_override)" >&2
  exit 2
fi

# --- forbidden keys: protocol pins + closed champion constants ---
for a in "$@"; do
  case "$a" in
    --subset_file|--split|--img_dir|--density_map_dir|--annotation|--splits|--model_name|--log_file|--no_skip|--log_results)
      echo "FORBIDDEN key: $a (protocol/path is pinned)" >&2
      exit 2
      ;;
    --adaptive_dei_thresh|--adaptive_dei_depth|--adaptive_dei_max|--filter_thresh_scale|--mass_weight_exemplar|--divide_et_impera|--divide_et_impera_twice|--filter_background|--ellipse_normalization|--ellipse_kernel_cleaning|--count_readout)
      echo "FORBIDDEN key: $a (champion constant closed by bans)" >&2
      exit 2
      ;;
    --tta_flip|--scale_views|--input_size|--density_warp|--feature_modulation|--dense_fs_gate|--dense_norm_gate|--dense_hr_gate|--multi_scale_ex|--dense_struct_stage|--context_aware_sim|--soft_exemplar_cef|--dual_convnext|--dual_density|--mlvl_fuse|--filter_otsu|--filter_prenorm|--roi_norm_per_exemplar|--per_exemplar_filter|--transductive_proto|--exemplar_reduce|--box_peak_residual|--thresh_expand|--bg_sub_integral|--tile_split|--local_contrast|--input_unsharp|--cosine_similarity|--boxwise_counts|--count_readout|--mass_restore|--thresh_mode|--filter_thresh_scale)
      echo "FORBIDDEN key: $a (closed family; see graph.py bans)" >&2
      exit 2
      ;;
  esac
done

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

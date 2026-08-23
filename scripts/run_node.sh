#!/usr/bin/env bash
# Launch training for a node on the server (tmux session node_<ID>). Usage: bash scripts/run_node.sh <NODE_ID> [EPOCHS]
set -euo pipefail
NODE=${1:?usage: run_node.sh <NODE_ID> [EPOCHS]}
REPO=/data/repo
RUNS=/data/runs/$NODE
PY=${PY:-/data/miniconda/envs/cac/bin/python}
mkdir -p "$RUNS" /data/asset/hf
export HF_HOME=/data/asset/hf            # persistent HF/timm cache (only /data survives)
export HF_ENDPOINT=https://hf-mirror.com # reachable mirror on this network
cd "$REPO"
git pull --ff-only || true
ARGS="--node_dir $REPO/tree/nodes/$NODE --run_dir $RUNS"
[ -n "${2:-}" ] && ARGS="$ARGS --epochs $2"
tmux kill-session -t "node_$NODE" 2>/dev/null || true
tmux new-session -d -s "node_$NODE" "$PY -u code/engine/train.py $ARGS 2>&1 | tee $RUNS/train.log"
echo "[run_node] tmux session node_$NODE started; watch: tmux capture-pane -t node_$NODE -p | tail -30"

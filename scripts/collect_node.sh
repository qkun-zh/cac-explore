#!/usr/bin/env bash
# 本地侧：把服务器上的实验产物回传到节点目录。用法: bash scripts/collect_node.sh <NODE_ID> [REMOTE_HOST]
set -euo pipefail
NODE=${1:?usage: collect_node.sh <NODE_ID> [HOST]}
HOST=${2:-cac-server}
HERE="$(cd "$(dirname "$0")/.." && pwd)"
NODE_DIR="$HERE/tree/nodes/$NODE"
mkdir -p "$NODE_DIR"
scp -q "$HOST:/data/runs/$NODE/result.json" "$NODE_DIR/result.json" && echo "[collect] result.json ok"
ssh "$HOST" "tail -n 500 /data/runs/$NODE/train.log 2>/dev/null" > "$NODE_DIR/train.log" || true
echo "[collect] train.log tail -> $NODE_DIR/train.log"
ls -la "$NODE_DIR"

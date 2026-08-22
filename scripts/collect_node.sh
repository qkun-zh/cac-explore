#!/usr/bin/env bash
# Local side: pull experiment artifacts from the server into the node directory. Usage: bash scripts/collect_node.sh <NODE_ID> [HOST]
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

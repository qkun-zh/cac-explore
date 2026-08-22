# T0001_smoke-node execution

- status: done
- created: 2026-08-22T23:00+08:00
- role: executor
- node: tree/nodes/S0001_smoke
- inputs: torch cu128 ready (env `cac`, cuda=True verified)
- outputs: /data/runs/S0001_smoke/{result.json,train.log,best.pth} → collect into tree/nodes/S0001_smoke/
- steps:
  1. ssh cac-server: cd /data/repo && git pull && bash scripts/run_node.sh S0001_smoke
  2. watch: tmux capture-pane -t node_S0001_smoke -p | tail -30
  3. local: bash scripts/collect_node.sh S0001_smoke
  4. update STATE.md/journal, rename task card to _done_
- outcome: success (val MAE 46.69, real data, 27s)

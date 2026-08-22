# T0001_冒烟节点执行

- status: pending
- created: 2026-08-22T23:00+08:00
- role: executor
- node: tree/nodes/S0001_smoke
- inputs: torch cu128 就绪（/data/asset/torch_fix.log 出现 TORCH_OK 且 cuda=True）
- outputs: /data/runs/S0001_smoke/{result.json,train.log,best.pth} → collect 回传 tree/nodes/S0001_smoke/
- steps:
  1. ssh cac-server: cd /data/repo && git pull && bash scripts/run_node.sh S0001_smoke
  2. 观察: tmux capture-pane -t node_S0001_smoke -p | tail -30
  3. 本地: bash scripts/collect_node.sh S0001_smoke
  4. 更新 STATE.md/journal，任务改名 _done_

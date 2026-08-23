# T0006_exec_N0002_dino_protocorr

- status: pending
- created: 2026-08-23T09:12:00+08:00
- role: executor
- node: tree/nodes/N0002_dino_protocorr
- inputs: model.py, config.py, /data/dataset/FSC147
- outputs: result.json, train.log via collect; tree.json status -> running -> done/failed
- notes: run via tmux on cac-server (HF proxy + git proxy active); τ_max=30min

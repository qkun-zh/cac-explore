# T0012_exec_N0005_swin_promptseg

- status: pending
- created: 2026-08-23T10:10:00+08:00
- role: executor
- node: tree/nodes/N0005_swin_promptseg
- inputs: model.py, config.py, /data/dataset/FSC147
- outputs: result.json + train.log via collect; tree.json running -> done/failed
- notes: smoke green @0e8eb76 (28.22M); swin channels-last fix applied; watch for eval MAE swings seen in smoke

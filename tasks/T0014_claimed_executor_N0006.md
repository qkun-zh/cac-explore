# T0014_exec_N0006_swin_promptv2

- status: pending
- created: 2026-08-23T10:24:00+08:00
- role: executor
- node: tree/nodes/N0006_swin_promptv2
- inputs: model.py, config.py, /data/dataset/FSC147
- outputs: result.json + train.log via collect; tree.json running -> done/failed
- notes: smoke green @2ee70f0 (28.37M); 30ep est ~8min; tests H0012/H0013 vs parent N0005 (32.66)

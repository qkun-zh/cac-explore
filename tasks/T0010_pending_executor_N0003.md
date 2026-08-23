# T0010_exec_N0003_convnext_xattn

- status: pending
- created: 2026-08-23T09:43:00+08:00
- role: executor
- node: tree/nodes/N0003_convnext_xattn
- inputs: model.py, config.py, /data/dataset/FSC147
- outputs: result.json + train.log via collect; tree.json running -> done/failed
- notes: smoke green @d6cf84d (16.93M, init stabilized); 14 epochs est ~10min

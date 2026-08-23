# T0035_exec_N0016_seqcount

- status: pending
- created: 2026-08-23T14:55:00+08:00
- role: executor
- node: tree/nodes/N0016_dino_seqcount
- inputs: model.py, config.py (paradigm=seq), extended engine, /data/dataset/FSC147
- outputs: result.json + train.log via collect; tree.json running -> done/failed
- notes: USER PRIORITY — SeqCount paradigm first. smoke green 26.40M both paradigms. H0024 bar <=19.0 & ratio<3.3; eval AR decode slow (~196 steps/batch), watch wall-clock

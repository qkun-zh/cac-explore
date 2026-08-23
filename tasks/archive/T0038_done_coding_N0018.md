# T0038_coding_N0018_dino_protoiter
- status: pending
- created: 2026-08-23
- role: coding
- node: tree/nodes/N0018_dino_protoiter
- inputs: idea.md (LOCA-style iterative prototype refinement T=2 K=16), parent N0010 model.py/config.py, memory/failure_modes.md
- outputs: model.py (build_model(cfg)) + config.py; tree.json -> coded; smoke green before done
- notes: +Linear(384->384) ~0.148M -> ~23.3M; loop inside forward(), backbone runs once, adapter+head re-run per round, output density = mean of 3 maps (engine contract unchanged); H0027 bar MAE<=19.5 & RMSE/MAE<3.4; DISPROVED IF MAE>21.53 or OOM

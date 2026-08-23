# T0040_exec_N0019_dino_tailup
- status: claimed
- created: 2026-08-23T16:20:00+08:00
- role: executor
- node: tree/nodes/N0019_dino_tailup
- inputs: model.py (N0010 verbatim), config.py (+tail_reweight True, tail_exp -0.5 UP-weight dense), engine generic path
- outputs: result.json + train.log; tree.json -> done
- notes: smoke green @0c95a77 23.11M; H0028 bar ratio<3.4 & MAE<=21.53; early-stop armed

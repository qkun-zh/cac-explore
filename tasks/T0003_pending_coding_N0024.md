# T0003_coding_N0024_ebc_partialft

- status: pending          # pending -> claimed_<agent> -> done | cancelled
- created: 2026-08-24T08:37:45+08:00
- role: coding
- node: tree/nodes/N0024_ebc_partialft
- inputs: tree/nodes/N0024_ebc_partialft/idea.md + memory/failure_modes.md + parent tree/nodes/N0022_dino_ebc_fullft/{model.py,config.py} (copy base) + freeze-mask/param_groups pattern from champion tree/nodes/N0021_dino_partialft/model.py:39-66
- outputs: tree/nodes/N0024_ebc_partialft/model.py + config.py adapted (freeze blocks 10-11+norm, backbone_lr_mult 0.1, dropout 0.1, keep paradigm="ebc"/num_bins=16) + smoke green (`python code/engine/train.py --node_dir tree/nodes/N0024_ebc_partialft --smoke`) + flip card done
- notes: SINGLE targeted change vs parent — partial FT swap only, NO head/engine/data edits. Copy parent model.py verbatim, then: (1) add champion freeze mask (requires_grad True ONLY for backbone names matching `blocks.10.`/`blocks.11.`/`norm.`), (2) cfg backbone_lr_mult 0.05→0.1 and dropout 0.15→0.1. Update docstring to N0024 identity. Smoke must print params≈23.2M and pass the <32M assert; verify trainable-param split (~4.6M trainable). Do NOT touch engine, data code, other nodes, or git.

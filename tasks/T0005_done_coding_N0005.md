# T0005_code_N0005_swin_promptseg

- status: pending          # pending -> claimed_<agent> -> done | cancelled
- created: 2026-08-23T09:30:00+08:00
- role: coding
- node: tree/nodes/N0005_swin_promptseg
- inputs: node idea.md, memory/failure_modes.md, code/engine/train.py contract
- outputs: model.py (build_model), config.py; tree.json status -> coded; --smoke green
- notes: frozen timm swin_tiny_patch4_window7_224.ms_in22k; input_size=224; ~30.5M total (tight)

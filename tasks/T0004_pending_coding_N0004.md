# T0004_code_N0004_effnet_pyrmatch

- status: pending          # pending -> claimed_<agent> -> done | cancelled
- created: 2026-08-23T09:30:00+08:00
- role: coding
- node: tree/nodes/N0004_effnet_pyrmatch
- inputs: node idea.md, memory/failure_modes.md, code/engine/train.py contract
- outputs: model.py (build_model), config.py; tree.json status -> coded; --smoke green
- notes: frozen timm efficientnet_b0.ra4_in1k; total <=12M; input_size multiple of 32

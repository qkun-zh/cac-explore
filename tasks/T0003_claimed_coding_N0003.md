# T0003_code_N0003_convnext_xattn

- status: pending          # pending -> claimed_<agent> -> done | cancelled
- created: 2026-08-23T09:30:00+08:00
- role: coding
- node: tree/nodes/N0003_convnext_xattn
- inputs: node idea.md, memory/failure_modes.md, code/engine/train.py contract
- outputs: model.py (build_model), config.py; tree.json status -> coded; --smoke green
- notes: frozen timm convnext_nano.in12k; FPN+cross-attn decoder; attention at stride 16 for memory safety

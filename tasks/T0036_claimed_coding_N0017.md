# T0036_pending_coding_N0017

- status: pending          # pending -> claimed_<agent> -> done | cancelled
- created: 2026-08-23T15:53:42+08:00
- role: coding
- node: tree/nodes/N0017_dino_tailreweight
- inputs: tree/nodes/N0017_dino_tailreweight/idea.md, tree/nodes/N0010_dino_multilayer_long/{model.py,config.py}, memory/failure_modes.md
- outputs: tree/nodes/N0017_dino_tailreweight/{model.py,config.py}; flip tree.json status -> "coded"
- notes: Champion recipe VERBATIM from parent N0010 (frozen DINOv2-S reg4 taps(6,11), scalar gate,
  area-prompt, adapter768, MLP head, 392px). ONE change only: engine-level sample reweighting
  w_i = 1/max(count_i,1)**tail_exp with tail_reweight=true, tail_exp=0.5, normalized to mean 1 over
  batch, applied to per-image base loss (MSE density + count L1). Do NOT touch architecture.
  Smoke (--epochs 2) must pass before card rename to _done_.

# cac_uot — Clean UOT Counting

HF-only stack: `datasets` + `transformers` (DINOv3) + standard unbalanced OT as loss.
576 piles token-direct, prompt-gated, 40ep, batch8.

```
cac_uot/
  configs/        # GodConfig dataclass
  datasets/       # FSC147 HF parquet + processor
  models/
    backbone/     # DINOv3 HF backbone (frozen)
    prompt/       # Prompt gates (A/B)
    heads/        # PileHead (w,p)
    losses/       # UOT solver + repulsion + anchor
    god_model.py  # Assembly (DI)
  training/       # Trainer + metrics (closed/open readout)
```

Design: dependency inversion — `GodModel` depends on abstractions (`Backbone`, `PromptGate`, `PileHead`, `TransportSolver`), not concretions.

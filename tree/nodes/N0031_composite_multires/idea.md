# Idea — N0031_composite_multires (parent: N0027_norm_flip_swa)
## Delta vs parent: cfg.multires=True only ({392,518} epoch-parity alternation, engine P1 machinery from N0028 minus SCB-lite which was never validated).
## Rationale: F4 (GOD v6 §1) shows dense buckets improve monotonically with eval resolution but sparse buckets suffer U-shift; TRAINING-time exposure should make 518 features in-distribution so dense-side evidence rises without sparse damage. Tests H0042 mechanism properly (N0028 died E3, unresolved).
## Hypothesis (reuses registered H0042): IF {392,518} joint training THEN routed readout <=18.2 AND tail[500+) <=350. DISPROVED IF routed >=19.178 (banked fusion ceiling = zero progress) OR bulk degrades.
## Readout plan: banked routing (>=200->518) + hybrid variants from fusion_probe.py (19.033 best known).

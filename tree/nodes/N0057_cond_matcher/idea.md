# Idea — N0057_cond_matcher (parent: N0054_xscale_exemplar, frozen)

THE ONE untested structural class. Every prior loss since N0036 was a STACKED ADD-ON (gates, keys,
extra-scales, aux heads) that degraded under frozen 30ep. N0057 adds nothing — it REPLACES the
condenser with a strictly cheaper cosine-similarity prototype matcher, testing whether the learned
cross-attn condenser pays for its capacity at all.

## Change (structural REPLACEMENT, FROZEN, optimizer unchanged)
**Matcher Condenser (H0079, ~ -0.3M params).** Gate `use_matcher` inside Condenser. When on, the
MultiheadAttention + FFN + 2 LayerNorms are dropped; instead: q = proj_in(dense tokens), keys = L2-norm(exemplar
embeddings), w = softmax(q_norm · K^T / 0.1) per query->exemplar, agg = w·e (weighted exemplar summary),
out = proj(agg). No learned attention, no FFN. Identical surrounding head (FineFuser/Exemplar/GCA/Decoder
unchanged = N0054). Single-switch: `use_matcher=True` is the ONLY delta vs N0054.

## Why (grounding)
- "Less is more" is the dominant empirical theme: N0036-era modulators, N0055 keys, N0056 extra-scale all hurt.
- The condenser cross-attention is the largest untested capacity block; a simpler matcher is the cleanest
  falsification of "capacity matters" vs "capacity hurts" under short-horizon frozen training.
- If matcher ≈ N0054 → condenser capacity is wasteful (can shrink head, more margin to 32M). If matcher worse
  → cross-attn condenser is genuinely load-bearing; deliverable even more robustly locked.

## Hypothesis
**H0079** IF [condenser REPLACED by cosine-similarity prototype matcher (no MHA/FFN)] IN [frozen-N0054 base,
use_ddca=False], THEN [val MAE within +0.5 of N0054 (19.647; i.e. ≤20.15)] BECAUSE the learned cross-attn
capacity adds little under 30ep while the simpler matcher fits the same exemplar evidence with less to optimize.
DISPROVED IF [val MAE > 20.15 (regresses ≥ +0.5) OR matcher clearly worse than parent trajectory at same epoch].

## Gates
- R1 smoke: stub backbone, use_matcher on; params ≤32M; density (B,1,96,96); finite & drops. use_matcher=False
  restores N0054 attn path (single-switch).
- R2 30ep @384 frozen recipe. Compare vs N0054 (19.647). Early-stop if ep16+ ≥+1.5 worse than parent.
- R3 pluggability: toggling use_matcher touches only the Condenser module; no coupling to other components.

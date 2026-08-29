# Feedback — N0059_pom_morph (Quant)

**Run**: 30/30ep success, best 20.958 (E18), final 21.415, RMSE 76.06 vs champion 74.05 (+2.01).

**Computed deltas vs N0054 (19.647, given verbatim)**:
- E16 same-ep: **+2.13** (24.468 vs 22.338). Early-stop bar (≥+1.5) **FIRED**: 24.468 ≥ 23.838. Condition met but not acted on — run ran to completion; the immediate E17/E18 recovery would have been missed by an E16 stop.
- E18 same-ep: **−0.61** (20.958 vs 21.567). Only shared epoch (− of 13) where N0059 ≤ N0054.
- Floor: best 20.958 vs 19.647 = **+1.31**.
- Shared-epoch mean deficit ≈ **+1.62** (12/13 epochs worse; worst E15 +3.05, E20 +2.53).

**Gate fired: KILL (R4)**. Both prongs: best 20.958 ≥ 19.90, and ep16+ same-ep bar fired (E16 +2.13, again E20/E25/E27/E29/E30). NOT CONFIRM (<19.40) and NOT TIE (>19.90, by only 0.058 — but the 20.958 is a 1-epoch dip, so the margin is illusory; see below). **H0082 DISPROVED**: 20.958 ≥ 20.40 falsification threshold (+0.56 margin, non-marginal).

**Instability (flag=false is too lenient)**: E18 20.958 is a single-episode spike; E19 rebounds +1.40 to 22.358, and the tail E21–E30 oscillates 21.088–21.858 (≈0.77 band ≈ 3× the ±0.25 noise). The best-of-30 estimate is upward-lucky; typical late performance sits ≈21.4, i.e. +1.75 above champion — the KILL is more decisive than the best-episode number alone suggests.

**N=1 sufficiency**: KILL defensible at N=1 (idea.md R4/N=1 rule). 2nd seed mandatory only if first best < 19.40 — not met (20.958). The typical-vs-best gap (21.4 vs 20.958) can't move classification across two gates, so a 2nd seed is NOT required; a caveat is that the +1.31 floor is a 1-sample estimate with a favorable-sample bias.

**Operator-class attribution: STRONG.** Clean single-switch (use_pom): capacity matched (head 3.52M vs 3.50M, total 31.34M vs 31.32M), all 49 tokens kept (no part-pool), residual path matched (norm_first + residual), condenser/GCA/XScale/decoder untouched, frozen backbone, identical recipe, smoke-verified parametric identity when use_pom=False. The solo structural change — producer 2× self-attention → 2× PoM-PolyMorpher — is the causal agent of the +1.31 floor deficit, since no confound survives. **This re-frames N0058**: its failure was NOT purely capacity+pooling; the operator swap itself carries a real, now-isolated cost, and non-attention (moment+gated polynomial) mixing is confirmed under-expressive at n=49 in this regime.

**Verdict**: **KILL — H0082 DISPROVED (H0081 not confirmed)**; producer self-attention load-bearing; N0058's capacity/pool confounds subtracted, operator-class deficit confirmed ≈+1.3..+1.6.
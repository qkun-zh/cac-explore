# Diagnostic — N0058_pompart_exemplar (early-stopped E15)

## Root-cause verdict
**Model under-convergence** (PMOM head trains slowly, not converged-weak, not operator-crash).
No code bug in PMOM operator. No infra confound. Stop was justifiable on the KILL bar, though
the ep16+ early-stop gate is applied at E15 (=15<16) by a hair.

## Evidence

### Trajectory (val MAE / best / train loss)
- E07 25.511 / 25.511 / 4.900 · E10 24.679 / 24.679 / 4.346
- E13 23.151 / 23.151 / 4.063  ← best; +3.504 over 19.647
- E14 23.505 / 23.151 / 4.020 · E15 23.805 / 23.151 / 3.965

Train loss STILL descending at E15 (3.97, ~0.4/ep pace E13→E15); val MAE still noisy (23–28)
and best only improved at E13 — no convergence plateau. Not "converged weak"; the head is a
slow learner. Val oscillation amplitude (Δ5+ between adjacent epochs) ≈ convergence scale:
the PMOM head has not settled.

### Same-epoch vs parent
N0055 20.835, N0056 24.313, N0057 21.076 (E15) — N0058 23.151 is the second-worst crossing,
comparable to N0056 (+3.06). Parent N0054 at E15 ≈20.98 (given +2.17 gap). Parent's late drop
E18-27→19.65 came ONLY after it passed sub-21 at E17; N0058 never touched sub-21.3.

### Gate compliance (pre-registered, idea.md #R2)
- KILL ≥20.4: reached by E7 (25.51), never left the ≥20.4 zone through E15. Stop correct.
- ep16+ bar (≥21.147): eligible AT E16; E15 shows no breach yet, but MAE 23.1–23.8 with best at E13
  and no convergence → lethal-zone verdict sound; continuation adds only a formality the trajectory
  already decides. E15 stop CORRECT.

## Code-level check (PMOM) — CLEAN
- gate = Linear(768,16)+GELU+Linear(16,1) → softmax over 4 parts; m = cat([h,h*h]) 768-dim;
  shapes verified (parts (BK,4,384), Hs=a·m sum (768), moment_proj 768→256, +shape_mlp+xproj).
  Matches idea.md. No NaN: loss bounded 3.97–6.66 throughout; AMP stable; GCA bias negligible.
  use_pmom=False restores exact N0054 (smoke-proven). No operator defect — under-performance is
  genuine slow/weak convergence, not a gate/cat bug.

## Infra check — CONSISTENT
- run_dir /data/runs/N0058_pompart_exemplar identical in every RESULT; result.json (best 23.150863,
  E15, 852.7s) exactly matches log tail @08:38:46+08. Abrupt end = clean pkill on second attempt;
  pkill self-match trap (failure_modes L15) NOT triggered. No infra confound, no config mismatch.

## Should it have been kept?
No — pre-registered KILL bar ≥20.4 fired at E7 with no recovery. The "keep for late drop"
counterfactual (N0054 E18-27) is weak: parent dropped only after passing sub-21 at E17; N0058
shows no descent into that zone. E15 stop correct; H0080 NOT confirmed.

## failure_modes.md
No append — no genuinely new mode: slow frozen-head convergence already covered (L36); pkill
self-match already covered (L15, NOT triggered).
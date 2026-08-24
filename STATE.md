# STATE — Session 2026-08-24 (closed, distilled)

**Champion ckpt**: N0021_dino_partialft val MAE **20.438** @23.11M (`/data/runs/N0021_dino_partialft/best.pth`)
**Effective best**: **MAE 19.18 / RMSE 66.37** — routing readout (N̂@392≥200→518px), split-half validated
**New node**: N0028_scb_multires val MAE **~48.4 @E2** (23.55M, 23.11→+0.44M), SCB-lite γ=0 + multi-res {392/518} alternation

**Server**: UP (RTX3060 12GB idle) · creds: `local/address_and_password.md`

## Read `docs/DISTILLED_2026-08-24.md` FIRST — it replaces node-dir archaeology

Contains: confirmed/refuted levers with evidence, 5 key measured facts, and the three fully-specified open proposals:

- **P1 N0028_scb_multires** — SCB-lite exemplar gating (γ=0 init) + {392,518} joint training; target routed ≤15.9; H0040/H0042 bars pre-specified
- **P2 COIN R0** — physics kill-gate: Spearman ρ(similarity-map sum, count)>0.75, no training — **FAIL** (ρ=0.018)
- **P3 AXIOM-TTC R0** — drift audit on our own ckpt before any TTT work — **PASS** (Spearman 0.349, 0.426)

## Session ledger

- Nodes run: N0024 early-stopped (H0032 no cross-paradigm transfer), N0025 eval lab (H0033/34 refuted → mass-anchoring discovery), N0026 res sweep (H0035/36 ✓ routing), N0027 hygiene tie (H0037 band-disproved, H0039 SWA refuted)
- Hypotheses H0030–H0039 booked; index rebuilt; 3 code bugs found (input-norm missing→fixed in N0027, flip never on→flag exists, result.json last-ep headline→UNFIXED)
- New node N0028_scb_multires added: SCB-lite + multires training (smoke→40ep pending); tree status updated

## Distillation: dead nodes/cards archived to `tree/archive_2026-08-24/` & `tasks/archive/`

## Next queue

1. **P1 N0028_scb_multires** — monitor training (currently E2 @48.41 improving toward 15.9 target); if H0040/H0042 pass → R1 attribution arms; otherwise refute and back to champion lineage
2. P2/P3 R0 kill-gates are GPU-cheap; results recorded: COIN FAIL (ρ too low), AXIOM PASS (drift predictive) — proceed to R1 only if needed
3. If P1 wins: refit routing threshold split-half on new ckpt; then P1's R2 attribution arms
4. Champion lineage alternates: lr_mult {0.05,0.2}; blocks 9-11; TT-Norm eval-only (research-ranked #1 lever)
5. Unverified flags: our manifest train-size (3659 official vs 6591 mission text); DINOv3-S availability

## Rules refreshed this session

- Multi-angle ideation mandatory (math+physics lenses + champion keeper) — AGENTS.md Step 3
- Gates: novelty_check → check_hypothesis → calibration_report (never skip)
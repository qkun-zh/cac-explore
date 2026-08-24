# STATE — Session 2026-08-24 (closed, distilled)

**Champion ckpt**: N0021_dino_partialft val MAE **20.438** @23.11M (`/data/runs/N0021_dino_partialft/best.pth`)
**Effective best**: **MAE 19.18 / RMSE 66.37** — routing readout (N̂@392≥200→518px), split-half validated
**Server**: UP (RTX3060 12GB idle) · creds: `local/address_and_password.md`

## Read `docs/DISTILLED_2026-08-24.md` FIRST — it replaces node-dir archaeology
Contains: confirmed/refuted levers with evidence, 5 key measured facts, and the three fully-specified open proposals:
- **P1 N0028_scb_multires** — SCB-lite exemplar gating (γ=0 init) + {392,518} joint training; target routed ≤15.9; H0040/H0042 bars pre-specified
- **P2 COIN R0** — physics kill-gate: Spearman ρ(similarity-map sum, count)>0.75, no training
- **P3 AXIOM-TTC R0** — drift audit on our own ckpt before any TTT work

## Session ledger
- Nodes run: N0024 early-stopped (H0032 no cross-paradigm transfer), N0025 eval lab (H0033/34 refuted → mass-anchoring discovery), N0026 res sweep (H0035/36 ✓ routing), N0027 hygiene tie (H0037 band-disproved, H0039 SWA refuted)
- Hypotheses H0030–H0039 booked; index rebuilt; 3 code bugs found (input-norm missing→fixed in N0027, flip never on→flag exists, result.json last-ep headline→UNFIXED)
- Distillation: dead nodes/cards archived to `tree/archive_2026-08-24/` & `tasks/archive/`

## Next queue
1. Materialize P1 (integrator → coding → 45min train) — biggest number move available
2. P2/P3 R0 kill-gates are GPU-cheap; run while P1 trains
3. If P1 wins: refit routing threshold split-half on new ckpt; then P1's R2 attribution arms

## Rules refreshed this session
- Multi-angle ideation mandatory (math+physics lenses + champion keeper) — AGENTS.md Step 3
- Gates: novelty_check → check_hypothesis → calibration_report (never skip)

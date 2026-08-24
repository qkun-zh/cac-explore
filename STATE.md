# STATE — Session 2026-08-24

**Champion ckpt**: **N0021_dino_partialft** val MAE 20.44 @ 23.26M
**Effective best (eval-only routing, H0036)**: **MAE 19.18 / RMSE 66.37** — route N̂@392≥200→518 readout else 392; zero training
**Target**: ≤32M params, MAE ≤10 FSC147 test · Server: UP (RTX3060 12GB)

## Progress this session
1. Takeover + housekeeping; server rotated & onboarded (install_key.py)
2. N0024 ebc+partialFT early-stopped E16 (26.15 vs bar 22.80): H0032 doesn't transfer to EBC; conf 0.5175
3. 5-lens ideation (math/dynamics/counterintuitive/detail/physics) integrated
4. Verified 3 code bugs: missing DINOv2 input norm (fsc147.py:68 only /255); flip-aug never enabled; result.json headline = last-epoch not best
5. N0025 eval lab: H0033 TT-Norm REFUTED (exemplar mass ≈0.03–0.19, no per-object anchoring → 108 MAE); H0034 isotonic REFUTED (cross-fit hurts both halves); diagnostic: 75.9% SSE in 17 imgs N≥500
6. N0026 res sweep: H0035 PASS (448: RMSE −8, tail error ↓ monotonically with res); H0036 ROUTING PASS → new best
7. Hypotheses banked H0030–H0036 (conf: H0035 .585, H0036 .585, H0032 .517)

## Architecture (champion recipe)
```
Frozen DINOv2-S reg4 @392 → taps(6+11) → gate → Fourier+area prompt
→ adapter(384→768→384 d.15) → conv head → density [B,1,28,28]; N=Σρ
+ blocks10-11 unfrozen lr×0.1; 40ep bs8 lr1e-3 cosine AMP, 24min/GPU
+ ROUTING READOUT (no params): N̂@392≥200 → re-read at 518px
```

## Refuted (do NOT retry)
Full FT · EBC-paradigm transfer of partialFT · per-token gate · Huber · highres decoder-as-model · seqcount AR · tail-reweight · proto-iterative · scale-deform · point detect · mosaic-in-domain · TT-Norm gain calib (no mass anchoring) · global isotonic recalibration · trimmed-sum readout

## Next queue
1. N0027 hygiene retrain: input-norm fix (H0037 exp ≥0.5) + flip aug (H0038 exp ≥0.2) + SWA-lite piggyback; engine reports eval@392+448
2. Split-half confirmation of routing threshold before test deployment
3. If H0037 confirms: consider multi-res TRAINING (392+518 joint) targeting tail bucket
4. Later: exemplar-box averaging (3 boxes), log-VST head, drift-budget schedule

## Key Files
- Champion: `tree/nodes/N0021_dino_partialft/best.pth` (server) · Routing data: `tree/nodes/N0026_res_sweep/res_results.json`
- Eval labs: `scripts/eval_readout_lab.py`, `scripts/eval_res_sweep.py` · Engine: `code/engine/train.py`


## Progress this session
1. Takeover housekeeping: tree status fixes (fullft→failed), stale tasks archived
2. Causal feedback written for champion (missing third feedback) — timescale-separation account of why partial FT works where full FT collapses; lr_mult confounded with scope (falsifiers pre-registered)
3. Researcher report saved → `docs/research_notes_2026-08-24.md`. Key facts: no published ≤35M method beats ~test MAE 9–12; TT-Norm is biggest inference lever (CountGD −0.7 val /−1.3 test); mosaic HURTS in-domain FSC147 (MGCAC); backbone lr×0.1 is de facto recipe (MGCAC); DINOv3-S = open opportunity, unverified on HF-mirror
4. Synthesis done for champion: H0032 booked create+evidence, confidence 0.500→**0.575** (uncertain); index rebuilt (H0030/H0031/H0032); tree quality/avail/score populated; tested_hypotheses fixed H0030→H0032
5. Dual selection run: parent=**N0022_dino_ebc_fullft** (score .902), Q_t=[H0032, H0031]

## Architecture (champion recipe)
```
Frozen DINOv2-S reg4 @392 → taps(blocks 6+11) → scalar layer-gate
→ Fourier+area prompt → adapter(384→768→384, dropout 0.15)
→ conv head(384→128→1) → density [B,1,28,28]
+ blocks 10-11 unfrozen, lr = 0.1× head lr; 40ep bs8 lr1e-3 cosine AMP
```

## Confirmed Levers | Refuted (do NOT retry)
DINOv2-S substrate ✓ · multi-tap+area-prompt ✓ · partial FT ✓ ||
Full FT · per-token gate · Huber · highres decoder · seqcount AR ·
tail-reweight · proto-iterative · scale-deform · point detect · mosaic-in-domain

## Next queue (needs server)
1. Rotate server → preflight
2. Idea card ready: N0022-lineage EBC + partial-FT swap (tests H0032 in new paradigm context) per selector Q_t
3. Champion lineage alternates: lr_mult {0.05,0.2}; blocks 9-11; TT-Norm eval-only (research-ranked #1 lever)
4. Unverified flags: our manifest train-size (3659 official vs 6591 mission text); DINOv3-S availability

## Key Files
- Champion: `tree/nodes/N0021_dino_partialft/` (synthesis complete, all 3 feedbacks)
- Engine: `code/engine/train.py` · Selection: `code/selection/select_next.py`
- Archive: `tree/archive_gen0_5/` · Research: `docs/research_notes_2026-08-24.md`

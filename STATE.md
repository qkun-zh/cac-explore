# STATE — Session 2026-08-24 (degraded mode: server DOWN)

**Champion**: **N0021_dino_partialft — val MAE 20.44 / RMSE 83.06** @ 23.26M, 1441s
**Target**: ≤32M params, MAE ≤10 on FSC147 **test** (relaxed from ≤4 this session)
**Blockers**: `ssh cac-server` TIMEOUT — user must rotate/restore server

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

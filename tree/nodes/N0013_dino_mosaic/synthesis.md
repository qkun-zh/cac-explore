# synthesis.md — N0013_dino_mosaic

## Verdict
**Augreg package REFUTED AS WRITTEN.** Best MAE **22.404**@E38 (final 22.761, RMSE/MAE 3.62) vs parent N0010 **21.531**@E26 → **+0.87 (+4.1%) worse**, breaching the pre-registered disproof bar (MAE>21.53). Clean run: 40/40ep, success, 23.11M, no OOM/instability, 1410s (+10.6% wall-clock from Python-loop jitter). **Attribution CONFOUNDED**: jitter(p=.5)+bbox(.15), dropout .1→.2, wd ×5 moved together — "augmentation hurts" vs "over-regularization of a ~2.1M head" unresolved; only the bundle is priced at −0.87 (causal: plausibly near seed noise). Regularization demonstrably engaged (train floor 8.26 vs 7.78; best epoch 38>26 sub-claim holds) yet val floor got WORSE → parent's plateau was not regularization-limited. Tail mechanism DEAD: photometric+bbox jitter injects zero count variance, so RMSE/MAE stayed 3.62 (parent 3.63); SeqCount+'s benefit needs paste+sum high-count mass, which mosaic-lite omits.

## Result Summary
best 22.404@E38 · final 22.761/82.48 · ratio 3.62 · 1410s · ahead of parent through E20 (24.38 vs 25.50@E13), then missed the E26 low-lr dip entirely (24.74@E26) — jitter+reg sped early fit, blunted fine-phase gains.

## Quality Gate (7 dims)
| dim | rating | note |
|---|---|---|
| mechanistic | 0.8 | why-tail-untouched and why-reg-engaged both explained; 3-lever confound caps attribution |
| scoped | 0.9 | exact champion clone, 392px fixed, orthogonal to N0012 |
| predictive | 0.85 | bars pre-registered; MAE≤20 missed +12%, ratio<3.4 failed (3.62), best>26 held |
| falsifiable | 0.95 | disproof criterion met cleanly on full schedule |
| novel | 0.4 | mosaic-lite proxy already shown insufficient; true mosaic untested |
| transferable | 0.6 | revert-regs + early-stop-rule lessons carry to N0014-N0016 |
| actionable | 0.8 | jitter-only ablation, vectorization, gate logging all concrete |
Gate PASS (evidence bookable). Node scores: quality ≈ **0.40**, score ≈ **0.55**.

## Dedup & Contradiction Resolution
- **H0024 contradicts**: quant 0.70 / qual 0.80 / causal "not decisive" → **0.75** (explicit bar breached on clean identical-recipe run; capped below 0.85 for the 3-lever confound + small effect size).
- **H0017**: quant neutral 0.20 / qual neutral 0.30 / causal supports 0.50 → **supports 0.30** (stability-under-jitter signal real and thrice-replicated; but gate weights never logged → "gate balance persists" unverifiable, so weak positive only).
- causal's indirect **H0022-supports 0.40 NOT booked**: banked H0022 (tail-reweight) was not exercised by N0013; ratio 3.62 merely reconfirms the outlier problem. Left untouched (n_tested 0) per quant rec.
- qual's proposed low-dose annealed-jitter follow-up (H0025 candidate): deferred to Idea agent — not booked this round.
- Bookkeeping defect confirmed by all reviewers: idea.md:20 mislabels the augreg claim "H0022" (= banked tail-reweight). Booked as NEW **H0024**; H0022 confidence untouched. Add failure_modes entry: idea agents must derive hyp ids from memory/index.json max-id, never reuse booked ids.

## Bookings → memory/hypotheses.jsonl (ts 2026-08-23T15:10:00+08:00)
1. `create H0024` c=0.5 — IF photometric+bbox jitter augmentation (jitter_prob 0.5) + dropout 0.2 + wd 5e-4 IN FSC147 champion recipe, THEN MAE ≤20.0 AND RMSE/MAE<3.4, BECAUSE augmented diversity + regularization improve generalization. DISPROVED IF MAE>21.53 OR ratio≥3.63.
2. `evidence H0024 contradicts` w=0.75 — 22.404 vs 21.53; regularized train floor higher yet val worse; tail ratio unchanged 3.62; attribution confounded across jitter/dropout/wd.
3. `evidence H0017 supports` w=0.30 — dual-tap stack stable under jitter+regularization, transfers.

## Tested Hypotheses & Recommended Scores (Lead applies)
- tested_hypotheses: **["H0024","H0017"]**
- tree.json N0013_dino_mosaic → status `"synthesized"`, best_metric 22.404, quality 0.40, score 0.55

## Recommendations for Lead / Next Nodes
1. Do NOT carry augreg bundle into N0014 as-is; if kept at 518px: keep jitter, REVERT dropout→0.1 wd→1e-4, hard precondition re-anchored cosine + timeout ≥55min + grad-accum bs8 (else skip merge).
2. Jitter-only ablation (dropout 0.1/wd 1e-4 unchanged) is the single salvage run if an augreg track survives gen-5.
3. Tighten early-stop rule: fire only if best-so-far gap ≥+1.5 held 2 consecutive epochs after E16 (instantaneous deltas would have killed a run reaching 22.40).
4. Log softmax(layer_logits) each eval epoch; vectorize jitter ops (+~135s/run); cap bbox scale w,h≤0.95·S.

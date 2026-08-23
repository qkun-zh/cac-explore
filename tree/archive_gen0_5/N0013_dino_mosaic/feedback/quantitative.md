# feedback/quantitative.md — N0013_dino_mosaic

## reasoning
- Headline: best MAE 22.404@E38 / RMSE 81.08 vs parent N0010 21.531@E26 / 75.91 → **+0.87 (+4.1%) WORSE**; final-epoch 22.761 vs 22.607 also worse. Success status, 40/40ep, 23.11M, no OOM/instability, 1410s (+10.6% vs 1275s — per-image Python-loop jitter overhead).
- Falsification of the augreg hypothesis (labeled "H0022" in idea.md): primary bar "DISPROVED IF MAE > 21.53" → **22.404 FAILS**. Secondary clauses pass vacuously: best epoch 38 > 26 ✓, RMSE/MAE 3.618 < 3.63 ✓ (final 3.624 ≈ parent final 3.626) — catastrophic-tail ratio unchanged.
- Trajectory: best-so-far tracked ahead of parent through E20 (24.38 vs 25.50 @E13; 23.59 vs 24.32 @E18), then parent's E26=21.53 broke away; N0013 never dipped below 22.40. Per-epoch deltas noisy (±2–4 MAE), no sustained lead after E20.
- Overfit attack FAILED: train loss @E40 8.259 vs parent 7.781 (+6%) — dropout0.2 + wd5e-4 + jitter did engage (less train fit) — yet val floor got worse. ⇒ Parent's val plateau was NOT regularization-limited; head capacity / feature informativeness is the likelier binding constraint.
- SEVERE CONFOUND: 3 levers changed together vs parent (jitter stack, dropout 0.1→0.2, wd ×5). Failure cannot be attributed to jitter; over-regularizing a ~2.1M-param trainable head is equally consistent with the data.
- Gate-balance reuse claim (H0017): train.log logs no layer_logits/gate values → unverifiable; only indirect signal is stable training under jitter.
- Bookkeeping defect: banked H0022 = tail-aware reweighting (N0011 lineage, n_tested 0). The augreg hypothesis was never booked; next free ID = H0024.

## actionable_feedback
1. Do NOT carry the augreg package into N0014 highres+augreg merge as-is: lever is unvalidated AND confounded. If salvaged, run jitter-only ablation (dropout 0.1, wd 1e-4, jitter unchanged) — one GPU run, judge on best-MAE vs 21.53.
2. N0014/N0015/N0016 should revert to champion regs (dropout 0.1, wd 1e-4) so resolution remains the sole variable.
3. Tail direction unchanged (ratio ~3.62 across N0010-N0013): augmentation does not touch outliers; bank-H0022 reweighting / count-calibration stays the open tail lever.
4. Log softmax(layer_logits) each eval epoch in future runs — cheap; makes gate claims falsifiable.
5. Vectorize jitter (batched tensor ops) if reused: +135s/run is pure loop overhead.
6. Early-stop rule validated: N0013 ran ~+1.5–3.2 behind parent E21–E26; killing it would have lost nothing.

## hypothesis_updates
- hypothesis_id: H0024 (NEW — book idea.md augreg pkg; it is mislabeled "H0022" there)
  evidence_type: contradicts
  strength: 0.7
  reasoning: disproof bar met (22.404 > 21.53) on an identical-recipe clone, same schedule/seed regime; strength capped below 0.85 because the 3-lever confound blocks attributing the regression to jitter vs the reg bump.
- hypothesis_id: H0017
  evidence_type: neutral
  strength: 0.2
  reasoning: multi-tap stack trains stably under jitter (no instability flag; steady late gains E33–E38), but gate weights are unlogged so "gate stays balanced" is unverifiable; net MAE regression gives no positive signal.
- hypothesis_id: H0022
  evidence_type: neutral
  strength: 0.1
  reasoning: bank-H0022 (tail reweighting) was not exercised by N0013; the run only reconfirms the outlier problem persists (ratio 3.62). Leave confidence at 0.5.

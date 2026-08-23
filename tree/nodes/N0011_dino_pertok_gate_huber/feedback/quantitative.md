# feedback/quantitative.md — N0011_dino_pertok_gate_huber

## reasoning
Result: best MAE 26.678 @E34 (result.json), final 26.895/RMSE 94.337, 40ep/1299s, params 23.16M, no OOM/instability.
Parent N0010: best 21.531, final 22.607/RMSE 81.974 (ratio 81.97/22.61=3.63). Delta = **+5.15 (+24%)** where idea.md promised −1.0; no epoch (E13–E40, all 26.7–31.9) ever approached 21.5.
Bars: H0019 needed ≤20.53 → missed by 6.15. H0020 needed RMSE/MAE <3.0, disproved if >3.5 → observed 93.93/26.68=3.52 @best, 3.51 final — marginally trips its own disproof bar AND fails target; tail ratio barely moved vs parent 3.63 (−3%).
Training dynamics: train loss 5.00→3.44 over E13–E40 (smooth, monotone); val plateaued ~27–28 from E13, best E34, then flat. Healthy optimization + flat val ⇒ representation/objective ceiling, not an optimization artifact. RMSE stayed ≥93 every epoch — outlier tail never shrank.
Confound warning: node bundles THREE deltas vs parent (model.py:30-42 per-token gate MLP; config.py:15-16 Huber δ=5 replacing MSE; config.py:10 count-w 0.3 vs parent 1.0). idea.md's "reverts count-w to isolate Huber" actually de-isolates vs parent — part of the +5.15 may be the lost count-w=1.0 signal (cf. H0018), so per-lever attribution is partial.

## actionable_feedback
- Keep scalar layer gate (2 params) in champion recipe; per-token softmax gating over 2 highly-correlated post-proj taps adds ~50K trainable params + variance with no gain — drop this line unless a future multi-tap set (≥3 distant layers, e.g. N0016 4-tap) reopens real depth choice per location.
- Do not reuse Huber δ=5 here (engine support stays unused): gradient capping did not reduce the tail; if outliers matter later, test count-aware reweighting or log/sqrt density transform in ISOLATION on the champion base.
- Process rule: ≤1 mechanism change per node when parent is champion; keep parent's count-w fixed. This node burned 1300s GPU on an unattributable 3-way delta.
- Executor note: same-epoch rule (≥+1.5 worse at ep16+) would have killed this at E16 (28.57 vs parent trajectory) saving ~770s — apply strictly next refutation-class run.

## hypothesis_updates
- H0019: evidence_type=contradicts, strength=0.80, reasoning: per-token gate failed its ≥1.0-improvement bar by 6.15 MAE in-architecture; direction opposite to prediction. Deducted from 0.9 because count-w 1.0→0.3 + Huber ran concurrently (3-way confound), so the gate's isolated harm is unproven even though the deployed combination decisively fails.
- H0020: evidence_type=contradicts, strength=0.70, reasoning: RMSE/MAE 3.52>3.5 trips the written DISPROVED criterion (marginally) and misses the <3.0 target by wide margin; tail ratio ≈ parent 3.63 despite capped gradients — mechanism produced none of the predicted outlier suppression.
- H0018: evidence_type=neutral, strength=0.25, reasoning: N0011 (count-w=0.3) regressed everywhere yet ratio ticked 3.63→3.51, cutting both ways for "count-w=1.0 improves ratio"; Huber+gate confound makes this run unusable as H0018 evidence either direction.

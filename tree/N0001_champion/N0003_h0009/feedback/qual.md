# QUAL — N0003_h0009 (H0009 channel gate)

## 1. What the gate mechanically does
A per-image, per-channel gain on the exemplar tokens before they enter the Condenser's cross-attention.

- `channel_gate = nn.Linear(D=128, embed_dim=256)` — `model.py:160`.
- `gate = torch.sigmoid(self.channel_gate(fine.mean(dim=(2,3))))` — `model.py:169`: global average-pool of the 128-ch fused fine map → one 256-dim logit vector per image → sigmoid → a gate in (0,1) per channel.
- `e = e * gate.unsqueeze(1)` — `model.py:170`: the gate (B,256) is broadcast across **all K exemplars** of the image; every exemplar token shares the same 256-dim channel scale. 33K new params, trainable-only head, frozen backbone.

Plain terms: the condenser's exemplar keys/values are *one per object*; this gate is *one per image*, applied identically to every object in that image. It reweights channel subspaces of the exemplar representation as a function of the whole-image fine features, i.e. a self-modulated feature emphasis.

Important nuance vs the hypothesis's BECAUSE: the gate is **image-global, not per-exemplar**. It cannot differentially steer weights between exemplars of the same image; it can only amplify/suppress whole channels shared by all of them. The mechanism does not deliver exactly what the hypothesis's mechanism text claims.

## 2. Learning behavior in the log (per-epoch val/mae from tb scalars)
Curve: 168.2 → 71.2 → 49.0 → 40.0 → 34.6 → 31.4 → 29.1 → 27.6 → 26.5 → 25.5 → 24.9 → 24.3 → 24.0 → 23.9 → 23.8 → 23.6 → 23.4 → 23.3 → 23.1 → 23.1 → 23.0 → 23.0 → 22.9 → 22.9 → 22.8 → 22.8 → 22.7 → 22.7 → 22.6 → 22.625 → 22.614 → **22.608**.

- Monotone: val improves **every single epoch**, 32/32, no plateau window. This stands out against the parent's own documented behavior (config notes: parent val plateaued ~22–25 across E14–25 with train loss still falling). Here val never flattens mid-run.
- Early phase (E1–E8) is a normal fast head-first descent (168→27.6); no instability, no NaN. train/loss_epoch falls 30.7 → 2.68 monotonically, with a few loss spikes in the first epochs (e.g. step 249: 22.6, step 1799: 71.2) typical of the count-term mixing.
- Reading: the addition neither destabilizes nor obviously accelerates the parent's trajectory — it looks **delay/converge-shaped**: the same monotone descent pushed out to the schedule's edge rather than a faster climb.

## 3. "Best at the very last epoch, budget nearly tripped" — dynamics reading
- Training stopped at 1718.9s / 1800s hard ceiling; 32/32 epochs, `budget_hit:false` but essentially wall-to-wall. The ep32 checkpoint is a **snapshot taken while the curve was still moving downward** — per-epoch deltas in the last 5 epochs are still −0.016, −0.011, −0.035, −0.011, −0.007, i.e. the slope had not flattened at cutoff.
- Two readings are consistent: (a) the gate buys real late-training expressiveness and this run was simply truncated; (b) the added layer just re-slowed convergence, and the descent would have bu<ted into a plateau within a few more epochs. Nothing in the log discriminates.
- Signal that matters: val/rmse bottoms at 88.76 @ E26 and **then reverses upward** to 89.78 @ E32 while val/mae keeps falling. The final stretch is not uniformly better — MAE gains are concentrated on the common case while the tail (squared, large-error) worsens. So the "best at last epoch" is partly a re-shaping that trades tail error for bulk error, not clean shared improvement.
- Implication for the Δ: once the seeded parent number arrives, the gate's 22.608 sits on a steep, still-descending slope, so the comparison is between two points on a live curve; the trailing −0.007/−0.011 per-epoch steps mean the Δ in the last epochs alone wobbles by ±a few hundredths, and the parent's own seeded value carries the same kind of schedule-position uncertainty. The observable gap between this run and the parent's migrated 19.647 is far wider than any such wobble — but that migrated number is unseeded/pre-regime and is NOT a valid comparator; only the pending seeded parent run is. Directionally the pre-registered 0.30 bar is a long way off; the trailing per-epoch steps are a tenth or less of that margin's scale.

## 4. Mechanism-specific risks for the causal review
1. **Not per-exemplar.** The hypothesis's BECAUSE ("per-exemplar channel-gain … steering scale-consistent exemplar weights") is not what the code does; one gate vector is shared across all objects in the image (`model.py:170`, unsqueeze over dim 1). Any effect is a global channel-emphasis effect, one step removed from count-dependent per-object attention.
2. **No identity-preserving init.** `nn.Linear` default init puts logits near 0 → sigmoid ≈ 0.5 at t=0, i.e. all exemplar channels are **halved from the first step** relative to the parent's untouched exemplar pathway. The condenser suddenly sees protagonists key/value magnitudes ~0.5× while `d_sim` norms and the `proj_in` (128→256) input are unchanged. A systematic scale perturbation the head must relearn — a paid early-cost, and a confound when crediting the gate for any later gain.
3. **Saturation / hard channel pruning.** Sigmoid over an unbounded linear output allows channel gates to drive toward 0, permanently amputating that channel from *every* exemplar. The late-run RMSE rise alongside MAE fall is consistent with the gate pruning channels that fit the majority class at the cost of rare/tail appearance — a failure mode distinct from a slow-converging readout.
4. **Input is a moving target + shared readout with GCA.** The gate's input is the fine GAP — produced by the very head being trained, so its distribution drifts; the 128→256 readout is perpetually retuned to a shifting input. Worse, GCA also reads that same `fine.mean(dim=(2,3))` and the gated `e_mean` is computed *after* gating (`model.py:173`) and feeds the count-aux — so gating and the count aux draw from the same source and the gate rescales what GCA reads. Credit for any observed effect is split across two coupled paths; a clean ablation story will be harder.
5. **EMA / eval**: given a moving-target input, a weight-EMA of the gate and its fast current weights can disagree at eval time; here best.pth is a snapshot, but the checkpoint chosen is the last epoch on a still-moving slope, so its in-run eval and post-run eval both carry the "mid-slope" caveat.

## 5. Budget stance
Observation, not a tuning recommendation: the 1800s budget is exhausted and 32/32 epochs is the pre-registered minimum; this node records a single test event. Nothing here argues for extending or refining this node.

## Bottom line
A mechanically coherent but hypothesis-mismatched retrofit: an image-global (not per-exemplar) sigmoid channel gate that enters the run by halving exemplar magnitudes, produces smooth monotone but delayed convergence with a "best at last epoch" that is partly a bulk-vs-tail trade (MAE falls while RMSE rises through the final ~6 epochs), and stops mid-slope at the wall-clock limit. The plausible readings — genuine late expressiveness vs deferred/slowed convergence — are indistinguishable from this log alone; the pending seeded parent comparison is the only fair yardstick, and the observable trailing-step dynamics suggest a wider scheduling-position error bar around any Δ.
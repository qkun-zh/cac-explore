# N0004_h0003 — Causal feedback (H0003 `use_padapt`)

**Verdict: genuine mechanism failure, not an artifact.** The adapter trained
*fully* and *actively* (its zero-init output gate moved to ‖W‖=2.4946), and it
made the head strictly worse on **both** train and val: the child's training
loss plateaus at ≈5.56 from ep12 on, while its parent's training loss keeps
falling to 2.618. The best val MAE (25.5661 @ep13) loses to the live parent by
+3.002 and misses the pre-registered bar (22.26) by +3.306. The dominant cause
is the adapter corrupting the Condenser K/V by injecting an unmodulated,
image-content residual into the only 3 vectors that define the class — i.e., the
survey's named F3 failure mode (survey_mechanisms.md:84-87; idea.md:165,173) —
not overfitting and not undertraining. Code-path, diff, checksum, and curve
evidence all rule out a harness bug.

Evidence artifacts: server `result.json`
(`/data/cac/tree/N0001_champion/N0002_h0001/N0004_h0003/result.json`) and
`run/latest/best.pth`; parent `result.json:1-14`; TB scalars from
`run/latest/tb/t/events.out.tfevents*`; checkpoint probe run under
`/data/miniconda/envs/cac/bin/python` (CPU) reading `ckpt["model"]`
(275 tensors; top keys `['epoch','model','best_mae']`).

---

## 1. Measured adapter (best.pth) — is it active or dead?

`model.py:251-252` zero-inits `padapt.out.weight`/`.bias`, so the *only* way the
adapter can affect the forward is for that gate to move off zero. It did:

| tensor | shape | measured ‖·‖_F | init reference | read |
|---|---|---|---|---|
| `head.padapt.out.weight` | (256,64) | **2.49461** | **0.0** (model.py:251) | gate moved decisively off zero — adapter **live** |
| `head.padapt.out.bias` | (256,) | **0.290492** | 0.0 (model.py:252) | live |
| `head.padapt.fproj.weight` | (64,192) | 4.5811 | 4.6188 (Linear default) | −0.8%: rotated, not blown up |
| `head.padapt.fproj.bias` | (64,) | 0.254562 | ≈0 (default) | small |
| `head.padapt.eproj.weight` | (64,256) | 5.08468 | 4.6188 | +10.1% |
| `head.padapt.eproj.bias` | (64,) | 0.241259 | ≈0 | small |
| `head.padapt.attn.in_proj_weight` | (192,64) | 9.19888 | 9.7980 (xavier) | −6.1% |
| `head.padapt.attn.in_proj_bias` | (192,) | 0.399095 | 0 (constant) | moved |
| `head.padapt.attn.out_proj.weight` | (64,64) | 4.0908 | 4.6188 | −11.4% |
| `head.padapt.attn.out_proj.bias` | (64,) | 0.118185 | 0 | moved |
| `head.padapt.norm.weight` | (64,) | 7.06412 | 8.0 (ones) | −11.7% |
| `head.padapt.norm.bias` | (64,) | 0.339975 | 0 | moved |

Init references are analytic (Linear default = kaiming_uniform a=√5 ⇒
‖W‖=√(fan_out/3); MHA `in_proj` = xavier ⇒ ‖W‖=√(96)=9.798; LayerNorm weight =
ones ⇒ √64=8). Conclusion: the whole adapter is engaged, and the one tensor that
gates the residual path (`out`) sits at 2.4946 — 27% of what a *default*
(non-zero) init of a (256,64) Linear would be (√(256/3)=9.238). So the
`idea.md:172` escape clause ("a refute with `out` norm ≈0 means the optimizer
discarded the adapter") **does not apply**: the adapter was used, and it hurt.
The module's forward residual is `out(a)` with `a` a softmax attention over all
2304 h2 tokens (model.py:256-261), and `out` has full generative scale; a
non-negligible residual is being written into `e_cond`.

Reference — the confirmed H0001 path also re-trained to a *different* solution,
so the child is not merely "parent + adapter":

| tensor | child | parent (`.../N0002_h0001/run/latest/best.pth`) |
|---|---|---|
| `head.simprior.temp` | 0.0203301 | 0.0393157 (matches N0002 run event, journal/events.jsonl: "temp 0.07->0.0393") |
| `head.simprior.out.weight` | 0.877079 | 0.587438 (matches N0002 run event, journal/events.jsonl: "out-proj norm 0.5874") |
| `head.simprior.kproj.weight` | 6.56189 | 7.35467 |
| `head.simprior.qproj.weight` | 5.26119 | 5.73144 |

The child's SimPrior got *sharper* (temp ÷1.93) and *stronger* (out ×1.49). Its
forward reads the original `e` (model.py:181), so this is not a direct adapter
input; it changed because gradients through the shared decoder/Condenser shifted.
That is expected and harmless, but it is evidence that the run is a genuinely
different head, not a mechanical copy of the parent.

---

## 2. Curve shape — where the harm is born

Extracted from the child TB event file (val MAE, val RMSE, train loss_epoch) and
the parent event file:

| ep | child val MAE | parent val MAE | child train loss | parent train loss |
|---|---|---|---|---|
| 1 | 152.7835 | 153.2874 | 41.560 | 38.024 |
| 2 | 69.2195 | 70.0534 | 9.003 | 8.653 |
| 3 | 49.6918 | 48.7882 | 8.366 | 8.409 |
| 10 | 25.9875 | 25.4092 | 5.628 | 5.492 |
| 12 | 25.5786 | 24.2563 | 5.563 | 5.598 |
| **13** | **25.5661 (best)** | 23.9073 | 5.582 | 5.226 |
| 14 | 25.5921 | 23.5598 | 5.588 | 5.068 |
| 20 | 26.0180 | 22.9341 | 5.579 | 3.986 |
| 30 | 26.1628 | **22.5641 (best)** | 5.591 | 2.704 |
| 32 | 26.1668 | 22.5705 | 5.589 | 2.618 |

Child RMSE mirrors MAE: min 96.0988 @ep13, rising monotonically to 96.8000 @ep32
(parent best val MAE 22.5641 @ep30, RMSE ratio ≈3.76 at child best).

Three facts drive the causal reading:

1. **The child is worse than the parent from ep3 onward** (49.69 vs 48.79, and
   every epoch after). The adapter is only useful at ep1-2 (child marginally
   ahead before `out` moves). The mechanism degrades the solution at *all* times,
   not just late.
2. **The bulk of the 3.00 gap exists before the val peak.** At ep13 the child is
   already +1.659 vs the parent's same-epoch 23.9073. Post-peak drift (25.5661 →
   26.1668) adds only +0.60 over 19 epochs. So overfitting can explain at most
   ~20% of the loss; ~80% is present at the best epoch.
3. **Train loss tells the same story.** The child bottoms at 5.563 @ep12 and is
   flat (5.56-5.59) thereafter; the parent continues down to 2.618. A model that
   is *overfitting* fits the train set well and fails val. This child fits the
   train set **worse** than the parent and fails val — the adapter suppressed the
   head's training fit, not merely its generalization.

---

## 3. Failure-mode adjudication

**(a) Overfitting from +62k params, augment=false, val peaking ep13 — NOT the
primary cause (secondary at most).** The classic overfit signature (train ↓,
val ↑) is absent: train loss is flat while val drifts, and train loss never
reaches the parent's. Ruled out further by timing: 1.66 of the 3.00 gap is
already present at ep13. The post-ep13 val drift contributes ≤0.60. Also, +62,208
trainable params (idea.md:138) is ~1.8% of the ~3.5M head — far too small to
cause a 3.0 MAE regression by capacity alone.

**(b) Adapted prototypes as Condenser K/V disrupt the matching geometry and
never recover — SUPPORTED (observable level).** The code inserts the adapted `e`
on exactly one edge, `e_cond → Condenser` (model.py:172-175), and the Condenser's
`attn(norm1(tok), e_cond, e_cond)` is the sole 3-way matcher (parent
model.py:121-123). A large, moving key/value distribution is a moving target for
the Condenser; once `out` leaves zero the joint optimum shifts, and the child
train-loss plateau (5.56 vs parent 2.62) is exactly the signature of settling
into a worse basin from which it cannot recover. The immediate divergence at ep3
indicates the disruption is early and structural, not a late-training artifact.

**(c) No LOCA objectness modulation / normalization ⇒ background hallucination —
LEADING mechanism, not separable from (b) in one run.** The module attends over
*all* 2304 h2 tokens with a plain softmax and no foreground/objectness mask or
spatial normalization (model.py:256-261). `survey_mechanisms.md:84-87` names this
precisely ("adapting e to the query can hallucinate target appearance from
background (LOCA needs modulated/object-normalized attention)"), and `idea.md:173`
prices it in ("unmodulated blending pulls background statistics into the
prototype"). LOCA's own ablation (no-att 10.24→11.99; replacing the first OPE
cross-attention with a plain sum costs +5% MAE, idea.md:49) says the modulation is
load-bearing. Our measured active `out` (2.4946) plus a worse-not-better fit is
consistent with a background-bearing residual corrupting the 3 class vectors; a
foreground-modulated variant is the specific test that would separate (c) from
(b), and it was not run.

**(d) Mechanism fine, undertrained / needs more data — REJECTED for steps;
not testable for data in one run.** More steps strictly hurt: best ep13, then
monotonic increase to ep32; train loss is already flat by ep12. The failure is
maximal at the model's own best epoch, so no training-length tweak addresses it.
"More data" cannot be excluded from one run, but it is a weak rescue for a
mechanism that is already 1.66-3.00 MAE worse at its optimum; and the per-image
diagnostic (perimage_diagnostic.md:58, top 1% of images = 79.4% of ΣΔ²) says the
error is concentrated in dense/high-count images — a matching-side perturbation
that corrupts the prototype is the wrong lever to fix those, per first_principles
S1/S3 (first_principles.md:178-207).

**Honest limits.** One run; no attention-map dump, no per-image adapter on/off
intervention, no `n_steps=1` or modulated-attention control. Therefore (b) and
(c) are *not* separately identified — they are the same failure viewed at the
optimization level (b) and the content level (c). What *is* pinned by measurement:
the adapter is active (out ≠ 0), the harm predates overfitting, and the harm is
present on train and val alike.

---

## 4. Is this a coding/harness artifact? — RULED OUT

- **Only intended lines changed.** `diff parent/model.py child/model.py` = the
  `PrototypeAdapter` class, the `use_padapt` flag (model.py:162), the
  `e_cond` insertion (model.py:172-175), and the post-SimPrior construction
  (model.py:289-291). Nothing else. `diff config.toml` = exactly one added line,
  `use_padapt = true` (config.toml:28); loss/optimizer/schedule/data unchanged.
- **Flag-off is a no-op by construction.** With `use_padapt=false`,
  `e_cond = e` (model.py:172) and the `if` is skipped, so
  `self.cond(fmap, e_cond)` (model.py:175) is byte-identical to the parent's
  `self.cond(fmap, e)` (parent model.py:167). `PrototypeAdapter` is constructed
  **after** `SimPrior` (model.py:282-283, 290-291), so every parent init draw
  keeps its RNG position (AGENTS §5 rule 13); the extra params receive no
  gradient when unused, and AdamW skips `p.grad is None` params, so they cannot
  affect a flag-off run. The parent N0002 run therefore *is* the flag-off
  reference and it reproduced to 2e-6 (perimage_diagnostic.md:31).
- **Run integrity.** Child `result.json`: 32/32 epochs, `best_mae`
  25.56608963, `best_epoch` 13, `budget_hit` false, 1767.6s of the 1800s cap —
  no timeout/truncation. `config_sha256 769ddfa05d25dfa9` /
  `model_sha256 93c221209869fe69` differ from the parent's
  `4d4c9baec459f657` / `611be2a9d70768b5`, confirming the intended child was run;
  the active `out` norm (§1) proves the new path executed rather than silently
  falling back to flag-off.

Artifact **ruled out**.

---

## 5. Per AGENTS §11 — a valid re-encode, or drop the family?

H0003 is REFUTED (25.5661 vs parent 22.5641, bar 22.26; ledger H0003 created at
conf 0.5, memory/hypotheses.jsonl:5). Under AGENTS.md:147-149 any re-encode must
book **contradicts**-evidence on H0003 **and** carry a *new* falsifier.

- **Not valid (same mechanism, no new falsifier):** fewer steps / `n_steps=1`,
  stronger weight decay, longer schedule, or "more data." §2/§3 already show the
  failure is not step-, capacity-, or overfit-limited (train loss is *higher* and
  flat; val best is at ep13). These would silently re-run a refuted mechanism.
- **Valid next test (only candidate worth a node):** replace the unmodulated
  attention with a LOCA-style objectness-modulated one — mask/weight the keys by
  a frozen similarity foreground estimate (or restrict `F` to top-sim h2 tokens)
  — booked as `contradicts` on H0003 with the falsifier: **"final val MAE ≤ 22.26
  at seed 20260830 under canonical 32ep/1800s, vs live parent 22.5641."** This
  tests (c) directly and is a different mechanism (modulated adaptation), so it
  satisfies §11.
- **Recommendation on spend:** on current evidence the pre-Condenser prototype-
  adaptation family is **not worth another GPU node ahead of F1/F4/F6**. It has
  now failed once with an active adapter that worsened both train and val; the
  survey ranks F1 (dense similarity prior) → F4 → F3/F6 (survey_mechanisms.md:10-17,
  174-175), and the heavy-tailed error (perimage_diagnostic.md:58) points at
  explicit matching/evidence readouts, not a re-parameterized quantized K/V.
  If it is retried, retry **only** with the modulated-attention variant above and
  the new falsifier booked; otherwise close the family.

---

## 6. One-line causal summary

`use_padapt` did not under- or over-fit: it inserted a live, unmodulated,
image-content residual into the Condenser's 3 prototype keys/values
(`out`=2.4946 from zero-init), which prevented the head from reaching the
parent's training fit (train loss plateau 5.56 vs parent 2.62) and settled val at
25.5661 — worse than parent at every epoch from ep3, +3.002 behind the live
parent and +3.306 behind the bar. Code, config diff, checksum, and the flag-off
path all reproduce the parent; the failure is in the mechanism, and the
unmodulated attention (no LOCA objectness normalization) is the leading
explanation.

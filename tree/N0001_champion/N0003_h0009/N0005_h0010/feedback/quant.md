# Quantitative Feedback — N0005_h0010 (frozen-random channel-gate input control)

Role: Quantitative. Node: `tree/N0001_champion/N0003_h0009/N0005_h0010/`
Config delta vs parent N0003_h0009: `use_exemplar_gate=true` (H0010) and
`use_const_gate_input=true` (H0011) added; channel-gate input is a frozen
random vector (`frozen_gate_input`, seeded, `requires_grad_(false)`) instead of
the content-conditioned `fine.mean(dim=(2,3))`.

All numbers below are read directly from the three `result.json` files, the
node `config.toml`, `model.py`, and `discovery prove` ledger output. No SSH, no
run logs.

## 1. The numbers

| Quantities | Value | Source |
|---|---|---|
| Child best val MAE (N0005_h0010) | **22.810184478759766** @ ep32 | child result.json |
| Parent best val MAE (N0003_h0009, content-conditioned gate) | **22.607566833496094** @ ep32 | parent result.json |
| Grandparent best val MAE (N0001_champion seed 20260830) | **21.458858489990234** @ ep32 | root result.json |
| Δ (child − parent) | **+0.202617645263672** | computed |
| Relative penalty | **+0.2026176 / 22.6075668 ≈ 0.90%** | computed |
| Δ (child − grandparent) | **+1.351325988769532** | computed |
| Channel gate size | `nn.Linear(128, 256)` = 128·256 + 256 = **33,024 params** | node model.py |
| Exemplar gate size | `nn.Linear(256, 1)` = 257 params | node model.py |
| config_sha256 / model_sha256 | f922e3663ad3efb5 / 1c80d0133f69ea63 | child result.json |

**Bar checks.**
- H0011 falsifier: gate benefit is attributed to *feature conditioning* only if
  severing the conditioning makes final val MAE **≥ 0.30 worse** than 22.608;
  DISPROVED if within 0.30. Observed Δ = **+0.2026 < 0.30** → falsifier **NOT
  met** → H0011 **DISPROVED** (inert capacity). Ledger already recorded
  `contradicts` w=1.00 on N0005_h0010 for H0011.
- H0010 falsifier (per-exemplar gate ≥ 0.30 better than gate-removed 22.608 →
  target ≤ 22.308): observed 22.810 is **+0.2026 worse** than 22.608, missing
  the bar by a wide margin. Ledger already recorded `neutral` on N0005_h0010
  (confounded), correctly — the per-exemplar gate alone is not isolable here.

## 2. Interpretation — severing gate conditioning cost only +0.20

Removing **all information-carrying input** to a 33,024-param channel gate —
replacing the conditioned fine-feature summary with a frozen, per-sample
constant random vector — degrades val MAE by only **+0.203 absolute ≈ 0.9%
relative**. On a same-lineage, same-seed, same-epoch comparison this is close
to noise-level drift. Two consistent readings:

1. **The gate's conditioning stream is not load-bearing.** The gate maps
   `sigmoid(Linear(·))` over 128→256; with a constant input it reduces to a
   per-channel output gate constant across the batch and spatial domain, and
   the model loses almost nothing. The Δ that survives (+0.203) is best read as
   the cost of the gate no longer being able to do *anything* per-sample, not
   as durable channel-attention signal.
2. **Scale check against the lineage.** The parent's gate was worth ~−1.15 vs
   grandparent per H0009 refutation (Δ = −1.149 vs champion, from the H0009
   ledger event); that entire benefit survives with the gate input frozen.
   I.e., a content-free gate reproduces essentially all of N0003's gain over
   champion. The H0011 falsifier's premise — that the *conditioning* is what
   pays — is the part disproved; the onboarded fully-connected gate per se is
   neither confirmed nor refuted by this node.

**Honesty caveat on H0009 linkage:** the −1.149 H0009 figure is cited from the
ledger event text, not re-derived here; it is consistent with, not re-measured
by, this node.

## 3. Confound — exemplar gate (H0010) is co-present

The child config turns on **two** switches at once: `use_exemplar_gate=true`
and `use_const_gate_input=true`. The +0.2026 delta vs parent therefore bundles

- the effect of freezing the channel-gate input (H0011), and
- any effect of the per-exemplar reliability gate (H0010, identity init, bias
  +3).

If the exemplar gate helps, the true channel-input-severing penalty is
**larger** than +0.2026; if it hurts, the penalty is **smaller**. The direction
and magnitude of that coupling are unobservable from this node alone. The
+0.2026 should therefore be treated as an *upper/lower-confounded* band, not a
clean measurement. This is why the H0010 event on this node was booked neutral
rather than contradicts. (N0004_h0010, the other H0010 probe, is likewise
joint-confounded with H0009: 23.382 vs seeded parent 21.459, per ledger.)

## 4. Budget & training dynamics (result.json fields only)

- `budget_hit: false`, `elapsed_s: 1705.8` (raw `elapsed`: 1700.40 s) — run
  finished comfortably under the τ_max 1800 s ceiling; no `_BudgetStop`, status
  is a clean `done`.
- `epochs: 32`, `n_epochs_done: 32`, `best_epoch: 32` — the best checkpoint is
  the **final** epoch. Same for parent (best_epoch 32) and grandparent
  (best_epoch 32). So the cosine schedule ran full-length in all three and the
  model was at its best when training stopped; no mid-run optimum was lost to
  the budget.
- Both `config_sha256` (f922e3663ad3efb5) and `model_sha256`
  (1c80d0133f69ea63) are recorded, so the "same config" / side-by-side claim
  vs parent (a5ea5d5daf44acbb / 976e313c971d2531) is re-checkable.
- Notably, the child's best_mae (22.810) is attained at the *last* epoch, and
  the parent's too — the +0.2026 gap is at the train-to-the-end operating
  point, not a staleness artifact.

## 5. Verdict

| Hypothesis | Bar (pre-registered) | Observed | Met? | Ledger action taken |
|---|---|---|---|---|
| **H0011** — channel-gate benefit attributable to *feature conditioning* of its input | final val MAE **≥ 0.30 worse** than 22.608 (otherwise DISPROVED) | Δ = **+0.2026** worse (22.810 vs 22.608), rel. **≈0.90%** | **No — falsifier not met → DISPROVED (inert capacity)** | `contradicts` w=1.00 on N0005_h0010 (already in ledger) |
| **H0010** — per-exemplar gate ≥ 0.30 better than gate-removed 22.608 | target ≤ 22.308 | 22.810 (but co-present with frozen-random input → **confounded**) | Not cleanly assessable here | `neutral` on N0005_h0010 (already in ledger) |

Bottom line for synthesis: on the N0003 lineage, a content-free (frozen-random)
channel-gate input costs only ~0.9% relative MAE — inside the 0.30 falsifier —
so H0011 is disproved as a *conditioning-driven* mechanism; the gate's value,
if any, does not come from what is fed into it. The H0010 co-presence keeps the
exact magnitude uncertain, and H0010 needs a clean (single-switch) probe before
its own bar can be adjudicated.

**No ledger/confidence numbers were written by this agent; events above are
verbatim from `discovery prove` output. Nothing committed or pushed.**
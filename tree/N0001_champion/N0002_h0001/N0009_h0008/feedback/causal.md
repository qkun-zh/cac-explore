# Causal feedback — N0009_h0008 (H0008 `use_hires` HiResDetail)

Parent: N0002_h0001 (EMA val MAE 22.5641 @ep30). Child outcome: **26.4535 @ep31**
(`result.json` code ok, 32/32ep, `budget_hit` true, elapsed 1809.7 s).
Margin **+3.89** vs parent; falsifier bar 22.26 missed by **4.19**. Verdict on the
booked dual bar: **REFUTED (used-and-harmful)** — not a clean null, not a harness
failure. All numbers below were read off the server checkpoint / tb logs on
2026-09-20, not recollected.

## 1. What was tested (one switch)

Post-decoder detail residual `HiResDetail` (~15.4k, verified 15,433 params in
`best.pth`; node total 31.365M ≤ 32M): `Guide(h2: 192→24 1x1+GN)` → bilinear ×2
to 96×96 → `cat([L.detach(), g])` (25ch) → `Ref1/Ref2 (3×3+GN+GELU, 24ch)` →
zero-init `Head (24→1 1×1)` → residual `r` (zeros at init); output
`dens = softplus(L + r)`. GCA path untouched, decoder `in_ch` untouched (192).
Design claim under test: decoder-side placement + detached inputs + private
parameters leave the confirmed simprior readout isolated, so a miss reads as
mechanism rejection, never readout corruption.

## 2. Checkpoint probe (server `run/latest/best.pth`, epoch 31)

Hires — **engaged, not null**:
- `head.weight` norm **0.2250** (init 0), `head.bias` 0.0461.
- `ref1.0.weight` 3.4001, `ref2.0.weight` 3.1044, `guide.0.weight` 3.6963;
  GN scales 4.50–4.65 (trained, not init).
- Functional (real val batch, 4 imgs): L mean −6.44 / std 1.73 (deep softplus
  tail); **r mean −0.502 / RMS 0.899 / max|·| 5.35** — same order as L's
  useful range, i.e. the residual re-centers the softplus operating point.
- Count direction on that batch: base 42.46 → hires 34.20 (**−19.5%**); all 4
  images suppressed (22.2→14.2, 31.8→24.3, 31.6→17.9, 84.2→80.4 vs gt
  13/15/19/82). The module learned a **net-suppressive bias**, not separable
  peak splits/fills.

Simprior readout — **isolation FAILED**:
- Child: qproj **6.7138** / kproj **6.8004** / temp **0.00334**.
- Parent `best.pth` (ep30, re-read): qproj **5.7314** / kproj **7.3547** / temp
  **0.03932** — reproduces the card's working point (5.73/7.35/0.0393).
- Drift: q +17%, k −7.5%, **temp collapsed 11.8× to the clamp floor** (1e-3).
  Softmax over K is ~12× sharper than the confirmed operating point: brittle
  top-1 evidence. `simprior.out` norm 0.587→0.933 (+59%) — the readout
  re-amplified around the collapsed temp.
- Decoder norm 35.82 vs 34.75 (+3%), GCA 6.09 vs 6.19 (−1.6%): backbone of the
  head is intact — the damage is localized to hires + simprior temp/out.

## 3. Learning dynamics (tb scalars, child vs parent)

- Child `train/loss_epoch`: 48.94 (ep1) → **5.96 by ep8, then flat 5.96–6.00
  to ep32**. Parent: 38.02 → 6.15 → 4.52 → 3.20 → **2.62**. Final train ratio
  **2.3×** — early optimization stall, not late overfit.
- Child `val/mae`: 141.5 (ep1) → 28.32 (ep8) → 26.54 → 26.46 → **26.45**;
  flat within 0.08 from ep16. Parent descends monotonically to 22.56. The gap
  opens by ep8 (+2.0) and never closes — interference from the start.
- Signature is distinct in the family: not the N0008 NaN wall, not the N0007
  shared-projection catastrophe (48), not the N0006 mild +1.02. This is a
  **mid-severity stall (+3.89)** with a healthy-looking plateau.

## 4. Harness ruled out

- Checksums re-verified on server: `config.toml` sha256
  `cb9ed415419598f2…` ✓, `model.py` sha256 `dcf9ce12b2cec929…` ✓ — match the
  `result.json` record (`config_sha256 cb9ed415419598f2`,
  `model_sha256 dcf9ce12b2cec929`).
- Server config diff vs parent is **exactly one switch**: `+ use_hires = true`.
  `hparams.yaml` confirms canonical protocol: seed 20260830, augment false,
  deterministic true, AdamW/cosine/AMP/MSE+0.4-L1 identical.
- Run completed **32/32 epochs**, best @ep31 on a flat ep16–31 plateau
  (Δ0.08) — the +9.7 s budget overrun (1809.7 vs 1800, same marginal class as
  N0006's +9.8 s) truncated nothing load-bearing. `timeout` status is
  administrative, not a verdict threat.

## 5. Mechanism verdict

**H0008 REFUTED as used-and-harmful.** The card's isolation premise is falsified
by this run: `detach()` blocks a direct second gradient path into simprior
projections, but it does **not** isolate the readout, because the residual
re-centers the shared loss operating point — `softplus(L+r)` rescales every
gradient flowing back through L into the decoder/condenser/simprior stack.
Joint optimization then drags temp to the floor (0.0393→0.0033) and the readout
re-amplifies around it, while hires itself settles into suppression (−19.5% on
the probe batch) rather than the hypothesized peak sharpening. The optimizer
stalls at 2.3× parent train loss from ep8, so the +3.89 MAE is the joint
equilibrium of this coupling, not an unlucky seed or a budget artifact.

Family update: the §11 ban on *shared projections / second grad paths* is
insufficient — **post-decoder additives that shift the softplus/logit operating
point couple into the confirmed readout through the loss even with detached
inputs and private parameters**. Any future decoder-side proposal must
pre-register a temp-trajectory guard (e.g. temp within 2× of parent through
ep8) and a residual-scale guard (r RMS ≪ L std at ep1–2), else it risks the
same attractor. Suggested evidence: `contradicts` on H0008, weight w=0.90
(+3.89 used-and-harmful with readout-coupling signature, temp collapse 11.8×,
train stall 2.3×, harness clean).

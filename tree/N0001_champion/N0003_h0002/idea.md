# N0003_h0002 — idea (booked H0002; drafted as H0003)

> **Booking header — mechanical record only, not a content change.**
> Node: **N0003_h0002** · Parent: **N0001_champion** · Booked hypothesis: **H0002** (drafted as H0003 pre-booking) · Switch: `use_gca_cal` (config default `false`) · Falsifier bar: final val MAE **≤ 22.99** at seed 20260830 (live parent 23.293 @ep26, canonical protocol 32ep/1800s).
> The hypothesis line in §5 is byte-identical to the booked text (`/tmp/book_gcacal.txt`); its embedded scope string retains the draft node name verbatim per booking.

---

# H0002 — Learnable signed global-count coupling (`use_gca_cal`)

- **Hypothesis id:** H0002 · **Child node:** N0003_h0002 (child of N0001_champion) · **Switch:** `use_gca_cal` (config default `false`)
- **One-line intent:** Mass-calibrate the load-bearing GCA aux by making its global count coupling learnable and signed, starting exactly from the parent's forward at init.

## 2. Multi-angle reasoning

### (a) Pure-mathematics lens — what the current map can and cannot represent

Parent forward (`tree/N0001_champion/model.py:179-183`):

```
z     = gca(cat(GAP(fine), e_mean))          # pre-activation logit, (B,)
n_aux = softplus(z)  >= 0
bias  = (n_aux / (Hf*Wf)) * 0.02             # Hf=Wf=96, so /(9216), broadcast to every cell
out["density"] = dens + bias                 # dens >= 0 cell-wise (softplus head, model.py:139-140)
```

The global path is the affine map `bias(n_aux) = c·n_aux/N` with **fixed** `c = 0.02 > 0`, `N = 9216`, over a **non-negative** argument. Consequences:

1. **One-sided range.** `bias ∈ [0, ∞)` per image. The model can only *add* mass through the global path; "this image is overcounted, subtract δ" is unrepresentable here and must be routed through local density suppression in the decoder, which couples localization to calibration.
2. **Monotone Jacobian.** `∂bias/∂z = c·sigmoid(z)/N ≥ 0` everywhere: every direction in GCA-logit space can only push the count up. The sign degree of freedom does not exist.
3. **Starved trust.** `c = 0.02` means the aux contributes on the order of 2% of a full-count-scale correction (first_principles §2 S4 predicts the bias term is <5% of Σdensity at the parent). The GCA MLP gets gradient only through this 0.02-scaled sum term, since `n_aux` has **no direct loss** (`src/cac/models/pl_module.py:76-78` — only `out["density"]` feeds `counting_loss`).

Proposed map (init-equivalent, §4):

```
n_cal = s1 * softplus(z) + s2 * z + b,   s1 init 0.02, s2 init 0, b init 0
bias  = n_cal / (Hf*Wf)
```

At init this is algebraically the parent (`n_cal = 0.02·n_aux`). Once training moves, with `s2 ≠ 0` (or `s1 < 0`, or `b < 0`) the range is all of ℝ: the global anchor can add *and* subtract. The linear branch additionally gives the GCA MLP a direct, non-saturating gradient path (`∂bias/∂z ⊇ s2/N`) even where `sigmoid(z) ≈ 0`, i.e. where the softplus branch is dead. Optimization starts from a proven point and moves only on gradient signal — safe exploration around the parent.

### (b) Champion-lineage lens — file:line evidence, heavy tail, count error IS MAE

- **GCA is load-bearing.** `tree/N0001_champion/README.md:11`: GCA-only 20.60, ~1.6 better than no-GCA. The channel carries useful signal yet is deliberately near-detached (2%, one-sided) — exactly the profile of a component whose *calibration*, not existence, is the open question. This card is the direct execution of first_principles **R3** (`use_gca_signed_scale`, rank 3) and **Cheap-2** (§5).
- **No direct supervision on the aux.** `pl_module.py:20-32` upsamples `dens` 96→384 sum-conservingly and applies MSE + `0.4·L1(sum, count)` (`pl_module.py:31`); `pl_module.py:76-78` feeds only `out["density"]` to the loss. So today the GCA scale `0.02` is a human prior, never optimized against anything — first_principles §1.3/S4.
- **Heavy tail ⇒ global additive correction is arithmetically sufficient.** Live parent: EMA val MAE **23.293 @ep26**, RMSE ≈ 90.8 ⇒ RMSE/MAE ≈ **3.90** (vs ~1.25 Gaussian), so a few catastrophic images dominate (first_principles §1.3). With N=1286 val images, Σ|Δ| ≈ 29,955 counts and the bar (0.30 MAE) is **386 counts** — one GT≈2000 image improved 20% clears it alone (inference, not measurement; D1/D3 dumps would confirm direction).
- **Count error IS MAE.** Val metric sums the density map (`pl_module.py:86-100`); a per-image uniform additive bias in count units acts *directly* on the scored quantity without touching localization. A signed global term is the minimal-capacity intervention on the scored quantity.
- **Untouched load-bearing interfaces.** hs(2,3) readout + cross-attn Condenser stay intact (`model.py:97-108, 121-125`); XScale stays on (`config.toml:25`); contract `build_model(cfg) → forward → {"density", ...}` unchanged.

### (c) Counter-intuitive / low-cost lens

The fix is not more capacity but **letting the existing global head subtract**: +3 scalar parameters (~+12 bytes, 0.00001% of the 32M budget), zero new layers, negligible FLOPs, init-identical forward so the worst case is ≈ the parent (modulo 3-scalar optimizer dynamics). The learned values double as the verdict readout: collapse-to-init proves the coupling was already optimal, which is itself a publishable negative. No loss/schedule/eval change; no protocol risk; smoke-identical.

## 3. Citations (RULE 15 satisfied by the two research reports; URLs below verify claims)

- **CounTR** — test-time exemplar-mass normalisation is one of its largest single gains (val 17.40→13.13, Table 2) plus test-time cropping for tiny exemplars (§3.4): global/exemplar-level count calibration moves MAE by *points*, not tenths. <https://arxiv.org/abs/2208.13721> · confidence **H** for "calibration matters", **M** for transfer (theirs is eval-time, ours is learned train-time).
- **first_principles R3 + Cheap-2** (`local/research/first_principles.md`, §3 R3 / §5 Cheap-2): this card's direct parent — learnable signed GCA coupling at +1 param, falsifier "learned scale stays within ±10% of 0.02". Confidence **H** (same codebase, same parent number 23.293).
- **survey §5 three-subproblem decomposition** (`local/research/survey_mechanisms.md`): with a frozen backbone the head is readout+calibration; mass calibration (leg 2) is unhandled except by GCA's global anchor; mechanisms changing *where comparison happens* dominate post-processing. Confidence **M** (interface reasoning, unmeasured on N0001).
- **LOCA OPE ablation** (+6.0 val MAE removed, 10.24→16.24): conditioning-side changes move val MAE at point scale. <https://arxiv.org/abs/2211.08217> · confidence **M** (different backbone/interface; direction-of-effect only).
- **SWA / Mean Teacher** (weight-averaging helps generalization): <https://arxiv.org/abs/1803.05407>, <https://arxiv.org/abs/1703.01780> · confidence **L** here — cited only to bound what this card is *not* (no training-dynamics change).
- **Champion README ablations** (`tree/N0001_champion/README.md:7-14`): GCA+, XScale+, frozen hs(2,3), condenser load-bearing; DDCA/RGA/extra-summaries/unfreeze/final-layer killed — this card touches none of the killed families. Confidence **H** (local lineage fact).

## 4. Exact implementation spec for the Coding Agent (child node N0003_h0002 only)

**Parent reference.** Copy `tree/N0001_champion/model.py` verbatim, then apply exactly the delta below. Parent GCA is `model.py:171-183`; parent wiring is `Counter.__init__ model.py:186-197` and `Counter.forward model.py:204-217`. Parent config `tree/N0001_champion/config.toml` (note `use_gca = true`, line 23); child config = parent config + one key.

**Config delta (child `config.toml` only):**
```toml
use_gca = true     # unchanged from parent
use_gca_cal = true # NEW key; default false (see below) — child run sets true
```
Default-handling rule: every reader uses `_get(cfg, "use_gca_cal", False)`, so the parent config (key absent) takes the hard-coded 0.02 path bit-identically.

**Constructor order (AGENTS.md rule 13 — append-only RNG order).**
`GCA.__init__` signature becomes `GCA(D, d_model, use_gca_cal=False)` (default keeps any parent-style call working). Body order:
1. Build `self.gca = nn.Sequential(nn.Linear(D + d_model, 64), nn.GELU(), nn.Linear(64, 1))` **exactly as parent** — same layer order, same shapes — followed by the parent's zero-init of the last layer (`nn.init.zeros_` on `self.gca[-1].weight/bias`). No existing layer is added, removed, resized, or reordered.
2. **After** all parent modules, create exactly three new learnables (plain `nn.Parameter` from constants — **no RNG consumption**, so shared-module init draws stay bit-identical to the parent's):
```python
self.use_gca_cal = bool(use_gca_cal)
self.s_pos = nn.Parameter(torch.tensor(0.02))  # s1: scale on the softplus branch (free sign)
self.s_lin = nn.Parameter(torch.tensor(0.0))   # s2: scale on the signed linear branch
self.b_cal = nn.Parameter(torch.tensor(0.0))   # b:  global offset, in count units
```
`Counter.__init__` is otherwise identical: backbone → head → (if `use_gca`) `GCA(D, embed_dim, use_gca_cal=_get(cfg, "use_gca_cal", False))` at the same position. If `use_gca` is false, GCA is absent and `use_gca_cal` is inert (document, do not branch elsewhere).

**Exact forward math (`GCA.forward`, parent-equivalent at init):**
```python
def forward(self, fine, e_mean, Hf, Wf):
    gap = fine.mean(dim=(2, 3))
    z = self.gca(torch.cat([gap, e_mean], 1)).squeeze(1)  # pre-activation logit (NEW name; same tensor parent applied softplus to)
    n_aux = F.softplus(z)                                 # parent quantity, kept for out["n_aux"]
    if not self.use_gca_cal:
        bias = (n_aux / float(Hf * Wf)).view(-1, 1, 1, 1) * 0.02   # parent path, untouched
    else:
        n_cal = self.s_pos * n_aux + self.s_lin * z + self.b_cal  # signed calibrated count
        bias = (n_cal / float(Hf * Wf)).view(-1, 1, 1, 1)
    return n_aux, bias
```
At init (`s_pos=0.02, s_lin=0, b_cal=0`): `n_cal = 0.02·n_aux` ⇒ bias is mathematically the parent's (up to fp associativity of the reordered multiply — "as close as init-equivalence gets"). `Counter.forward` is unchanged: `out["density"] = dens + bias; out["n_aux"] = n_aux` (n_aux keeps its non-negative logging semantics; the signed correction lives only in `bias`). Only `out["density"]` feeds the loss — contract preserved; XScale / hs(2,3) / Condenser untouched.

**Param arithmetic.** +3 scalars (float32) = **+3 params** (≈ +12 bytes). Head 3,504,643 → +3; total ≈ 31.320003M ≤ 32M (680k headroom, first_principles §1.1). AdamW picks the new params up automatically via `model.parameters()` (`pl_module.py:56`); EMA shadow includes them automatically (`ema.py:23-27` iterates `named_parameters`). No loss/schedule/eval/seed change; canonical protocol (augment=false, EMA eval, seed 20260830, 32ep/1800s) applies unmodified.

**RNG-order statement.** All parent constructors run first in identical order with identical shapes; the only new objects are three constant-initialized `nn.Parameter`s (no `randn`/`empty`/layer init draws). Data stream stays pinned to `config.seed` (`runner.py:61-62`, datamodule seed wiring), zero dropout anywhere, smoke-fit quirk identical for parent and child (first_principles §1.4). A `use_gca_cal=false` child must reproduce the parent trajectory bit-identically (modulo the 3 inert scalars in the state dict).

**No-repeat-register check.** Touches none of AGENTS.md §11: no DDCA, no extra spatial summaries/RGA, no readout/backbone change, no unfreeze, no pre-condenser exemplar gating — the delta is a decoder-side global calibration of the existing GCA aux (first_principles R3, explicitly live).

## 5. Exact hypothesis text (booked verbatim by the Lead — single line)

```
IF learnable signed global-count coupling via use_gca_cal IN child node N0004_h0003 of N0001_champion trained under the canonical protocol at seed 20260830 for 32ep/1800s, THEN final EMA val MAE falls to 22.99 or lower versus the live parent 23.293 at ep26, BECAUSE replacing the hard-coded non-negative 0.02 GCA bias with a learned signed scale plus a signed linear correction lets the load-bearing global count anchor add mass on undercounted heavy-tail images and subtract mass on overcounted ones where count error equals MAE. DISPROVED IF final EMA val MAE exceeds 22.99.
```

## 6. Falsification reading (primary bar + learned-parameter criterion)

**Primary bar (canonical protocol, seed 20260830, 32ep/1800s).** Live parent: final EMA val MAE **23.293 @ep26** (`tree/N0001_champion/result.json:5-6`). Bar: child final EMA val MAE **≤ 22.99** (≥0.30 lower). **DISPROVED IF final EMA val MAE exceeds 22.99** — same-seed paired contrast per AGENTS.md §6 (precision ~0.02 claimed; plateau noise ~0.05–0.2 observed, first_principles §6.2 — the residual risk is inherited, no new seeds allowed).

**Learned-parameter criterion (null/negative diagnostics).** `best.pth` stores `{"epoch", "model": state_dict, "best_mae"}` (`calls/best_checkpoint.py:21-29`), saved from `on_validation_epoch_end` while the EMA shadow is swapped in (`calls/ema.py:46-63`) — so the stored weights are the EMA averages that produced the val number. Read them with:
```python
import torch
ck = torch.load("run/latest/best.pth", map_location="cpu")["model"]
print({k: float(ck[k]) for k in ["gca.s_pos", "gca.s_lin", "gca.b_cal"]})
```
- **Null (coupling was already optimal):** `|s_pos − 0.02|/0.02 < 0.10` **AND** `|s_lin| < 1e-3` **AND** `|b_cal| < 0.5` (count units) ⇒ the optimizer left the calibration at its init point; the mechanism is falsified on parameters even if MAE wiggles within noise.
- **Negative:** final MAE worse than parent, or `s_pos → ~0` (the head detached its own global path), or large-magnitude `s_lin/b_cal` with val RMSE up ⇒ the signed path is harmful or overfits the tail; book as contradicts-evidence, do not rescue.
- **Moved-but-missed:** params moved well outside the null box yet MAE > 22.99 ⇒ still DISPROVED on the registered bar (no outcome shopping, AGENTS.md rule 1); the movement is evidence for a follow-up card, not a rescue of this one.
- **Caveat:** EMA (decay 0.999, ≈1000-step trailing average, `runner.py:77`) attenuates late-epoch parameter movement, so small deviations read from `best.pth` understate the raw trajectory — the null box above is evaluated on the EMA values as stored; a borderline null should be confirmed against the tensorboard `train/loss_epoch` + val curve shape before claiming "already optimal".

**Open risks.** (1) A uniform per-image bias cannot fix spatial misallocation — if the error is localization rather than calibration, expect a clean null. (2) Large negative `n_cal` on near-empty images could push cells negative and spike the L1 term early; init-at-parent plus cosine/AdamW should contain this, but a diverging `train/loss` in epochs 1–3 is an abort-and-report signal, not a tuning invitation. (3) Single-seed verdict inherits the harness noise budget (§6.2 of first_principles); the parameter criterion is the hedge against lucky/wild MAE draws.

## Booked hypothesis set (machine-readable)

1. **H0002** — IF learnable signed global-count coupling via use_gca_cal IN child node N0004_h0003 of N0001_champion trained under the canonical protocol at seed 20260830 for 32ep/1800s, THEN final EMA val MAE falls to 22.99 or lower versus the live parent 23.293 at ep26, BECAUSE replacing the hard-coded non-negative 0.02 GCA bias with a learned signed scale plus a signed linear correction lets the load-bearing global count anchor add mass on undercounted heavy-tail images and subtract mass on overcounted ones where count error equals MAE. DISPROVED IF final EMA val MAE exceeds 22.99.

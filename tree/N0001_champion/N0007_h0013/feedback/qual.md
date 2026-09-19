# Qualitative feedback — N0007_h0013 (H0013 solo)

Verdict: H0013 disproved. Child best val MAE **22.4545 @ep32** vs champion **21.4589**
(Δ = +0.9956 child-worse; pre-registered support bar ≤21.1589, `idea.md:7`,
`result.json:4-9`). Run completed 32/32 epochs, budget not hit (1708.1 s), ledger
`contradicts w=1.0` → H0013 conf 0.400 (`memory/hypotheses.jsonl:44`).
Sources read: child/parent `model.py`, `config.toml`, `idea.md`, `result.json`,
`run/latest/exec.log:551`, and the tensorboard scalars under `run/latest/tb/t`
(both child and champion `run/latest/tb/t`, tags `val/mae`, `val/rmse`,
`train/loss_epoch`).

## 1. Mechanism fidelity — the code does what H0013 says, but the claim's premise is inverted

**What is replaced.** Exactly one call: the parent's first
`F.interpolate(self.top(h3), scale_factor=2, mode="bilinear", align_corners=False)`
(`tree/N0001_champion/model.py:68`) becomes `top = self.top(h3)` then
`top = self.top_up(top)` when the switch is on (`N0007_h0013/model.py:80-82`).
The second/final bilinear upsample (`N0007_h0013/model.py:90`, parent `:74`
48→96≈S/4) is **untouched in both**. This matches the IF clause, which scopes the
replacement to "the fixed bilinear upsampling that brings the coarse backbone
readout to the fused resolution" (`idea.md:7`) — the coarse-branch merge, not the
density-grid upsample.

**The block.** `self.top_up = Sequential(Conv2d(d_fine, 4*d_fine, 3, padding=1,
groups=d_fine), PixelShuffle(2))` (`N0007_h0013/model.py:74-77`), built only when
`use_subpixel_up` is true (`:64`). With `d_fine=128` (`config.toml:20`) this is a
depthwise conv [512,1,3,3] + bias[512] = **5,120 params** — correct arithmetic, and
0.15% of the 3.5 M trainable params in `exec.log:31-33`. Grouping `groups=128`
means strictly per-channel kernels (no cross-channel mixing), i.e. an
ESPCN-style depth-to-space block that is faithful to the hypothesis text
(depth-to-space, learned, per-channel).

**Scale.** The replaced bilinear doubled h3's grid to match h2's: h3 (coarse) →
2·h3, i.e. the fused 1/8-input grid (H3→2·H3, here 24→48 at input 384; inferred
from the required `torch.cat([lat(h2), top])` at `model.py:85` and the
`Hf = Wf = S//4` concat at `:183`). Output resolution is identical to bilinear;
the claim "same resolution" is accurate.

**Receptive field / smoothing — the load-bearing change.** Bilinear scale-2 with
`align_corners=False` is a **fixed, non-negative, partition-of-unity 2×2 support
filter**: every output is a convex combination of ≤4 input pixels, weights sum to
1, and the operator is a separable triangular low-pass. The learned depthwise
3×3 conv gives each PixelShuffle parity its own independent signed kernel over a
**3×3 support** (9 taps, 36 free weights/channel vs bilinear's fixed neighborhood),
with no sign/sum constraints. Two consequences the hypothesis got backwards:

- **Mass/aliasing prior is removed, not "detail" added.** The coarse branch is the
  stride-16 (aliased) feature; bilinear was the only fixed low-pass in the coarse
  merge (the remaining convs are learned; the final upsample at `:90` is still
  bilinear in both). An unconstrained signed 3×3 per parity can amplify
  parity-alternating high-frequency components of h3 (checkerboard-family risk),
  and there is no sum-to-one constraint preserving local density mass. The only
  count anchors are global: the scalar `loss_count_weight=0.4` L1 and the GCA bias
  (`model.py:196-200`, `config.toml:49`); neither constrains *where* mass lands.
  Against smooth adaptive-Gaussian GT (`gauss_sigma=1.5`, `config.toml:22`),
  high-frequency density is a pure MSE liability.
- **The "detail" that is claimed to help is re-smoothed anyway** by the still-fixed
  bilinear at `:90` before the decoder head, so the hypothesis's stated payoff
  could only have come through the fused feature, where it in fact regressed.

So the run is a faithful test of the stated mechanism; the result falsifies the
mechanism's premise ("bilinear smears; learned detail tightens"): on this
architecture the fixed low-pass was apparently load-bearing regularization, and
removing it costs ~1 MAE.

## 2. Initialization / identifiability

**Random, not identity.** `top_up` receives no custom init anywhere in
`N0007_h0013/model.py` (the constructor `:74-77` is bare; no init in
`CountingHead.__init__` or `Counter.__init__`). PyTorch's default Conv2d init
(kaiming-uniform a=√5, fan_in=9 for grouped conv; uniform bias ±1/3) yields a
random, parity-dependent signed operator whose per-pixel gain is ≈1/√3 in
variance terms — bounded, zero-mean, non-identity.

**Identity-at-init here is not free and was not attempted.** Note this is unlike
N0006's `CountPrior` (`N0006_h0012/model.py:140-147`: zero-init last layer →
tau=1 exactly = champion at step 0). For H0013 there is no "trivial" zero-init
no-op: zero-initing the conv makes the coarse branch exactly 0 (a much larger
function change). An exact bilinear identity *is* representable by encoding the
interpolation coefficients into the four parity kernels — interior-exact, and
border-exact only with replicate padding rather than the zero padding used at
`:76` — but the simplest safe construction (bilinear + zero-init residual) was
not used.

**Could init damage alone explain ~1.0 MAE? No.** The curves contradict it:
- val: child is **better than champion every epoch through ep11** (ep1 135.45 vs
  157.32, ep7 28.08 vs 28.75, ep11 24.62 vs 24.75), then crosses at **ep12**
  (24.468 vs 24.350) and the gap widens monotonically to +0.996 at ep32;
- train loss: child is **lower than champion from ep1 onward** (ep1 23.3 vs 50.3,
  ep32 2.470 vs 2.706), i.e. the random block did not impede optimization, it
  helped it;
- RMSE: child better through ep13, worse thereafter (child 84.68 vs champion
  79.77 at ep32).
A bad-init/sharp-basin story predicts an early deficit that re-converges; the
observed pattern is the opposite (early lead, late regression), so init damage is
at most a second-order contributor.

## 3. Capacity vs optimization/inductive bias — the evidence says inductive bias

+5,120 params cannot be a capacity constraint: 0.15% of trainable weights, and the
child trains *better* (lower train loss throughout). What changed is the
**function class at the coarse merge**: a fixed low-pass mass-preserving operator
became an unconstrained signed local linear map. The learning curve reads as the
canonical bias–variance shift:

| ep | champion val MAE | child val MAE | child−champ | champion train | child train |
|---|---|---|---|---|---|
| 1 | 157.323 | 135.450 | −21.87 | 50.31 | 23.30 |
| 7 | 28.749 | 28.076 | −0.67 | 6.49 | 6.15 |
| 11 | 24.745 | 24.623 | −0.12 | 5.93 | 5.68 |
| 12 | 24.350 | 24.468 | **+0.12** | 5.35 | 5.18 |
| 16 | 23.412 | 24.035 | +0.62 | 4.69 | 4.63 |
| 24 | 21.862 | 22.931 | +1.07 | 3.46 | 3.17 |
| 32 | 21.459 | 22.455 | **+1.00** | 2.71 | 2.47 |

All values from tensorboard `val/mae` and `train/loss_epoch` scalars
(`run/latest/tb/t`; the exec.log only retains the final epoch bar,
`exec.log:551`). The child is simultaneously **better on train and worse on val
from ~ep12** — the signature of extra functional freedom being spent on
train-specific high-frequency detail rather than of an optimization failure. The
gap does not close in the tail (ep24→32 hovers at +0.98…+1.09), and both models
best at ep32, so "needs more epochs" is not supported by this curve; the child's
late slope is no steeper than the champion's. Caveat: single-seed architecture
comparison, run-to-run noise floor still unmeasured (STATE.md queue item:
`repro_run.py` pending), so sub-0.3-MAE early differences are not interpretable;
the ~1.0 endpoint gap is far outside any plausible seed noise and is decisive.

## 4. Pluggability / ablation purity

The diff surface against `tree/N0001_champion/model.py` is exactly:
- `N0007_h0013/model.py:12-16` — docstring only;
- `:58-61` — FineFuser docstring/signature, new `use_subpixel_up=False` kwarg;
- `:64` — `self.use_subpixel_up = use_subpixel_up`;
- `:74-77` — conditional `top_up` construction;
- `:79-84` — forward split: `top = self.top(h3)` then branch to `top_up` vs the
  parent's literal single-line bilinear expression;
- `:167-168` — `CountingHead` forwards
  `use_subpixel_up=_get(cfg, "use_subpixel_up", False)`;
- `config.toml:27` — `use_subpixel_up = true` (key absent from parent config,
  which ends at line 26).

With the switch off, `top_up` is never constructed, so `state_dict` keys, module
construction order and RNG consumption are identical to the champion; the forward
path is the parent's op-for-op (`:80` + `:84` ≡ parent `:68`). The only residue is
an inert `self.use_subpixel_up=False` flag and the branch test. OFF is therefore
structurally byte-identical to the champion path (verified by diff; no torch on
this host to execute a tensor-equality check). This is a clean, single-switch
ablation and it passes the pluggability rule — the failure is the mechanism's,
not the interface's.

## 5. Cross-node pattern (observational)

| node | component | step-0 function vs champion | result vs champion |
|---|---|---|---|
| N0006_h0012 (H0012) | zero-init count-prior temperature, `tau=clamp(1+z)`, z=0 | **exactly identity** (`N0006_h0012/model.py:140-147`) | +0.036 (`N0006_h0012/result.json`), ledger `neutral`-miss `:41` |
| N0003_h0009 (H0009) | `sigmoid(Linear(D,256))` channel gate, default init (`N0003_h0009/model.py:158-170`) | non-identity (≈0.5 per-channel scale) | +1.149 (`memory/hypotheses.jsonl:32`) |
| N0004_h0010 (H0009+H0010 joint) | above + per-exemplar gate, zero-w/bias 3 (`N0004_h0010/model.py:158-178`) | non-identity (joint) | +1.923 (`:35-36`) |
| N0005_h0010 (H0011) | frozen-random-vector gate on N0003 lineage | non-identity, content-free | 22.810 vs 22.608 conditioned; +1.35 vs champion (`:39-40`) |
| N0007_h0013 (H0013) | random-init subpixel upsample | non-identity, and **replaces a fixed operator** | +0.996 (`:44`) |

The one component that was exactly the champion at step 0 (H0012) was the only
near-neutral one; every component that changed the head's function at step 0 lost
~1 MAE or more. H0013 is the sharpest instance because it is only 5 k params: the
loss tracks the **function perturbation** (replacing a fixed low-pass with a
random learned map), not the parameter count. Observationally, the champion head
sits in a narrow basin calibrated to its exact fixed operators; with the backbone
frozen (`AGENTS.md` standing regime) the head cannot re-coordinate upstream
features, and 32 cosine-decayed epochs did not re-converge after the
perturbation. Caveats kept explicit: n=1 per cell, different lineages for
H0009/H0010/H0011, the H0009+H0010 joint is attribution-confounded, and the noise
floor is not yet measured — this is a regularity, not a law.

## Implications for the next hypothesis

- **Make identity-at-init structural, not hoped-for.** For any component that
  replaces or wraps a fixed operator, author it as `fixed_op(x) + zero_init(conv(x))`
  (or encode the exact fixed kernels) so the switch-ON graph is the champion at
  step 0; H0012 shows the only neutral insertion was the exact-identity one, while
  every non-identity insertion paid a ~1 MAE perturbation tax that swamps a 0.30 bar.
- **Do not remove the coarse branch's low-pass/mass-preserving prior implicitly.**
  If subpixel upsampling is revisited, initialize the parity kernels to the
  bilinear interpolation weights (and consider kernel-sum constraints or an
  anti-alias penalty); the N0007 curve (lower train, higher val from ep12) says
  the cost was high-frequency overfit, not under-capacity — and note the final
  upsample at `model.py:90` is still bilinear, so "detail preservation" should be
  tested at the stage that actually feeds the density head.
- **Price non-identity perturbations before spending a solo run.** A new falsifier
  should be booked as `contradicts` on H0013 (`memory/hypotheses.jsonl:44`) with a
  bilinear-kernel-init or residual version measured against a seeded OFF baseline;
  with 32 epochs and an unmeasured noise floor, prefer a short A/B first to show
  the mechanism clears the ~1 MAE perturbation tax, otherwise the run cannot
  attribute anything to the mechanism.

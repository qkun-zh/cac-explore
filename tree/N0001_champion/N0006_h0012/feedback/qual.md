# Qual feedback - N0006_h0012 (H0012 count-conditioned condenser temperature)

Scope: mechanism fidelity and learning-behavior reading. The verdict is already in the
ledger (contradicts, strength 1.0, `memory/hypotheses.jsonl:41`; pre-registered bar
0.40 in `idea.md:7`) and is not re-judged here.

Sources actually read: node `model.py`, `config.toml`, `idea.md`, `result.json`,
`info.json`; parent `tree/N0001_champion/model.py`; node `run/latest/exec.log` and the
tb event file under `run/latest/tb/t/`; champion tb event under
`tree/N0001_champion/run/latest/tb/t/`; `src/cac/{engine/runner,models/pl_module,
models/counter,data/fsc147,data/pl_datamodule,calls/ema,calls/best_checkpoint}.py`.

One local-artifact gap: the node's `run/latest/` contains only `exec.log` + `tb/`, no
`run_node.log` (the champion run dir has one). Per-epoch numbers below come from the tb
`val/mae` / `train/loss_epoch` scalars; the final-epoch value is corroborated by
`exec.log:553-554` (21.672) and `result.json:5` (best 21.4952 @ ep28).

Observed context: 21.4952 @ep28, 32/32 epochs, `budget_hit=false`, 1765.6 s budget
elapsed (`result.json:5-9`); champion 21.4589 @ep32 (`tree/N0001_champion/result.json:5-6`).

## 1. Fidelity: it is an image-descriptor temperature, not a count prior

`CountPrior` maps `cat(GAP(fine), e_mean)` through `Linear(384,64)-GELU-Linear(64,1)`
to a scalar z (`model.py:143-150`); the call site is `model.py:194-197`:
`e_mean = e.mean(dim=1)`, `z = self.count_prior(fine.mean(dim=(2,3)), e_mean)`,
`tau = clamp(1.0 + z, 0.25, 4.0)`.

What tau does: `Condenser.forward` divides the projected queries and keys by
`sqrt(tau)` (`model.py:132-133`). Since `nn.MultiheadAttention` applies its own
`1/sqrt(d_head)` after the projections, scaling q and k each by `1/sqrt(tau)` scales the
pre-softmax logits by exactly `1/tau`. That is one temperature per image, shared by all
4 heads (`model.py:122`) and all `96x96 = 9216` query positions (`S=384`, `S//4`, `model.py:191`).
The softmax is over K=3 exemplar tokens: the dataset emits `bboxes3` with 3 boxes
(`fsc147.py:55-62`) and `Counter` passes `bboxes3` through (`model.py:240-241`);
`ExemplarEncoder` returns one token per box (`model.py:104-109`).

What conditions it: GAP of the fused 1/4-res feature map (a pooled 128-d scene
descriptor) and the mean of 3 exemplar tokens (a 256-d class prototype that discards
exemplar multiplicity). Nothing else.

Supervision: only `out["density"]` enters the loss (`pl_module.py:77-78`), whose count
term is `L1(sum(density), count)` with weight 0.4 (`pl_module.py:31`, `config.toml:49`).
There is no count head, no count input, no count target attached to z, and no detach.
A grep over `src/cac` finds no logging or extra use of `tau`/`count_prior` outside the
node model itself (`mechanisms.py:43` only registers the switch).

So the "global count prior" phrasing in `model.py:12-16`, `model.py:141-142` and
`idea.md:7` is not supported by the code. The component is a learned global attention
temperature conditioned on an image descriptor. It cannot know the count, the instance
number, scene layout, exemplar-to-region correspondence, or any per-head/position demand.
Its reach is a single softmax sharpness applied uniformly everywhere.

## 2. Initialization and identifiability

Zero-init last layer (`model.py:146-147`) gives z=0, tau=1, and an identity map versus the
champion condenser path (`model.py:131-135` vs parent `model.py:121-125`). One wrinkle:
the runner smoke-trains the same model instance for 2 epochs x 4 batches before the timed
fit (`runner.py:96-100`), so the component has already had 8 steps to move when the 32-epoch
clock starts. The identity claim holds only for the first smoke step, not the run start.

Free to move? Yes. The parameters are ordinary `requires_grad` params included by
`self.model.parameters()` (`pl_module.py:50-59`), with no freeze or gate. AdamW weight
decay 0.08 (`config.toml:39`) pulls z toward 0, i.e. toward tau=1, throughout training, and
the cosine schedule decays the global lr to ~3e-4 by E20 and below ~5e-5 by E28 (tb
`lr-AdamW`).

The clamp `[0.25, 4.0]` (`model.py:197`) has zero gradient outside the range, so tau can
pin at a bound and go locally constant without any warning in the logs. Nothing records
whether it did.

Identifiability: the MLP is shared (not per-image free parameters), so the question is
whether the 384-d pooled descriptor predicts a useful sharpness and whether the indirect
gradient path (tau -> softmax(3) -> condenser -> decoder -> density loss) can find it in
32 x 228 = 7296 steps (`exec.log:553`). The trunk's own features stabilize by roughly E10
(champion val 25.19 at E10), and the child's advantage only appears at E20, when the lr is
already small. There is no measurement of z or tau anywhere in the artifacts, so
"enough signal" is not established either way. What is clear: the component is
low-capacity, globally shared, clamped, weight-decayed toward identity, and scheduled into
a shrinking lr.

## 3. Overlap with the champion's existing count-adaptive machinery

GCA reads exactly the same two tensors: `gap = fine.mean(dim=(2,3))`, `cat(gap, e_mean)`
(`model.py:213-217`), identical to `model.py:194-196`. The MLP shape is also the same:
two layers, hidden 64, GELU, zero-init last layer (`model.py:145-147` vs `model.py:209-211`).
The information channel is a structural duplicate of GCA. Only the use of the scalar
differs: GCA adds `0.02 * n_aux / (Hf*Wf)` to every pixel (`model.py:216`, `model.py:249`),
CountPrior scales attention logits.

GCA is already count-adapted through that bias: the count L1 sees `sum(density)` including
the bias (`pl_module.py:31`), even though the loss never reads `n_aux` directly. Beyond
GCA, the density head as a whole is trained to match the count sum (`pl_module.py:31`),
so total mass is already image-adaptive; XScale (`model.py:110-113`) is a multi-scale
exemplar summary, not count-adaptive; and the condenser attention is content-dependent by
construction. H0012 therefore adds a knob on an already adaptive attention, fed by the
same pooled vector GCA consumes.

With K=3 prototypes, "attend broadly over prototypes" means flattening a 3-way softmax.
A single scalar has little room to encode the broad/sharp regime split the idea describes.

## 4. Pluggability and ablation purity

Switch off: `Condenser` is built with the flag false (`model.py:118-120`), its forward
takes the else branch (`model.py:134-135`) which is operation-identical to parent
`model.py:121-123` (with `q = self.norm1(tok)` hoisted at `model.py:130`); `CountingHead`
skips `CountPrior` (`model.py:185-186`), tau=None (`model.py:198-199`), then calls
`self.cond(fmap, e, tau)` (`model.py:200`). `e_mean` is computed before the condenser
(`model.py:194`) instead of after (parent `model.py:167`), but it is a pure reduction.
A switch-off build reproduces the champion graph and parameter-init RNG.

Exact diff surface, child vs parent (from `diff -u`):
- `model.py:12-17` docstring only
- `model.py:118` and `model.py:120` (`use_count_temp` arg, stored flag)
- `model.py:128-137` (`tau` arg, q/k division, else branch)
- `model.py:140-150` (`CountPrior`, new)
- `model.py:180-186` (`use_count_temp` plumbed into `Condenser`, conditional module)
- `model.py:194-200` (tau computed, passed to condenser)
- `config.toml:27` `use_count_temp = true` (one key; also a lost trailing newline)
Nothing else differs, and the model/config hashes in `result.json:12-13` match the files.

Run-level caveat: the seed is set once before model construction (`runner.py:62,79`). The
child instantiates `count_prior` inside `CountingHead` (`model.py:185-186`), before
`Counter` builds `GCA` (`model.py:230-231`), so GCA's Linear layers receive different RNG
draws than the champion's. The loaders use no explicit generator (`pl_datamodule.py:76-83`,
`runner.py:117-119` does not pass the seed), so shuffle seeds and the flip augmentation
(`torch.rand`, `fsc147.py:64`) also descend from the global RNG, whose state differs
because of those extra draws. The executed switch-on run is same-seed but not RNG-paired
with the champion: GCA init, batch order, and flip sequence all differ. The code ablation
is clean; the comparison as run is not a one-knob A/B. This is cheap to fix (instantiate
added modules after GCA, or pass explicit generators).

Engine consistency: champion tb starts 13:37, N0006 tb starts 15:35, and local
`runner.py` was last edited at 15:55 (commit 72bf8e4, honoring `cfg.deterministic`).
Both runs predate that edit, so both used the pre-fix runner. Neither honored
`deterministic=true`, consistently.

## 5. Learning-behavior read from the tb curves

`val/mae` is EMA-inference: the 0.999 decay shadow is swapped in during validation
(`ema.py:58-64`, `runner.py:76-77`). Effective horizon ~1000 steps, ~4.4 epochs at 228
steps/epoch. Both runs use the same EMA.

| ep | champ | child | delta (champ-child; + = child better) |
|---|---|---|---|
| 1 | 157.3232 | 151.2546 | +6.0686 |
| 2 | 76.2375 | 74.9010 | +1.3365 |
| 3 | 53.2973 | 53.6889 | -0.3917 |
| 4 | 42.5154 | 42.4734 | +0.0420 |
| 5 | 35.7347 | 35.9252 | -0.1905 |
| 6 | 31.3749 | 31.8090 | -0.4341 |
| 7 | 28.7491 | 29.3477 | -0.5986 |
| 8 | 27.0146 | 27.8061 | -0.7915 |
| 9 | 25.9413 | 26.5801 | -0.6388 |
| 10 | 25.1930 | 25.7510 | -0.5580 |
| 11 | 24.7454 | 25.2333 | -0.4879 |
| 12 | 24.3498 | 24.7589 | -0.4091 |
| 13 | 24.1115 | 24.4825 | -0.3710 |
| 14 | 23.7738 | 24.2215 | -0.4477 |
| 15 | 23.6001 | 23.9233 | -0.3232 |
| 16 | 23.4119 | 23.5461 | -0.1342 |
| 17 | 23.1817 | 23.3585 | -0.1769 |
| 18 | 22.9429 | 23.0512 | -0.1084 |
| 19 | 22.7674 | 22.8534 | -0.0860 |
| 20 | 22.6830 | 22.5524 | +0.1306 |
| 21 | 22.4579 | 22.2128 | +0.2451 |
| 22 | 22.2909 | 21.9408 | +0.3502 |
| 23 | 22.0963 | 21.7400 | +0.3563 |
| 24 | 21.8620 | 21.6532 | +0.2088 |
| 25 | 21.7278 | 21.5716 | +0.1562 |
| 26 | 21.6732 | 21.5408 | +0.1323 |
| 27 | 21.6029 | 21.5147 | +0.0883 |
| 28 | 21.5615 | 21.4952 | +0.0664 |
| 29 | 21.5198 | 21.5374 | -0.0177 |
| 30 | 21.4860 | 21.5923 | -0.1063 |
| 31 | 21.4686 | 21.6369 | -0.1683 |
| 32 | 21.4589 | 21.6718 | -0.2129 |

Plateau: yes, earlier. The child peaked at E28, then gave back 0.1766 over E29-32; the
champion was still decreasing when the budget ended (E32 best). The child was behind from
E3-E19 (max 0.7915 at E8), ahead E20-E28 (peak +0.3563 at E23), and crossed under at E29.
Meanwhile `train/loss_epoch` is lower for the child in 17 of 32 epochs and at the tail
(E32: 2.5294 vs 2.7062 champion; E28: 2.8396 vs 2.9283). Better train fit with worse EMA
val at the end is a capacity/overfit signature, not an optimization failure.

Collapse to a constant? Not inferable from these artifacts. The tb scalars are only
`hp_metric`, `lr-AdamW`, `train/loss`, `train/loss_epoch`, `train/mae` (nan), `val/mae`,
`val/rmse`, `epoch`. No z/tau scalar, no per-image or per-count-bin logs, and no
`best.pth` in the local run dir, so the learned temperature cannot be probed offline
here. Stated plainly: whether tau collapsed to identity, saturated at a clamp bound, or
tracked anything is unknown.

## 6. Mid-run lead then tail flattening: what the curve can and cannot say

The E1 gap (+6.07) cannot be read as a temperature effect. Because the shuffle and flip
streams differ (section 4), the two trajectories diverge from the first epoch onward; in
the steep part of the error curve, small weight differences produce multi-point MAE
swings. Likewise the E8 deficit (0.79) and the E20-E28 lead (max 0.36) are of the same
order as run-to-run trajectory variation, and the lab noise floor is still unmeasured
(`STATE.md` gotcha, `scripts/repro_run.py` queued for that). So the honest reading is:
the shape is suggestive of a transient benefit, but this run alone does not separate it
from seed/RNG drift.

Mechanistic candidates, testable:
1. Capacity/overfit. Train loss keeps improving after E28 while EMA val worsens; test with
   an early stop, higher wd, or by logging raw-weight val at the tail.
2. lr-window interaction. The lead window coincides with lr falling under ~3e-4; the knob
   had little step size left by the time it looked useful. Test with a component-specific
   lr or a schedule that leaves late capacity to move.
3. RNG pairing. Rerun the child with the RNG prefix matched to the champion (build the new
   module after GCA, explicit loader generators) and see how much of the E20-E28 lead
   survives. Also note the same-seed champion replicate in flight will help size the noise
   floor, but it will not fix the pairing of this particular A/B.
4. Instrumentation. Log mean/std/histogram of z and tau per epoch, and tau against count
   bins at validation. That turns collapse, saturation, and direction into measured facts.

Untestable with what exists: whether tau collapsed or saturated; whether z correlates
with count or with a nuisance like exemplar box size; whether any gain was on dense or
sparse scenes (only aggregate MAE is logged); whether the final +0.0363 gap is mechanism
or noise. One more distinction: if z converged to a nonzero constant, tau would be a fixed
rescaled temperature, not identity. Only z -> 0 restores the champion attention, so
"temperature became constant" and "temperature became wrong" are different outcomes this
run cannot separate.

## Implications for the next hypothesis

- If a temperature is retried, condition it on something that is actually count-identified:
  reuse the count-calibrated path (e.g. `n_aux` or the trunk's own density sum) or add a
  direct count-proxy objective, and log z/tau per epoch. As implemented, it re-reads GCA's
  inputs and receives no direct count signal anywhere.
- If the target is regime-dependent attention, the knob has to be finer than one scalar
  per image: per-head or position-band temperature, or an explicit attention-mass target.
  A shared scalar over a 3-way softmax at 9216 positions cannot express much of the
  broad/sharp split the idea describes.
- Treat seed pairing and component budget as part of the mechanism, not as run hygiene:
  instantiate added modules after GCA (or pass explicit loader generators) so switch-on
  and switch-off share RNG, and note that a late-blooming component gets little movement
  from a global cosine schedule in 32 epochs. An early-stop or wd check is warranted given
  the overfit-shaped tail.

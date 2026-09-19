# N0007_h0013 - Causal identification critique (H0013, solo)

Scope: causal-identification critique only. The verdict itself is not re-litigated:
the ledger already recorded `contradicts` w=1.0 for H0013
(`memory/hypotheses.jsonl`, 2026-09-19T16:33:42, child 22.4545 vs champion
21.4589), H0013 sits at conf 0.4, `n_tested: 1`, status `uncertain`. The question
here is what the run identifies, what it merely bundles, and what a cleaner test
of the H0013 mechanism would look like.

Evidence actually read for this report:
- node: `idea.md`, `model.py`, `config.toml`, `result.json`, `info.json`;
  `diff` of both configs and both model files against the parent
- parent: `model.py`, `config.toml`, `result.json`; parent TB event file
- per-epoch `val/mae`, `train/loss_epoch`, `lr` from both TB streams:
  `tree/N0001_champion/run/latest/tb/.../events.out.*` (parent) and
  `tree/N0001_champion/N0007_h0013/run/latest/tb/.../events.out.*` (child).
  This run's `exec.log` carries no epoch rows; the TB scalars are the only
  per-epoch record.
- `memory/hypotheses.jsonl` and `memory/index.json` (H0013 block)
- harness source: `src/cac/engine/runner.py`, `src/cac/data/pl_datamodule.py`,
  listing of `src/cac/models/`, `src/cac/calls/best_checkpoint.py`
- server (read-only): checkpoint listing under
  `/data/cac/tree/N0001_champion/{run,N0007_h0013/run}/latest/`, plus two
  scratch scripts in `/tmp` (no tree writes, no repo writes) that (a) compared
  parent-config vs child-config `CountingHead` initializations at seed
  20260830 and (b) compared the global-RNG stream those constructions leave
  behind
- `feedback/quant.md` and `feedback/qual.md` were not present when this report
  was drafted (the feedback dir was empty at 16:34; the other two agents
  produced their files while this one was being written). Nothing here is
  taken from them.

## 0. Headline

This is a valid single-switch, pre-registered **bundle** test, and it misses its
bar by 1.30 MAE. It is **not** an identified test of the mechanism H0013 names
(bilinear low-pass smears support vs learned upsampling preserves detail). Two
reasons, both measured here rather than assumed:

1. The treatment is confounded with a global-RNG displacement. `top_up` is
   constructed in the middle of `FineFuser`, before `ExemplarEncoder`,
   `Condenser`, `DensityDecoder` and `GCA`; with the seed fixed, those modules
   receive measurably different initial weights in the child (verified on the
   server). And because `runner.py` never passes `seed` to
   `FSC147DataModule`, that RNG offset also changes the training shuffle order
   (also verified). The contrast is "different resampler + different init of
   most trained modules + different batch order", not "same training stream,
   one operator swapped".
2. The mechanism claim was never measured. The only endpoint is a count-level
   MAE; sharpness/localization cannot be read off it.

The +0.996 itself is larger than anything the hypothesis hoped to gain (0.30)
and larger than the two near-neutral nodes, so it is unlikely to be pure
finiteness of one number. But "unlikely" is not "unmeasured", and this run does
not supply the measurement that would settle it.

## 1. Identification: the exact contrast

**Solo is confirmed.** `idea.md` books exactly H0013; `info.json`
`tested_hypotheses: ["H0013"]`; no adjoin, no composition. The node's own
`model.py` is the one executed (`runner.py:36-43` takes the sibling file over
the registry).

**Config contrast is exactly one functional line.**
`diff parent/config.toml child/config.toml` =:
```
26a27
> use_subpixel_up = true
```
The parent has no such key; `FineFuser` defaults it to `False` and
`CountingHead` reads `_get(cfg, "use_subpixel_up", False)` (child `model.py:167-168`),
so deleting the key restores the champion exactly. Seed (20260830),
`deterministic = true`, AdamW/LR/wd/cosine, `loss_count_weight`, batch size,
`epochs = 32`, data settings, `use_gca`, `use_xscale`, `use_ddca = false` are
identical. The optimizer/loss/schedule invariant holds.

**Model contrast is one component plus its wiring.** `diff` of the two
`model.py` files shows only: the `use_subpixel_up` constructor arg, the
`top_up` module, the forward branch, the `CountingHead` pass-through, and the
docstring. `use_ddca` is false in both, so the DDCA branch is not built either
way.

**Where the intervention sits.** In `FineFuser.forward` the top branch is
`top = self.top(h3)` (1x1 conv + GroupNorm on the coarse backbone hidden state
h3) and the parent then does `F.interpolate(top, scale_factor=2, mode="bilinear",
align_corners=False)` before `torch.cat([self.lat(h2), top], 1)`. Shape
bookkeeping from the live code: the fused map `f` is 48x48 and the decoder runs
at 96x96 (`Hf = Wf = S//4`), so at 384 input the top branch is upsampled from
24x24 (S/16) to 48x48 (S/8) to match `lat(h2)`, and the *second* upsample
(48x48 into 96x96, S/8 to S/4, line 90 of both files) remains bilinear in **both
arms**. The replaced operator is therefore the first of two resamplers, sitting
at the input to fusion, not at the density output.

**What else that single change moves.**

- Sampling kernel: fixed 2-tap separable bilinear (0 params, weights shared
  across channels and positions, low-pass by construction) becomes a grouped
  3x3 conv on 128 channels (one input channel per group) plus `PixelShuffle(2)`
  (5120 params: 512x1x3x3 weight + 512 bias), i.e. a learned
  phase/position-dependent per-channel interpolation, applied **after** the
  GroupNorm in `self.top`.
- Downstream consumers of the changed stage: the fused 96x96 map feeds (a) the
  density decoder input via the channel concat, (b) the condenser query tokens
  (`fmap = fine.permute...flatten`) that exemplar cross-attention attends from,
  and (c) `GCA` via `fine.mean(dim=(2,3))`. The replacement therefore changes
  the decoder's local evidence **and** the attention queries **and** the global
  count aux, in both runs of the pipeline. This is broader than "an upsampler
  in front of the head".
- Init and data stream (verified, not inferred): constructing `top_up`
  consumes 5120 global RNG draws *before* the exemplar encoder, condenser,
  decoder (and GCA after). With `seed_everything(20260830)` and building only
  `CountingHead` for each config, `fuser.top/lat/fuse/refine` weights are
  bit-identical between parent and child, while
  `exemplar.proj.weight`, `exemplar.tr.layers.0.self_attn.in_proj_weight`,
  `cond.proj_in.weight` and `decoder.block.0.weight` all differ (server test,
  parameters: 3,479,938 -> 3,485,058, delta exactly 5120).
- The same offset reaches the data order: `runner.py:117` constructs
  `FSC147DataModule(cfg, smoke=False, img_size=..., batch_size=...)` **without**
  `seed=seed`, so `pl_datamodule.py:79-82` never installs its dedicated
  generator and `DataLoader(shuffle=True)` uses the global RNG (Lightning only
  adds `worker_init_fn`; no generator is injected). Replaying that sequence
  with the parent and child heads produced different first-epoch permutations
  (parent `[2318, 1003, 2169, ...]`, child `[2699, 1786, 1514, ...]`) while
  `torch.initial_seed()` stayed 20260830 in both. The val loader is
  `shuffle=False`, so the val metric is unaffected; the train stream is not.
  Note this applies to every node in the tree that constructs an extra module
  before fit, so the earlier N0006 feedback's "batch order is identical" line
  needs correcting: the datamodule receives a seed only if the caller passes it,
  and the shipped runner does not.

Net: this is a clean single-switch **bundle** contrast. It is not a contrast
that isolates the resampling operator from initialization and data order.

## 2. Magnitude interpretation

Endpoint: child best 22.4545 @ep32 vs parent best 21.4589 @ep32, so **+0.9956**.
The bar (<= 21.1589, i.e. 0.30 lower) is missed by **1.2956**, and no epoch of
the child comes within 1.29 of it. Both runs: 32/32 epochs, `budget_hit = false`
(child 1708.1 s, parent 1704.5 s). Best checkpoint is the last epoch in both.

Per-epoch `val/mae` delta (child - parent), extracted from the two TB event
files (step 227 = epoch 1, ..., step 7295 = epoch 32):

| phase | epochs | delta (child - parent) |
|---|---|---|
| child leads | 1-11 | -21.87 (ep1) tapering to -0.12 (ep11) |
| crossover | 12 | +0.12 |
| monotone divergence | 13-24 | +0.17 to +1.07 |
| plateau | 24-32 | mean +1.02, max +1.09 (ep25), final +1.00 |

Train side: the child's `train/loss_epoch` is lower at most epochs and at the
end (2.470 vs 2.706 at ep32; ep1 23.30 vs 50.30). So the child fits the train
objective earlier and lower while ending worse on val. That is a
fit/regularization shift, not a late-training fluke, and it is not explained by
the endpoint alone.

What the magnitude does and does not say about the champion's bilinear prior:

- It shows the champion's **endpoint is sensitive to this junction** under this
  protocol. The child's early advantage (until ep12) rules out the simplest
  story that the learned block is inert; it demonstrably changes training.
- It is consistent with the bilinear low-pass acting as a useful prior at this
  coarse branch (the child trades train fit for val), but it is equally
  consistent with (i) the random-init, unnormalized subpixel conv shifting the
  fused feature statistics, (ii) the verified re-init of exemplar/condenser/
  decoder changing the basin, and (iii) different batch order across the two
  arms. Nothing in the run separates these.
- It is **not** a general "learned upsampling is bad" result. The tested object
  is one grouped 3x3 conv + PixelShuffle at one junction (S/16 -> S/8, before
  fusion, after GroupNorm, random init, no residual), under one optimizer/loss/
  schedule/seed. Other placements (the surviving S/8 -> S/4 stage, decoder
  output), identity-init variants, or end-to-end-trained architectures (the
  ESPCN-style subpixel line the idea cites) are untouched.
- It is **not** yet a "bilinear smoothing prior is load-bearing" result either,
  because the bundle includes the init/stream confound above, and because the
  mechanism's own observable (map sharpness, localization) was never measured.
- The schedule is truncated: both models improve in the final epoch (parent
  21.4686 -> 21.4589, child 22.4662 -> 22.4545) and neither has plateaued.
  The comparison is between two non-converged endpoints of a 32-epoch cosine,
  so part of the gap can be convergence-speed rather than asymptote.

## 3. Same-parameter-class confound and direct mechanism measurements

The replacement is not matched on any of the three axes that matter:

- **Parameters:** bilinear 0 vs `top_up` 5120. That is 0.15% of the ~3.5M
  trainable head and 0.016% of the 31.3M total, so raw capacity is not a
  serious confound; the idea's "without meaningful parameters" phrasing is
  defensible. It does mean the treatment is not parameter-matched in a strict
  sense.
- **FLOPs:** the grouped conv costs roughly 2.6M MACs per image at 24x24 versus
  ~1.2M for the bilinear stage. Both are noise next to the backbone, so the
  FLOP mismatch is irrelevant here.
- **Function at init and training stream:** this is the real mismatch. The
  learned conv is randomly initialized, so the child starts from a different
  function, not from bilinear; and the verified RNG transfer re-initializes the
  exemplar encoder, condenser, decoder and GCA and reorders the training
  stream. The hypothesis' "without changing the feature interface" is true in
  shape/channel terms and false in statistics/trajectory terms.

Does the MAE contrast fairly test "preserves detail, tightens localization"?
No. FSC147 MAE is `|sum(predicted density) - count|` per image. A sharper map
can lose MAE by creating false positives under imperfect exemplar matching; a
blurrier map can win MAE by spreading mass under the right total. The
mechanism's observable is map-level and was never recorded.

Micro-measurements that would test the mechanism, in increasing cost:

1. **Kernel audit, weights only.** `top_up.0.weight` is `[512, 1, 3, 3]` plus
   bias; per input channel it defines four 3x3 phase filters. Compare each
   channel's 2x interpolation matrix to the bilinear matrix (Frobenius
   distance, separability, center-of-mass). This says whether training learned
   a near-bilinear interpolation (then the block's damage is its init/trajectory,
   not its function), an anti-smoothing filter, or noise. Runs from `best.pth`
   on CPU; minutes, no data, no GPU.
2. **Map sharpness on the val split.** With both checkpoints loaded, compute
   per-image participation ratio `(sum d^2) / (sum d)^2` of the predicted
   density, peak counts, and mass-weighted squared distance from predicted mass
   to the GT point annotations. The mechanism predicts the child is sharper and
   its mass sits closer to GT points; a null or reversed result falsifies the
   mechanism even though the MAE bar already failed.
3. **Count vs localization decomposition, stratified.** For each image, the
   count error `|c_hat - n_gt|` is what MAE sees; add a matching-based
   localization term (Hungarian distances of density centroids/peaks to GT
   points, or an EMD/Sinkhorn between predicted mass and GT points). Then
   stratify both by GT-count quintile and mean object area, where "smearing
   hurts" should concentrate. A flat `delta`-vs-count or `delta`-vs-size curve
   undercuts the mechanism claim independent of the global bar.

Post-hoc feasibility, honestly: the local artifacts are the two TB scalar
streams (32 val/mae points, train/loss, lr), `exec.log`, and the configs/models.
No per-image predictions, counts or feature dumps were saved anywhere (server
listing confirms). `best.pth` does exist on the server for both arms
(`/data/cac/tree/N0001_champion/run/latest/best.pth`, 125,383,139 B; child
`/data/cac/tree/N0001_champion/N0007_h0013/run/latest/best.pth`, 125,404,249 B;
the +21,110 B is the 5120 params plus metadata). The checkpoint stores the raw
state dict (`best_checkpoint.py:27-29`), not EMA. So the kernel audit needs only
the file, but the map/localization audit needs a **new server-side eval pass**
(load checkpoint, forward val, dump per-image maps/counts). That is cheap
(minutes of GPU, no tree state) but it is a new measurement: descriptive unless
pre-registered with numeric falsifiers.

## 4. Cross-node pattern with n=1 each

| node | treatment | best | delta vs champion | best ep | identity at init? |
|---|---|---|---|---|---|
| N0003_h0009 | image-global per-channel exemplar gate | 22.6076 | +1.149 | 32 | no (sigmoid of random linear) |
| N0004_h0010 | channel gate + per-exemplar gate (joint) | 23.3824 | +1.923 | 32 | no (channel gate is not; joint run) |
| N0006_h0012 | count-conditioned condenser temperature | 21.4952 | +0.036 | 28 | yes (zero-init MLP, tau=1) |
| N0007_h0013 | learned subpixel upsample in FineFuser | 22.4545 | +0.996 | 32 | no (random conv; re-seeds downstream) |
| N0005_h0011 | frozen-random gate on the gated lineage | 22.8102 | +0.202 vs 22.608 | 32 | n/a (control for H0010) |

The ordering tracks how far each treatment moves the step-0 function and
training stream more than it tracks any named mechanism: the identity-at-init
temperature head is nearly free (+0.04), the frozen-random gate on the already
gated lineage costs ~0.2, and the treatments that replace or rescale learned
features (H0009, H0010, H0013) cost ~1-2. That is not proof, but it is a
coherent alternative explanation the current data cannot exclude:

- **Fragile optimum / perturbation cost.** The champion's hyperparameters
  (wd 0.08, 32-epoch cosine, lr 1e-3) were tuned around the champion itself.
  Any addition that slightly helps also perturbs the basin and may need its own
  retuning to pay off; the fixed-hyperparameter regime will make such
  mechanisms look harmful.
- **Budget truncation.** All best checkpoints are late (28-32), train loss is
  still falling, and H0013's child clearly optimizes faster early. A treatment
  that converges faster can be penalized by an endpoint at a fixed epoch.
- **Unmeasured stream/seed noise.** n=1 per arm, no different-seed replicate,
  and (as established in section 1) parent and child do not even share an
  initialization stream or data order. The queued `repro_run` for the champion
  is same-seed, which measures engine determinism only and cannot bound this.
- **Scope-of-transfer.** SAFECount/BMNet+ components are designed inside
  end-to-end-trained architectures; their marginal value on a frozen
  intermediate-feature head with fixed hyperparameters may genuinely be small
  or negative. That is an explanation for the pattern, not evidence for it.

Cheap, pre-registerable diagnostics that separate "fragile optimum" from "bad
mechanism":

1. **Constructor placebo for this node (1 run).** Build `top_up` at the same
   point in `FineFuser` but never use it (champion function, same 5120 RNG
   draws, same re-init of downstream modules, same data-order shift). This
   isolates the exact stream side effect of N0007's code change. Decision rule
   to register before running: if the placebo lands within ~0.15 of 21.459,
   the child's +1.0 is attributable to the resampler function and the init/order
   confound is small for this node; if it lands near 22.4, the H0013 estimate
   is mostly stream, and the family verdict needs a re-run under matched
   streams.
2. **Seed ensemble for the champion (2-3 scratch runs, no tree writes).**
   Same config, different seeds (20260831, 20260832), via `repro_run.py` into
   scratch dirs. Report min/max of `best_mae`. This bounds how large a delta
   this protocol can resolve at all; if the champion spans 21.3-21.8, no
   sub-0.5 claim from any existing n=1 node is identifiable without more seeds.
   (A measurement, not a hypothesis; no verdict movement.)
3. **Function placebo for H0013 (1 run, new booking).** Bilinear plus a
   zero-init grouped-conv residual at the same site: identity at init, same
   parameter count, same construction position. If it recovers the champion,
   the harm is in what training did to the learned filters (and the kernel
   audit can say what they became); if it also loses ~1.0, it is the extra
   capacity/stream, not the specific learned kernel.

Harness fix worth proposing to the Lead (not done here, since it changes the
reference): pass `seed=seed` to both `FSC147DataModule(...)` calls in
`runner.py` (lines 96 and 117). That installs the dedicated generator and
worker init, holding the data stream fixed across arms so future single-switch
contrasts are identified. It would require re-running the champion reference
under the fixed datamodule to keep comparisons honest.

## 5. Next clean tests, ranked by identification cleanliness vs cost

1. **Proceed with N0008_h0014 (SAFECount) and N0009_h0015 (BMNet+ gate) as
   booked.** Both are single-switch, pre-registered (`<= 21.159`), and, from
   their `model.py`, both construct the new module after `DensityDecoder`
   (`SafeEnhance` / `DynExemplarGate`), so their RNG offset touches only GCA's
   hidden layer, and both are identity-at-init (`out` zero-init; gate
   `2*sigmoid(0) = 1`). That makes them cleaner than N0007 on exactly the axis
   this report criticizes. N0009 is the clean solo per-exemplar gate H0010
   never got (its two tests were joint or severed, ledger: neutral).
2. **Kernel audit + (if wanted) stratified map/localization eval on the two
   existing checkpoints.** No training, no tree state; directly reads the
   mechanism's own variables. Pre-register the expected signs/thresholds before
   looking if it is to feed the ledger; otherwise descriptive.
3. **Constructor placebo for H0013 (1 run)** as in section 4. Highest value if
   the upsampling family is kept; must book as a new hypothesis or as
   contradicts/neutral evidence on H0013 with a new falsifier (rule 11: no
   silent retry).
4. **Champion seed ensemble (2-3 scratch runs).** Lab-wide resolution fix;
   informs how to read every n=1 delta, not just this node.
5. **Re-place the learned upsampling (second stage or decoder output) with
   identity init (1-2 runs).** Last resort: this run gives it no extra
   discriminability, and the family already has one decisive bar miss.

What not to do: re-run H0013 with a different seed hoping to flip the recorded
sign (outcome shopping); lower the 0.30 bar; bundle the subpixel change with
another switch to rescue it.

## What this run does / does not license us to conclude

- **Licensed:** at seed 20260830, 32/32 epochs, budget not hit, enabling
  `use_subpixel_up` (grouped 3x3 conv + PixelShuffle replacing the first
  bilinear upsample, S/16 -> S/8, inside `FineFuser`) ends at val MAE 22.4545
  vs the champion's 21.4589, missing the pre-registered >=0.30 improvement by
  1.30; no epoch is within 1.29 of the bar; the sign is stable from ep12
  through ep32.
- **Licensed (descriptive):** the child fits the train objective faster and
  lower while val is worse, so the bundle changed the fit/regularization
  trade-off; and the reported comparison is between two non-converged
  endpoints of a 32-epoch cosine.
- **Not licensed:** "learned upsampling is bad", or even "subpixel upsampling
  at this junction is bad per se". The contrast bundles the sampling-function
  change with a verified re-initialization of exemplar encoder, condenser,
  decoder and GCA (mid-trunk RNG displacement) and a different training batch
  order (datamodule never receives `seed`), at n=1 with no bounded stream/seed
  noise.
- **Not licensed:** the H0013 mechanism story ("bilinear low-pass smears
  support, learned detail tightens localization"). No map-level sharpness or
  localization quantity was measured; count-level MAE cannot see either.
- **Not licensed:** extrapolation to other placements/initializations of
  learned upsampling, or to end-to-end-trained subpixel architectures. Also
  not licensed: treating this single run as a general statement about the
  champion's dependence on its smoothing prior, which is a sensible hypothesis
  but still needs the placebo and map-level measurements above.

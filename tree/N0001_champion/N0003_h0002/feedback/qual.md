# N0003_h0002 — qualitative feedback (text-log analysis, AGENTS §9)

Role: Qualitative Feedback subagent. Sources actually read (server artifacts, no
feature dumps stored, per §9):
- `ssh cac-server 'tail -80 /data/cac/tree/N0001_champion/N0003_h0002/chain_run.log'`
- `ssh cac-server 'grep -nE "WARNING|Error|Traceback|nan|inf|budget|_BudgetStop" .../chain_run.log'`
- `ssh cac-server 'ls -la .../run/latest/'`
- `.../chain_run.log` (training) and `.../run_smoke.log` (smoke)
- TensorBoard scalars under `.../run/latest/tb/t/events.out.tfevents.*` (the only
  per-epoch record; the text log only preserves the final progress bar)
- parent/sibling scalars from `N0001_champion` and `N0002_h0001` for contrast
- `best.pth` state dict (learned scalars)

## 1. Training character, and the "epoch-0 is parent-level" premise — NOT supported

`val/mae` per epoch (TensorBoard, step 227 = end of epoch 0):

| node | ep0 | ep1 | ep5 | ep10 | ep20 | ep24 | final |
|---|---|---|---|---|---|---|---|
| N0001_champion (parent) | 146.998 | 69.024 | 30.624 | 24.779 | 23.509 | 23.388 | 23.336 |
| N0002_h0001 (sibling) | 153.287 | 70.053 | 30.295 | 24.675 | 22.902 | 22.720 | 22.570 |
| **N0003_h0002** | **164.203** | **79.015** | **34.109** | **25.423** | **23.576** | **23.217** | **23.209** |

The premise "epoch-0 should be parent-level vs N0002's cold start" does **not
hold as measured**. N0003's epoch-0 val MAE (164.2) is the worst of the three,
above both N0002 (153.3) and parent N0001 (147.0) by +10.9 / +17.2 MAE. The
reason is structural, not a defect: `runner.py:79,116` builds the head fresh
(frozen backbone only — log line `Loading weights: 0/180` is the pretrained
backbone) and `run_train` has no parent-weight warm start, so **every** node
cold-starts its 3.5 M head; epoch-0 is evaluated after one trained epoch, not
at initialization. The idea's "init-identical forward" claim is a statement
about the parameter-initialization forward, and the 3 scalars do start at the
parent's coupling (`model.py` diff vs parent is exactly: `s_pos=0.02, s_lin=0,
b_cal=0` added after the parent GCA layer, no RNG draw). But that equivalence
is invisible to the epoch-0 metric, which is dominated by random head init.
Qualitatively, N0003 then descends **slower** than both baselines through
mid-training (e.g. ep5 34.1 vs 30.6/30.3; ep10 25.42 vs 24.78/24.68) and only
crosses below the parent around ep24, ending 0.13 better than N0001 but 0.64
worse than sibling N0002.

## 2. Learned-coupling branch — no early instability (the flagged risk did not fire)

Idea §6 open risk (2): large negative `n_cal` on near-empty images could push
cells negative and spike the L1 term in epochs 1–3, an abort-and-report signal.
Observed the opposite of divergence:

- `train/loss_epoch` ep0→ep3: **20.93 → 10.03 → 8.81 → 8.41** (monotone down),
  then a smooth decay to 2.86 at ep31.
- `val/mae` decreases monotonically every epoch; no sawtooth (EMA, decay 0.999).
- No `nan`/`inf`/`Traceback`/`Error` literal anywhere in `chain_run.log`.
- Per-step `train/loss` spikes exist (max 631.6 at the first logged step, step
  49) but the sibling has the same signature (its first step 328.3) and the
  spikes do not grow — heavy-tail batch noise, not a coupling blow-up.

The branch did move off init (`best.pth`, EMA-averaged weights that produced
the val number): `s_pos = 0.008398` (init 0.02, −58.0%), `s_lin = 0.021477`
(init 0), `b_cal = −0.119620` count units (init 0). So this is **not** the
"collapse-to-init / already optimal" null branch of idea §6 — the optimizer
used the new degree of freedom, bounded and small, with no instability. Per §6
the box read is quant's to own; flagged here only as the qualitative mechanic.

## 3. Warning / error inventory — all benign, none node-specific

`chain_run.log` (566 lines): 257 `UserWarning`s, **0** errors/tracebacks/NaN/inf.

| count | warning | reading |
|---|---|---|
| 256 | `fsc147.py:70` NumPy array not writable → tensor | PyTorch/numpy interaction; repeats once per DataLoader worker process (32 ep × 8 procs); present identically in N0002/parent. Benign. |
| 1 | `torchmetrics/.../prints.py:43` MeanMetric `compute` before `update` | Explains the `train/mae = nan` for all 32 epochs — and `train/mae` is nan for **all three** nodes (N0001/N0002/N0003), so it is a pre-existing harness quirk, not a symptom of this node. Benign. |

`grep -nE "...|budget|_BudgetStop"` matched only `"budget_hit": false` (line
561). No `_BudgetStop` print (`[budget] exceeded ...`) appears. Smoke log adds
only routine Lightning notices (151 eval-mode modules, `num_workers` hint).

## 4. best.pth presence / size

`run/latest/best.pth` exists, **125,383,956 bytes** (~119.6 MiB), `epoch=32`,
`best_mae=23.20878791809082`. Size agrees with Lightning's own estimate
(`Total estimated model params size (MB): 125.299` for 31.3 M params at fp32),
i.e. a complete state dict (incl. EMA-baked `gca.s_pos/s_lin/b_cal`), no
truncation. `run/latest/tb/t/` event file + `hparams.yaml` also present.

## 5. Smoke-vs-train consistency

`run_smoke.log` (17 lines) ends `[smoke] one step fits: ok (model=Counter)` /
`SMOKE_RESULT {'smoke': True}`; same node `model.py`, same seed 20260830, same
`Counter` build path. Train then logs `Counter 31.3 M` (3.5 M trainable),
matching the smoke's `model=Counter`. Consistent. Caveat (qualitative, not a
failure): smoke runs `limit_train_batches=4`, `limit_val_batches=2`
(`runner.py:91-92`) at bf16; it can only prove fit-in-memory, and by
construction cannot observe the near-empty-image tail risk of §2 above — which
is why the full-run loss curve was the meaningful check.

## 6. Confidence note on pass cleanliness

Mechanically this is a **clean pass**: 32/32 epochs, `budget_hit: false`,
elapsed 1704.2 s (< 1800 s τ_max), no errors/NaN/inf, smooth monotone
`val/mae`, complete `best.pth`, smoke and train consistent, and the code delta
is exactly the one config key + three no-RNG scalars specified in idea §4
(pluggable/ablation-safe per AGENTS §9/§11). The two warnings are shared with
the parent and sibling. The only substantive qualitative caveat is that the
pre-registered "epoch-0 parent-level" framing is not what happened (all nodes
cold-start the head), so epoch-0 cannot be used as evidence for or against the
coupling; the coupling's effect, if any, is a slow mid-training re-shaping of
the val curve.

**Verdict: clean mechanics (0 errors, 32/32 ep, budget_hit false, best.pth
125.4 MB, 1704 s) but the idea's epoch-0 premise is refuted (164.2 vs 153.3
sibling / 147.0 parent) and the moved coupling (s_pos 0.0084, s_lin 0.0215,
b_cal −0.120) lands at final EMA val MAE 23.209 — 0.084 better than parent
N0001 (23.293) yet 0.219 above the registered 22.99 bar and 0.645 behind
sibling N0002 (22.564).**

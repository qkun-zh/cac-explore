# N0006_h0005 — qualitative feedback (text-log training-dynamics read)

- **Node:** N0006_h0005 (H0005 `use_verify`, solo) · **Parent:** N0002_h0001.
- **Sources read:** `idea.md` §4 inspection list; server TB scalars for child +
  parent (`val/mae`, `val/rmse`, `train/loss_epoch` full 32-epoch series);
  `hparams.yaml`; `best.pth` (`head.verify.*` weight norms, CPU-only);
  `model.py` gate wiring lines. No stored feature dumps exist (AGENTS §9);
  this read is text-log only. No heavy compute was run.

## 1. Clean vs overfit vs underfit: underfit with a late tail-overfit garnish

- The child's val curve tracks the parent closely for the first ~10 epochs
  (slightly below it, in fact), then separates mid-run and flatlines high
  while the parent keeps descending. There is no val U-turn in MAE — the
  child never finds a lower basin and then loses it; it simply stops
  improving around two-thirds through and sits on a flat floor to the end.
- Decisively, the child fits **train worse** than the parent as well
  (higher terminal `train/loss_epoch`, still creeping down at the end).
  Classic overfit is train-better/val-worse; here it is train-worse/val-worse.
  That is the signature of optimization drag from the added multiplicative
  path, not of memorization — the gate made fitting harder, not easier.
- Garnish: child `val/rmse` bottoms around two-thirds through and then creeps
  upward while MAE stays flat. So late training is still moving mass, and the
  movement concentrates in the tail (catastrophic images worsening while the
  median holds). Mild tail-overfit superimposed on a general underfit level
  shift — the reverse of the intended "sharpen the dense tail" effect.

## 2. Early-stop cleanliness: unclean edge pick on a plateau (moot)

- Child best is ep31-of-32 with the last ~5 epochs spread at noise level —
  a plateau sample, not an interior minimum. The parent's best (ep30-of-32)
  is also late but sits at the end of a still-descending slope, i.e. a real
  if edge-adjacent minimum. The child's "best" carries no selection meaning;
  any of the last several epochs would do equally well/badly.
- The `timeout` status is bookkeeping, not truncation harm: all 32 epochs
  completed and the curve had been flat for ~5 epochs, so no recovery was cut
  off. Nothing about the run shape suggests more epochs would close the gap.

## 3. Gate used vs ignored: used — and that is the point

- TB carries **no** `use_verify/*` diagnostic scalars, so the curve alone
  cannot answer this — but the checkpoint can, cheaply. Against the spec'd
  inits (prop2 zero-init, ver2 bias-0 for V=0.5 at step 0), the best-epoch
  weights have moved far: prop2 weight norm is well nonzero, ver2 weight and
  bias norms are well nonzero. The residual path learned a non-trivial
  function; V necessarily polarizes away from 0.5 somewhere. The optimizer
  did not discard the gate.
- Combined with §1, the reading is active-but-harmful: a live gate that drags
  both train and val. That closes the "gate ignored / optimization skipped
  it" branch of idea §4's falsification tree without further experiments.
- One learned value stands out as a suspect for synthesis (not a verdict):
  the exemplar-softmax temperature fell from 0.07 to ~0.02, i.e. the
  K-exemplar attention went near-one-hot. A peaky exemplar weighting makes
  the verification evidence brittle — consistent with a gate that fires
  confidently on the wrong cells — but this is a lead for the next hypothesis,
  not a finding.

## 4. Anomalies and process gaps

- `train/mae` is NaN for every epoch on **both** child and parent — a
  pre-existing logging bug, not node-specific; ignore, but it should be fixed
  so future qual reads have a train-side counting signal.
- `budget_hit=true` with all 32 epochs done and `best.pth` at ep31: the wall
  clock fired during post-epoch bookkeeping. Status `timeout` is honest per
  the runner contract, but qualitatively this run is a complete curve.
- Idea §4 promised prop2-norm and V-histogram inspection, yet the runner logs
  no gate scalars at all. That evidence had to be recovered from the
  checkpoint instead — worth adding `verify/prop_norm`-style scalars to the
  runner so the next gated proposal is diagnosable from TB directly.
- Wiring verified in `model.py`: single gated residual line into `cond_map`,
  Verify constructed after all parent modules, decoder width untouched —
  no structural anomaly; the failure is dynamical, not a wiring bug.

## One-line assessment

- Live gate, dead result: the verification residual trained (nonzero prop2 /
  ver2, peaky temperature) yet dragged both train and val onto a flat floor
  with tail-RMSE creep — an active-but-harmful underfit, edge-best at ep31,
  pointing at brittle repetition evidence rather than a dead module.

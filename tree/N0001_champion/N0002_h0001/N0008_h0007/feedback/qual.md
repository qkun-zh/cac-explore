# N0008_h0007 — qualitative feedback (text-log training-dynamics read)

- **Node:** N0008_h0007 (H0007 `use_simcal`, solo on `use_simprior=true`) · **Parent:** N0002_h0001 (22.5641 @ep30).
- **Sources read:** `idea.md` §5 inspection list; `result.json` (best 27.6371 @ep9, 32ep, `budget_hit=false`, `code=ok`); server text log `/tmp/train-b3-n0008.log` (142K, single-grep reads); server TB scalars full 32-epoch series (`val/mae`, `val/rmse`, `train/loss_epoch`, `train/mae`); parent N0002 `val/mae` series for shape contrast; `model.py` gate wiring lines. No stored feature dumps exist (AGENTS §9); this read is text-log only.

## 1. Trajectory shape: tracks-then-separates, then a hard NaN wall — not overfit

- Epochs 1–5 the child tracks the parent almost exactly (child vs parent `val/mae`: 139.9/153.3, 67.0/70.1, 48.2/48.8, 39.3/39.1, 33.9/33.5). That is the zero-init contract working: step-0 forward starts at the parent and the first ~5 epochs learn the same basin.
- Epochs 6–9 it separates monotonically: 30.83/30.30 (+0.5), 29.16/28.41 (+0.75), 28.22/27.27 (+0.95), 27.64/26.29 (+1.35). The gap widens every epoch even before anything blows up — the child was already losing while still numerically healthy, and decelerating (gains per epoch shrink: −2.0, −1.7, −0.9, −0.6).
- Epoch 10–32 is not "worsening with training" in any overfit sense: `val/mae`, `val/rmse`, AND `train/loss_epoch` all go NaN simultaneously at ep10 and stay NaN for the remaining 23 epochs. Train-side collapse rules out a val-metric glitch — the forward/loss itself blew up. There is no U-turn, no plateau, no late recovery to discuss; the run died at ep10 and the harness kept stepping a corpse to ep32.
- So "early best @ep9" here means **last-healthy-epoch best**, not a selected minimum on a complete curve. The recorded best (27.64) is honestly the pre-collapse peak and `best.pth` is the ep9 checkpoint, but everything after ep9 is uninterpretable — 23 wasted epochs, not a training signal.

## 2. Clean vs dirty: DIRTY run, usable checkpoint

- Dirty by the TB record: a synchronized train+val NaN wall covering 70%+ of the run. Status `done` / `code=ok` with `budget_hit=false` is honest bookkeeping (all 32 epochs stepped, best recorded pre-collapse), but qualitatively this is not a complete curve — it is 9 healthy epochs plus noise.
- The ep9 best itself is still usable as a paired-contrast number (it predates the blow-up and comes from the same seed/protocol), and at +5.07 over the parent it misses the 22.26 bar by miles even before the collapse. The dirt does not rescue or condemn the hypothesis verdict — the failure was already legible at ep6–9 — but it does mean no late-epoch shape (plateau, convergence, tail behavior) can be read from this node.
- Pre-collapse training looked otherwise orderly: `train/loss_epoch` 41.9 → 5.84 by ep8, no spikes in the healthy window; smoke passed (31.3M total, 3.5M trainable).

## 3. Two pre-collapse tells worth logging (shape only, no mechanism verdict)

- **RMSE/MAE decoupling ep6–9:** `val/rmse` bottoms at ep6 (95.48) then rises two epochs running (95.84, 96.63, 97.47) while `val/mae` keeps falling to ep9. Late training was already moving mass the wrong way on the tail while the median still improved — the same "tail-first distress" signature seen on the sibling, here as a prelude to blow-up rather than a plateau garnish. Left for the causal read to attribute; as pure shape it says the selection dynamics were straining before they broke.
- **The text log is silent about the collapse.** Single-grep anomaly scan (270 hits) returns only the three standing benign warnings (151 modules in eval mode — the frozen backbone, expected; `MeanMetric.compute` before `update`; non-writable NumPy array) plus routine CUDA/TensorCore tips. No loss-spike line, no exception, no OOM, no gradient-overflow message — the progress-bar lines carry no per-epoch scalars and only the final line prints `val/mae: nan`. The blow-up is visible ONLY in TB scalars, not in the text log. Anyone re-reading this node from logs alone would see a clean `max_epochs=32 reached` + `code=ok` and miss that 23 epochs are NaN.

## 4. Anomalies and process gaps

- Synchronized NaN onset at ep10 across train loss + both val metrics: points at forward/loss numerics, not evaluation code. (Which op — the causal read owns that; this read only notes the synchronization.)
- `train/mae` is NaN for all 32 epochs — pre-existing logging bug already noted on the sibling/parent, not node-specific; still unfixed, still depriving qual reads of a train-side counting signal.
- Server `run/latest/` holds `best.pth` + TB but no `result.json` (the record lives at the node root / local copy) — minor sync artifact, not load-bearing, noted so the next reader does not go hunting.
- Runner-logging gap (repeat from sibling, now load-bearing): with zero gate scalars logged, a silent scalar-collapse like this one leaves no text trace. At minimum a per-epoch `val/mae` text line would have made ep10 visible without TB parsing.

## One-line assessment

- Dirty run with an honest pre-collapse best: 9 healthy epochs that already trailed the parent by a widening +0.5→+1.35 gap with RMSE/MAE decoupling, then a silent synchronized train+val NaN wall from ep10 — 27.64 @ep9 stands as the usable number (a clean miss of 22.26), the last 23 epochs are uninterpretable, and the text log alone hides the collapse entirely.

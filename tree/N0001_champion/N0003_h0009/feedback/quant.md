# Quantitative feedback — N0003_h0009 (H0009, channel-gate Condenser)

## 1. Verified numbers
| Quantity | Value | Source |
|---|---|---|
| best val MAE | **22.607566833496094** | result.json `best_mae` |
| best epoch | **32** (final epoch, = max_epochs) | result.json `best_epoch` |
| epochs done | 32 / 32 | result.json `epochs`, `n_epochs_done`; `Trainer.fit` stopped `max_epochs=32` (run_node.log:552) |
| budget_hit | **false** | result.json `budget_hit` |
| elapsed | **1718.86 s** run_clock (result.json `elapsed`); 1724.1 s wall (`elapsed_s`) | result.json |
| budget ceiling | τ_max 1800 s → margin **≈81 s (4.5%)** | AGENTS.md §4 |
| params TOTAL | **31.4 M** | run_node.log:35 (model summary) |
| Trainable / frozen | **3.5 M / 27.8 M** | run_node.log:33-34 |
| param gate | 31.4 M < 32 M OK | config.toml `max_params_M` |
| channel-gate params | **33,024 (0.033 M)** = Linear(128→256), w 128·256 + b 256 | recomputed from model.py:158-160 (D=d_fine=128, embed_dim=256); parent model.py has no channel_gate → this is the full delta |
| smoke | `[smoke] one step fits: ok (model=Counter)` | run_node.log:24 |

Seeded same-protocol parent comparator: **PENDING** (measured on GPU server; no number yet — not fabricated here).

## 2. Val/mae learning curve
The run log flushes only the FINAL progress bar — `Epoch 31/31 … val/mae: 22.608` (run_node.log:553-554). The full per-epoch series was recovered from TB scalars (`tb/t/events.out.tfevents.1789794368…4309.0`), whose last scalar (step 7295 = 22.6076) exactly matches result.json, so the series and the recorded result are internally consistent:
- Fast descent: 168.23 → 39.98 → 27.63 at epochs 1, 4, 8.
- Monotone non-increasing for ALL 32 epochs (every epoch improves MAE) — no val plateau or overfit visible in the tail.
- Tail is decelerating but NOT flat: 22.7741 (E26) → 22.7210 (E27) → 22.6743 (E28) → 22.6397 (E29) → 22.6250 (E30) → 22.6141 (E31) → 22.6076 (E32). Last three steps ≈ −0.011, −0.008, −0.0065 per epoch.
- Side effect worth flagging: val/RMSE bottoms at E26 (88.763) then RISES to 89.784 (E32) — the tail's MAE gain grows density spread. MAE-only checkpoint selection (best_checkpoint.py, best.pth via val/mae) ignores this.

## 3. Δ position vs the pre-registered bar
Falsifier (idea.md:7): support requires final val MAE **≥ 0.30 lower** than the unconditional condenser (parent) ⇒ Δ = parent_best − child_best ≥ 0.30, child = 22.6076, i.e. parent must be **≥ 22.9076**.
- Observed child 22.6076 is HIGHER than the only currently published parent number (migration artifact val_mae **19.647**, result.json — UNSEEDED, different protocol). Using that artifact only, Δ ≈ **−2.96** (child ~2.96 WORSE), far below the +0.30 bar → consistent with DISPROVED.
- That artifact is explicitly non-comparable (result.json notes "predates seeding and is not bit-reproducible"; best 21.94@E13 in an unseeded 48ep probe). The runtime training behavior of the gated head (22.61 at E32, still descending) could depend on the comparator landing wherever the seeded rerun puts it.
- Decision stays open at synthesis: plug the pending seeded-parent best val MAE into Δ = parent − 22.6076; support iff Δ ≥ 0.30 (parent ≥ 22.9076). Current evidence, on the 19.647 artifact as a placeholder, points toward refutation — but that placeholder is not the gating comparator.

## 4. Honesty notes
- **Best-at-last-epoch + near-budget is the frontier** between two risks: (a) budget stopped the run ~81 s (4.5%) short of τ_max — if the GPU had been slightly slower, the run would have been truncated mid-epoch and the "best" would be E31 (22.6141); (b) conversely the curve is still descending ~0.01/epoch, so 22.6076 is likely a mild underestimate of what 2-3 more epochs would give. Either way the reported number sits at the protocol's very edge.
- **Best == last epoch** means best_mae is trivially "the final checkpoint's MAE", not a genuine early-or-mid selection. Monotone improvement makes this expected, but best-at-E32 means the number carries zero robustness margin from checkpoint selection.
- Only one progress-bar line flushed to the log; per-epoch claims rest on the TB scalar file (tag `val/mae`, logged on_epoch in pl_module.py:100). A second TB event file (`…3367.0`, earlier ts) holds a divergent val/mae trace (5128→484 over 32 points) from a separate/earlier write in the same dir — ignored here; worth a runner-side check that two event writers coexist cleanly.
- `train/mae` is logged **NaN** (torchmetrics MeanMetric computed before update — the warning at run_node.log:15); train-side MAE is unusable, val MAE unaffected (val metric is updated then computed, pl_module.py:86-104).
- Metric definition: val MAE = mean over val batches of |Σdensity − count| (pl_module.py:90-91), RMSE = sqrt-mean of squared error; best.pth selected on val/mae only (best_checkpoint.py:21-29).
- Checksums recorded: config a5ea5d5daf44acbb, model 976e313c971d2531 (result.json) — reruns of the node are byte-checkable.
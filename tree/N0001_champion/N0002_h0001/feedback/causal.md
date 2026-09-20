# Causal feedback — N0002_h0001 (H0001 simprior) — mechanism attribution

- **Node:** N0002_h0001 · **Parent:** N0001_champion · **Switch:** `use_simprior` (child `config.toml:27=true`; parent config has no such key, `tree/N0001_champion/config.toml:23-26`)
- **Outcome:** child best val MAE 22.5641 @ep30 vs parent 23.2932 @ep26 (`tree/N0001_champion/N0002_h0001/result.json:5-6`, `tree/N0001_champion/result.json:5-6`) → Δ = **0.7291**, falsifier bar (≤22.99, `idea.md:4`) cleared with 0.43 margin.
- **Verdict: supports-mechanism.** The gain is attributed to the explicit similarity prior being *used* by the trained model, not to generic capacity or optimizer luck. One residual confounder (extra-gradient-path regularization) cannot be excluded from this single run; the separating test is named in §3.

## 1. Similarity prior vs generic capacity / optimizer luck

- **Optimizer luck is ruled out by construction.** Both runs use seed 20260830 (`config.toml:9` in both), `deterministic=true`, `augment=false`, cosine/AdamW/MSE(+0.4·L1) identical (`config.toml:36-55` both files line-for-line except the one switch line). AGENTS §6 fixes same-seed paired-contrast precision at ~0.02; Δ=0.7291 is ~36× that. `budget_hit=false`, 32/32 epochs both (`result.json:6-9` both). There is no RNG path left for "luck": SimPrior construction is appended after all parent modules (`model.py:236-242`), forward has no stochastic ops (`model.py:212-221` — softmax/einsum/norm only), and the flag itself consumes no RNG (`model.py:158-159`).
- **Zero-init makes step-0 ≡ parent, so the gain had to be learned.** `SimPrior.out` weight and bias are zero-initialized (`model.py:208-210`), hence `delta==0` at init and flag-off forward is the parent's byte-for-byte (`model.py:169-173`). Any divergence is learned trust in the prior, not an init lottery. Corroborated by the curves: child val MAE is *worse* at ep1 (153.29 vs 147.0) and takes the lead only from ep5 onward (33.50 vs 34.12), then leads for all 28 remaining epochs — TB `val/mae` series, child run `.../N0002_h0001/run/latest/tb/t`, parent run `.../N0001_champion/run/latest/tb/t` (full per-epoch series reproduced in §4). A lucky init would show an early offset, not a learned crossover plus a widening late gap (ep18: 22.92 vs 23.72, Δ=0.80).
- **"+24.8k params of generic capacity" cannot explain it architecturally.** The 24,769-param delta (`idea.md:85-90`: 8192+16384+1+192; shapes confirmed at `model.py:203-210`) is bottlenecked through exactly **2 evidence channels** (`top1`, `cons`, `model.py:218-220`) projected into `cond`. Those params can only express exemplar-cosine functions of `(fine, e)` — they add zero arbitrary decoder capacity (decoder `in_ch` stays 192, `model.py:156`; decoder/fuser/exemplar/condenser constructors untouched per the `diff` in §4). Generic-capacity stories require the extra params to be able to fit arbitrary residuals; here they cannot.
- **The prior is demonstrably used, not ignored.** Checkpoint inspection (`/data/cac/tree/N0001_champion/N0002_h0001/run/latest/best.pth`, keys `ckpt["model"]`, 263 entries; `best_mae`=22.5641 `epoch`=30 in the ckpt header):
  - `head.simprior.out.weight` (64,2,1,1) norm **0.5874** (RMS 0.0519/param) vs **0.0 at init** — the trust projection moved decisively off zero.
  - Both evidence channels used **equally**: ch0 (top1) norm 0.4158 vs ch1 (consensus) norm 0.4150. The optimizer kept *both* the sharp-detection and the agreement signal, matching the `idea.md:79` design rationale.
  - `head.simprior.out.bias` norm 0.0730 (mean −0.0011) — small, i.e. the gain rides on the evidence weights, not a learned constant offset into `cond`.
  - `head.simprior.temp` **0.0393** vs init 0.07 (`model.py:207`) — the softmax-over-K sharpened ~44%, i.e. learned calibration of the kind CACViT prescribes (`idea.md:40`), not an untouched hyperparameter.
  - `qproj` norm 5.73 (std 0.0633), `kproj` norm 7.35 (std 0.0575) vs ~4.62 expected under PyTorch default uniform init (bound 1/√fan-in) — both grew past init scale, so the projection geometry was learned, not frozen.
  - Falsifier logic from `idea.md:104-105` said a refute-with-use would look like proj norm ≈0; the opposite is measured on every SimPrior parameter.

## 2. Trained-weight inspection (feasible — done on server CPU)

- Script: `/data/miniconda/envs/cac/bin/python` + `torch.load(..., map_location="cpu")`, <30 lines inline; layout is `ckpt["model"]` (OrderedDict, 263 keys), **not** `state_dict` (first probe printed `NSIMKEYS 0` under that assumption; corrected probe found 5 `head.simprior.*` keys). All numbers above are measured, not inferred.
- Summary table (norms, float32, CPU):

| param | shape | init | trained | reading |
|---|---|---|---|---|
| `head.simprior.temp` | () | 0.07 | 0.0393 | sharpened, calibrating |
| `head.simprior.out.weight` | (64,2,1,1) | 0.0 | 0.5874 (ch0 0.4158 / ch1 0.4150) | mechanism active, both channels |
| `head.simprior.out.bias` | (64,) | 0.0 | 0.0730 | negligible offset |
| `head.simprior.qproj.weight` | (64,128,1,1) | ~4.62 (exp.) | 5.7314 | learned past init |
| `head.simprior.kproj.weight` | (64,256) | ~4.62 (exp.) | 7.3547 | learned past init |

## 3. Confounders and what remains open

- **Excluded:** seed/loader/cudnn nondeterminism (fixed seed + deterministic, §1); init lottery (zero-init, §1); GCA/XScale-path drift (§4); decoder-capacity change (in_ch 192 unchanged); budget/early-stop artefacts (both 32/32, `budget_hit=false`).
- **Remaining alternative:** the additive residual gives the decoder's `cond` path extra gradient flow during training, which could regularize/feature-shape the shared Condenser even if the cosine content were secondary. Against it: the learned temperature sharpening and the equal two-channel use are content-specific signatures (a pure-regularization path would not need calibrated sharpening of the K-softmax, `model.py:217`), and `qproj`/`kproj` growing past init scale means the geometry itself was optimized. But weight norms alone cannot fully separate "matching signal" from "helpful gradient path".
- **Separating test (future node, not this one):** freeze `qproj`/`kproj`/`temp` at init and train only `out` (tests whether *learned* geometry matters), or train the identical architecture with exemplar rows shuffled across the batch (destroys match content, keeps gradient-path regularization). Gain persisting under shuffle ⇒ regularization story; gain vanishing ⇒ matching-signal story. Per AGENTS §11 a revisit must book a new falsifier, not silently retry H0001.

## 4. Interaction with GCA/XScale (untouched — no sign of attribution there)

- `diff tree/N0001_champion/model.py tree/.../N0002_h0001/model.py` shows additions only: the `use_simprior` flag (`model.py:158-159`), the `cond_map` residual add (`model.py:168-173`), the `SimPrior` class (`model.py:193-221`), and its append-only construction (`model.py:236-242`). `FineFuser`, `ExemplarEncoder` (incl. XScale lines 104-107), `Condenser`, `DensityDecoder`, `GCA` are byte-identical.
- Config diff is one line: `use_simprior = true` (`N0002_h0001/config.toml:27`); `use_gca=true, use_ddca=false, use_xscale=true, xscale_size=3` identical (`config.toml:23-26` both files).
- SimPrior reads `e`/`fine` read-only and never modifies them (`model.py:212`), so the GCA input (`GAP(fine)+e_mean`, `model.py:186-190`) is unaffected by construction; XScale pooling (`model.py:104-107`) is upstream of and untouched by the prior.
- Curve corroboration (per-epoch `val/mae`, child vs parent): ep1 153.29/147.0, ep5 33.50/34.12, ep10 25.41/25.35, ep14 23.56/24.04, ep16 23.25/23.92, ep18 22.92/23.72, ep22 22.89/23.46, ep26 22.65/23.29, ep30 22.56/23.32, ep32 22.57/23.34. Parent plateaus ep24-32 (23.29→23.34, overfit flat); child descends monotonically to ep30. `val/rmse` best: child 88.92 @ep28 vs parent 90.64 @ep24; parent RMSE rises 90.64→91.75 over ep24-32 while child's holds ~89.0 — the prior improves RMSE *and* MAE with matched train-loss trajectories (final `train/loss_epoch` 2.62 vs 2.83), i.e. generalization, not faster fitting. RMSE/MAE at best (3.95 vs 3.90) is essentially unchanged, so the gain is broad-based rather than catastrophic-image-only — consistent with a dense per-cell prior, with no GCA-anchored count-scale disruption.

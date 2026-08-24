# DISTILLED — Research State as of 2026-08-24 EOD
> One-stop summary. Read THIS instead of excavating node dirs. Historical detail: `tree/archive_2026-08-24/`, `journal/events.jsonl`. Ledger: `memory/hypotheses.jsonl` (H0030–H0039 booked).

## Numbers that matter
| Artifact | val MAE | RMSE | Note |
|---|---|---|---|
| Champion ckpt `/data/runs/N0021_dino_partialft/best.pth` | 20.438 | 79.86 (last-ep) | 23.11M, 40ep/24min |
| + routing readout (H0036) | **19.18** | **66.37** | eval-only; split-half honest ~19.2; threshold plateau 150–300 |
| N0027 norm+flip+SWA | 20.403 | — | tie; converges faster only |

## Confirmed levers (cumulative)
DINOv2-S/reg4 substrate · taps(6+11)+area-prompt · partial FT blocks10-11@lr×0.1 · resolution: higher-res eval cuts tail error monotonically (quantization floor mechanism, H0035 ✓) · self-routing N̂@392≥200→518px (H0036 ✓)

## Refuted — do NOT retry
Full FT (drift collapse ×2) · EBC-paradigm transfer of partialFT (N0024 early-stop) · per-token gate · Huber · highres decoder-as-model · seqcount AR · tail-reweight ± · proto-iterative · scale-deform · point detect · mosaic-in-domain · TT-Norm exemplar-gain calibration (**root cause: head has NO per-object mass anchoring** — exemplar-box ρ̂ mass ≈0.03–0.19, not ≈1) · global isotonic count recalibration (cross-fit hurts both halves) · trimmed-sum readout · SWA/EMA weight averaging (+0.68 worse; epochs drift, don't bounce) · input-norm fix + flip aug (joint arm −0.17 = noise; faster convergence but same endpoint)

## Key measured facts (drive everything)
1. **75.9% of SSE sits in 17 val images with N≥500** (bucket table in `tree/nodes/N0026_res_sweep/res_results.json`). Bulk images are near-floor (~4–6 MAE contribution).
2. Tail error ↓ monotonically with eval res (487→406→319 @392/448/518). 448 alone: RMSE −8 at zero cost.
3. Head mass scale is unanchored → any "calibrate on exemplar regions" trick fails until anchoring exists.
4. Exemplar conditioning is scalar-only (box[0] of 3, Fourier+area token) — the architectural gap vs all sub-11-MAE published methods (they inject exemplar FEATURES spatially).
5. result.json headline `mae` = LAST epoch; use `best_mae`. Known engine bug, unfixed.

## OPEN PROPOSALS (specified, never materialized — pick up here)
### P1. N0028_scb_multires ← strongest incremental, target routed ≤15.9
Parent: N0027_norm_flip_swa. Two coupled changes:
(a) SCB-lite residual exemplar gating on gated tap tokens: pool each of 3 boxes on ps×ps grid → proj Linear(384,384) → cosine sim vs tokens → softmax over boxes → context e_ctx; `tokens += γ·sigmoid(MLP([tokens,e_ctx]))·e_ctx`, **γ=0 init ⇒ step-0 == parent**. +1.2M params → ~24.6M.
(b) multi-res joint training: second train loader @518 bs4, `loaders[ep%2]` alternation; eval@392 + existing dual_res rider@448. Data: fsc147.py must emit bboxes3[3,4] (same scaling/flip as bbox at :57-61).
- H0040 (SCB): best ≤ parent−1.0 AND median exemplar-box ρ̂-mass >0.25; DISPROVED IF ΔMAE >−0.3 or mass <0.15
- H0042 (multires): routed-readout ≤18.2 AND tail[500+) ≤350@518-arm; DISPROVED IF routed ≥19.2 OR bulk[0,75) degrades >+0.5
- Ladder: R0 smoke incl. γ-drift check → R1 main (~45min) → R2 attribution arms only if R1 wins → R3 re-run eval labs + refit routing threshold split-half on NEW ckpt
### P2. COIN (physics zero-base) — R0 kill gate first, no training
Exemplar-conditioned cosine similarity map S(x) from frozen backbone = pulse train; treat counting as inverse particle-sizing (Coulter coincidence correction: multiplicity gates score merged blobs k=1..4 via flux/area/inertia features; Poisson-NLL occupancy loss replaces MSE; variance-channel second estimator Fisher-fused). Gain path claim: 19→14→11→10. **R0 (<30min): Spearman ρ(Σ_x S(x), GT count) > 0.75 on ~300 val imgs else kill.** Reuse eval_readout_lab.py loading patterns.
### P3. AXIOM-TTC (systems zero-base)
Test-time training on label-free counting axioms: crop-additivity, scale-integral invariance, exemplar-swap invariance + trust-region anchor (KL to init kills zero-collapse). LoRA+LN gains ~0.5M adaptable params, ~30 steps/image. Claimed −1.5..−2.5 from a strong base. **R0 drift audit on OUR champion ckpt: corr(|count drift under zoom/crop/swap|, |error|) ≥ 0.3 else kill.**

## Server gotchas (not yet in failure_modes.md)
- ~~No tmux~~ **RESOLVED 2026-08-24**: installed tmux 3.6a into base miniconda (`conda install -y -c conda-forge tmux`; no sudo on box). Non-login ssh shells need `export PATH=/data/miniconda/bin:$PATH` before `tmux`/`run_node.sh`. libtinfo.so.6 version warning is benign (sessions verified working). Fallback if env is ever rebuilt: `setsid nohup ... </dev/null > log 2>&1 & echo PID_$!`
- scp into /data/repo creates untracked files that BLOCK later git pull → move aside, pull, diff-verify, delete backup
- `pkill -f "train.py"` self-matches your ssh command line → use `[t]rain.py`
- Engine SWA/dual-res riders exist since N0027 (backward-compat, flags default off)

## File map (post-distillation)
- Alive nodes: `N0021_dino_partialft` (champion), `N0027_norm_flip_swa`, `N0025_eval_readout` (lab), `N0026_res_sweep` (per-image dump backing H0035/36)
- Dead ends + stale cards: `tree/archive_2026-08-24/`, `tasks/archive/`
- Eval tools: `scripts/eval_readout_lab.py`, `scripts/eval_res_sweep.py`; gates: `check_hypothesis.py`, `novelty_check.py`, `calibration_report.py`

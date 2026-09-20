# Diagnostic — N0006_h0005 (H0005 `use_verify`) · timeout-status triage

- **Node:** N0006_h0005 · **Parent:** N0002_h0001 (22.5641 @ep30) · **Status:** `timeout`
  (`budget_hit: true`, `elapsed` 1809.83 s vs τ_max 1800 s → **+9.8 s, +0.5%**),
  `n_epochs_done` 32/32, best **23.5802 @ep31** (info.json `train_seconds` 1815.4,
  `best_metric` 23.5802 — matches server TB min exactly).
- **Question asked:** true timeout pathology (extra-compute budget blow, e.g.
  N0005_h0004) vs marginal overrun with complete + valid result (N0004 precedent).
- **Verdict: MARGINAL OVERRUN — not a true timeout pathology.** The result is
  complete, converged, and usable for the H0005 verdict. Details below.
- **Sources read (all verified, no recollection):** local `info.json` /
  `result.json` / `idea.md` / `config.toml` / `model.py` for N0006; parent N0002
  and sibling N0004_h0003 / N0005_h0004 `info.json` + `result.json`; server TB
  `val/mae` scalars + wall times for all four runs via single-shot ssh
  (no sleep-loops, no tmux; no training tmux survives — `no server running`).

## 1. Timing anatomy (server TB wall clocks — the load-bearing numbers)

| Run | TB span (ep0→last val) | /31 intervals | result.json elapsed | Status | Best |
|---|---|---|---|---|---|
| N0002 parent | 1680.7 s | **54.21 s/ep** | 1749.5 s | done | 22.5641 @ep30 |
| N0004_h0003 | 1698.8 s | **54.80 s/ep** (+0.6) | 1767.6 s | done | 25.5661 @ep13 |
| N0005_h0004 | 1737.7 s | **56.06 s/ep** (+1.85) | 1809.7 s | timeout | 48.1201 @ep14 |
| N0006_h0005 | 1738.6 s | **56.08 s/ep** (+1.87) | 1809.8 s | timeout | 23.5802 @ep31 |

- Parent headroom to ceiling was ~50 s (1800 − 1749.5). N0006's uniform
  ~1.9 s/ep overhead × 32 ≈ 61 s consumed that headroom plus 9.8 s. The overrun
  is **accumulated uniform drag, not a blow-up**: N0006 per-epoch wall deltas
  are 56.6–57.4 s from epoch 0 onward — flat, no progressive slowdown, no tail
  spike. (Final interval is 29.7 s / +96 train steps vs the usual +228: the
  `_BudgetStop` fired mid-final-epoch and the run logged its 32nd val point on
  a truncated epoch — see §3.)
- **N0004 precedent holds:** N0004 absorbed +18 s vs parent and stayed `done`;
  N0006 needed +60 s and missed by 10 s. Same phenomenon, different side of the
  line — not a different failure mode.

## 2. Is the ~19.7k Verify module the overrun cause, or noise? → NOISE-LEANING, UNATTRIBUTABLE

- The booked cost claim (`idea.md` §2c/§3) is ≈0.1 GFLOPs/ep-image, "<1% epoch
  impact". Observed overhead is ~+3.4% / +1.9 s/ep — larger than claimed, but
  **no measurement here isolates Verify**:
  - N0005_h0004 (a *different* module, simbank) shows a near-identical +1.85
    s/ep overhead. Two different code deltas producing the same slowdown to
    ±0.02 s/ep points at a **common cause outside the modules** (run-window
    server contention, data-loader/host noise) rather than Verify-specific
    einsum/activation cost.
  - The drag is present from epoch 0 at full magnitude — consistent with either
    a constant per-step surcharge or a constant environmental offset; timing
    alone cannot separate them.
  - No progressive growth rules out a memory-leak / activation-blowup pattern
    (the N0005_h0004-style pathology would show super-linear deltas, not flat
    57 s).
- **Conclusion:** do NOT book "Verify einsum is expensive" as a finding. The
  ~1.9 s/ep is within run-to-run noise until a controlled same-window A/B says
  otherwise. No action for the Verify design on cost grounds.

## 3. Is best 23.5802 @ep31 trustworthy despite `timeout`? → YES

- Full 32-point server val curve (steps 227→7163): 136.72 → 66.94 → 47.61 →
  38.39 → 33.16 → 30.19 → 28.02 → 26.54 → 25.64 → 25.10 → 24.88 → 24.61 →
  24.47 → 24.26 → 24.04 → 23.90 → 23.86 → 23.84 → 23.71 → 23.67 → 23.78 →
  23.77 → 23.74 → 23.71 → 23.68 → 23.67 → 23.64 → 23.589 → 23.581 → 23.581 →
  **23.580** → 23.581. Smooth, monotone-ish descent, plateau from ep26.
- Plateau levels: ep28–ep32 all within **0.001** (23.581/23.581/23.580/23.581).
  Best@ep31 is the min of a flat 4-epoch plateau, with the following point
  +0.0009 — **not an edge fluke**, any plateau epoch gives the same number to
  the 0.30-bar precision that matters (§6 semantics: precision ~0.02).
- Caveat (honest, non-blocking): the ep32 point sits on a truncated final epoch
  (+96 steps, 29.7 s wall — `_BudgetStop` mid-epoch). It is still a real EMA
  validation, and its agreement with the plateau corroborates rather than
  undermines best@ep31.
- Checkpoint: server `best.pth` (125,565,587 B, mtime 16:29:06, i.e. run end)
  exists; `train/mae` NaN-tail is a pre-existing logger artifact — parent N0002
  shows identical all-NaN `train/mae`, so it carries zero diagnostic signal.
- **Contrast with true pathology N0005_h0004:** diverged best 48.12 @ep14
  (garbage, +25 over parent) + timeout = wasted budget, unusable result. N0006
  is the opposite on every axis: converged, interior best, smooth curve, saved
  checkpoint. `budget_hit: true` here is a wall-clock margin technicality.

## 4. Implication for the H0005 verdict (for Synthesis; evidence booking stays Lead-only)

- Global bar: 23.5802 vs required ≤ 22.26 (parent 22.5641 − 0.30) → **misses by
  +1.32 MAE**, i.e. 44× the ~0.02 paired-contrast precision and *worse* than
  the parent by +1.02. The timeout does not soften this: the best is plateau-
  confirmed, so a few more seconds/epochs had no path to 22.26 (plateau slope
  ≈ 0.0005/ep over the last 4 epochs).
- The AND-clause (gt>500 dense-tail mean |Δ| vs 511.7 baseline) belongs to the
  qual/causal agents; on the global clause alone H0005 already fails its
  DISPROVED-IF. The mechanism read ("P×V sharpening restores dense recall")
  gets no support from a +1.02 regression — gate-diagnostic follow-ups (prop2
  norm, V polarization) are optional forensics, not verdict-blockers.
- Recommendation: treat N0006 as a **valid completed 32-epoch contrast**
  (status word `timeout` notwithstanding) and refute H0005 on the global bar
  unless qual/causal finds a dense-tail win large enough to force a re-read —
  which the +1.02 global regression makes implausible.

## 5. What NOT to do

- Do NOT re-run N0006 to "get a `done` status" (AGENTS §5.1 — outcome shopping;
  the number is already plateau-robust).
- Do NOT book a "Verify is over budget-cost" hypothesis on this evidence
  (unattributable per §2; N0005 parity implicates environment).
- Seed stays 20260830; no cross-seed anything (AGENTS §6).

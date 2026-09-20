# Diagnostic — N0009_h0008 (H0008 `use_hires`) · timeout-status triage

- **Node:** N0009_h0008 · **Parent:** N0002_h0001 (22.5641 @ep30) · **Status:** `timeout`
  (`budget_hit: true`, runner `elapsed` 1809.75 s vs τ_max 1800 s → **+9.7 s, +0.5%**;
  `elapsed_s`/`train_seconds` 1815.2 s → +15.2 s wall incl. runner overhead),
  `n_epochs_done` 32/32, best **26.4535 @ep31** (info.json `best_metric` matches
  result.json `best_mae` exactly).
- **Question asked:** true timeout pathology (extra-compute budget blow, e.g.
  N0005_h0004; catastrophic interference, e.g. N0007_h0006) vs marginal overrun
  with complete + valid result (N0006_h0005 precedent).
- **Verdict: MARGINAL OVERRUN — not a true timeout pathology.** The result is
  complete (32/32 epochs, 32 validation points), late-best, and usable for the
  H0008 verdict. Details below.
- **Sources read (all verified, no recollection):** local `info.json` /
  `result.json` / `idea.md` / `config.toml` / `model.py` for N0009; parent N0002
  plus siblings N0005_h0004 / N0006_h0005 / N0007_h0006 `info.json` + `result.json`;
  `memory/hypotheses.jsonl` H0005/H0008 lines; `journal/events.jsonl` tail (batch-4
  booking + N0006/N0007/N0008 run entries); N0006 `feedback/diagnostic.md` as method
  precedent. Server: single short ssh commands only per instruction (no sleep-loops,
  no tmux, no TB scalar dump, no best.pth probe): `ls` of `run/latest/` (best.pth +
  tb/t/events + hparams.yaml) and `grep -c "val/mae"` on the TB events file (= 32).
  TB scalar values, train-loss curve, and checkpoint norms were NOT read here —
  they belong to quant/causal.

## 1. Timing anatomy (runner `elapsed` — the load-bearing numbers under grep-only constraint)

| Run | result.json elapsed | /32 per-epoch | Status | Best |
|---|---|---|---|---|
| N0002 parent | 1749.53 s | **54.67 s/ep** | done | 22.5641 @ep30 |
| N0006_h0005 (verify, marginal) | 1809.83 s | **56.56 s/ep** (+1.89) | timeout | 23.5802 @ep31 |
| N0009_h0008 (hires, this node) | 1809.75 s | **56.55 s/ep** (+1.88) | timeout | 26.4535 @ep31 |
| N0005_h0004 (simbank, blow-up) | 1809.66 s | 56.55 s/ep (+1.88) | timeout | 48.1201 @ep14 (null) |
| N0007_h0006 (h2pool, catastrophic) | 1803.67 s | 56.36 s/ep (+1.69) | done | 47.8439 @ep32 |

- N0009's overrun is **byte-for-byte the N0006 signature**: trainer-elapsed delta
  vs parent +60.2 s (N0006: +60.3 s); over-budget +9.7 s (N0006: +9.8 s);
  per-epoch +1.88 s/ep (N0006: +1.89 s/ep) — identical to ±0.01 s/ep. Parent
  headroom to ceiling was ~50 s (1800 − 1749.5); uniform ~1.9 s/ep drag × 32 ≈
  60 s consumed it plus ~10 s. Accumulated uniform drag, not a blow-up.
- **TB-span column unavailable here** (N0006 diagnostic had TB wall clocks; the
  single-grep constraint forbids a TB scalar dump). The runner-`elapsed` proxy is
  still decisive: N0006 showed TB-span and `elapsed` move together, and three
  different modules (simbank/verify/hires) now show the same +1.85–1.89 s/ep to
  ±0.04 s — see §2.
- No progressive-slowdown evidence is available either way from greps alone, but
  the blow-up pattern is positively excluded on levels: a N0005-style compute
  pathology wastes the budget on garbage (best 48.12 @ep14, uninterpretable null);
  N0009 spent the same seconds to complete all 32 epochs with a late best.

## 2. Is the ~15.4k HiResDetail module the overrun cause, or noise? → NOISE-LEANING, UNATTRIBUTABLE

- The booked cost claim (`idea.md` §2-low-cost) is ≈0.1 GFLOPs at 96×96, "no
  192-grid tensors, no einsum, no second qproj — the three budget killers",
  learned "inside τ_max parity". Observed overhead is ~+3.4% / +1.9 s/ep.
- **No measurement here isolates HiResDetail**, by the exact N0006 §2 argument:
  - N0005_h0004 (simbank, full-grid token-max einsum) and N0006_h0005 (verify,
    P×V gate) show near-identical +1.85–1.89 s/ep overheads. Three different code
    deltas producing the same slowdown to ±0.04 s/ep points at a **common cause
    outside the modules** (run-window server contention, data-loader/host noise)
    rather than hires-specific conv cost.
  - Code inspection confirms hires adds only 25ch 3×3 convs at 96×96 plus a
    48→96 bilinear guide lift (`model.py` HiResDetail, lines 238–270) — no
    mechanism for super-linear blow-up, and the identical N0005/N0006 drag
    corroborates that nothing module-specific fired.
- **Conclusion:** do NOT book "HiResDetail is expensive" as a finding. The
  ~1.9 s/ep is within run-to-run noise until a controlled same-window A/B says
  otherwise. No action for the hires design on cost grounds.

## 3. Is best 26.4535 @ep31 trustworthy despite `timeout`? → YES (with one honest caveat)

- Completion: 32/32 epochs done, and server TB events contain **32 `val/mae`
  occurrences** (single-grep count) — every epoch validated, best @ep31 has a
  real following point (ep32 evaluated and worse). This is the N0006 position
  exactly (best @ep31 of 32), not a truncated edge.
- `_BudgetStop` semantics (`runner.py`: fires `on_train_batch_end`, sets
  `should_stop`): with `n_epochs_done` 32 and 32 val points, the stop fired
  during/after the final epoch — best selection (`BestCheckpoint` over all
  evaluated epochs) is unaffected. `budget_hit: true` here is a wall-clock
  margin technicality, same as N0006.
- Late-best argues against instability pathology: the NaN-wall signature
  (N0008_h0007: best @ep9, wall ep10–32) would put the best early; N0009's best
  at ep31 is the opposite — a converged late minimum, consistent with a clean
  engaged-and-evaluated run.
- **Caveat (honest, non-blocking):** unlike N0006 §3, no plateau-flatness claim
  (e.g. "ep28–32 within 0.001") is made here — the TB scalar values were not
  dumped under the single-grep constraint. What is established (position +
  full completion + late best + 32 val points) already rules out truncation and
  the N0005/N0008 failure positions. Plateau confirmation and the ep32-agreement
  check belong to quant; they cannot overturn the verdict below (§4) because the
  margin is 200× the paired-contrast precision.
- Checkpoint: server `run/latest/best.pth` exists per `ls`. Its norms were not
  probed here (for Synthesis/causal).
- **Contrast with true pathologies:** N0005_h0004 diverged best 48.12 @ep14
  (garbage, +25.6 over parent) + timeout = wasted budget, unusable null.
  N0007_h0006 best 47.84 @ep32 (+25.28, train stall 13.93 vs 2.62 from ep1,
  shared-projection corruption). N0009 is the opposite on every axis that matters
  for usability: complete, late-best, mid-range degradation (+3.89), no stall/NaN
  position signature. Same verdict class as N0006, larger margin.

## 4. Implication for the H0008 verdict (for Synthesis; evidence booking stays Lead-only)

- Global bar: 26.4535 vs required ≤ 22.26 (parent 22.5641 − 0.30) → **misses by
  +4.19 MAE**, i.e. ~210× the ~0.02 same-seed paired-contrast precision, and
  *worse* than the parent by **+3.8894**. The timeout does not soften this: the
  best is late with a following evaluated epoch, so a few more seconds had no
  path to 22.26.
- The AND-clause (gt>500 dense-tail mean |Δ| vs 511.7 baseline) belongs to
  qual/causal; on the global clause alone H0008 already fails its DISPROVED-IF
  first prong decisively. A dense-tail win large enough to force a re-read is
  implausible under a +3.89 global regression (same logic as N0006 §4, stronger).
- Mechanism read (`idea.md` §2: "zero-init post-decoder detail pass restores
  small-peak sharpness without touching the confirmed readout"): a +3.89
  regression gives it no support. Code inspection confirms the run obeyed the
  post-N0007/N0008 ban by construction — detached logits + detached frozen-h2
  guide, own parameters only, no S recompute, no shared projection
  (`model.py` lines 183–185 routing, 249–270 module, 292–297 append-only
  construction; decoder in_ch stays 192; GCA untouched) — so this reads as
  mechanism rejection (null-or-harmful decoder residual), never as readout
  corruption. The null-vs-engaged split (head norm ≈ 0 with train ≈ parent vs
  engaged-but-worse) is for causal via best.pth; it does not block the refute.
- Recommendation: treat N0009 as a **valid completed 32-epoch contrast**
  (status word `timeout` notwithstanding) and book H0008 `contradicts` at
  **w ≈ 0.90** (stronger than N0006's 0.85 for +1.02 given the +3.89 margin;
  below the N0004/N0007 0.95 catastrophic level since there is no stall/NaN
  signature) unless qual/causal surfaces a dense-tail result that forces a
  re-read. Weight choice is Lead-only; this is a recommendation.

## 5. What NOT to do

- Do NOT re-run N0009 to "get a `done` status" (AGENTS §5.1 — outcome shopping;
  the number is late-best with full completion).
- Do NOT book a "HiResDetail is over budget-cost" hypothesis on this evidence
  (unattributable per §2; N0005/N0006 parity implicates environment).
- Seed stays 20260830; no cross-seed anything (AGENTS §6).

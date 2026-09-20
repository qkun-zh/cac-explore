# Synthesis — N0002_h0001 (H0001 `use_simprior`) — Synthesis subagent, node N0002_h0001

- **Node:** N0002_h0001 · **Parent:** N0001_champion · **Hypothesis:** H0001 · **Switch:** `use_simprior` (child `true`; parent default `false`) · **Composition:** SOLO single-switch.
- **Pre-registered falsifier (`idea.md:4-6`):** confirm iff child final val MAE ≤ **22.99** at seed 20260830 under the canonical 32ep/1800s protocol (live parent 23.293 @ep26; ≥0.30 lower).

## 1. Consolidated verdict: CONFIRM H0001

All three feedbacks agree. Numbers below are the single authoritative record (deduped; all traced to the two `result.json` files, TB series, and `best.pth` inspection cited per-row).

| Metric | Parent N0001_champion | Child N0002_h0001 |
|---|---|---|
| Best val MAE | 23.29315376 @ep26 | **22.56414604 @ep30** |
| Final val MAE (ep32) | 23.3364 (TB) | 22.5705 (TB) |
| Margin vs parent | — | **0.7290** (2.43× the 0.30 bar) |
| Clearance below 22.99 bar | — | 0.4259; bar first cleared **ep18** (22.9224), held eps 18–32 |
| Epochs / budget | 32/32, `budget_hit: false` | 32/32, `budget_hit: false`, 1755.0 s wall |
| Status | baseline | `done`, `code: ok` |
| `config_sha256` / `model_sha256` | 679f37c26bc72103 / eaf2a8b93c280cba | 4d4c9baec459f657 / 611be2a9d70768b5 |

- **Quantitative:** CONFIRM — 22.5641 @ep30 ≤ 22.99; same-seed (20260830) paired contrast, full 32-epoch canonical run, no anomalies (32/32 finite TB points, max late uptick +0.026, no `_BudgetStop`/traceback/OOM). Best-vs-final +0.0064 (no overfit uptick); child last-5-epoch range 0.0282. RMSE also improves (child best 88.92 @ep28 vs parent 90.64 @ep24) with RMSE/MAE ratio flat (~3.9), so the win is broad-based, not catastrophe-only.
- **Qualitative:** clean full-protocol pass — green smoke (31.3M total / 3.5M trainable), smooth spike-free descent, benign-only warnings, epoch-30 best checkpoint properly written (`best.pth` 125 MB, two minutes before `result.json`). One logging artifact only: `train/mae` TB series NaN throughout (metric-logging artifact; loss descends monotonically, carries no signal).
- **Causal:** supports-mechanism — gain attributed to the similarity prior being *used*, not generic capacity or optimizer luck. Zero-init makes step-0 ≡ parent (divergence had to be learned; child trails at ep1, leads from ep5, gap widens to Δ=0.80 by ep18). The +24.8k params are bottlenecked through exactly 2 evidence channels (cannot express arbitrary residuals; decoder `in_ch` stays 192). Checkpoint proof of use: `head.simprior.out.weight` norm **0.5874** (vs 0.0 at init), both channels equal (ch0 0.4158 / ch1 0.4150), bias norm 0.0730 (negligible offset), `temp` 0.07→**0.0393** (learned sharpening), `qproj`/`kproj` norms 5.73/7.35 past ~4.62 init scale. **One named residual confounder:** extra-gradient-path regularization via the additive `cond` residual cannot be fully excluded from this single run — the separating test is booked as a new hypothesis in §2 (Booking 2).
- **No misattribution found:** GCA/XScale/decoder/FineFuser/ExemplarEncoder/Condenser byte-identical per model diff; config diff is the single `use_simprior = true` line; nothing to remap or discard.

## 2. Quality gate (AGENTS step 8; K_SYNTH=2)

**(a) Format gate — outputs pasted verbatim:**

`python3 scripts/discovery.py validate --all`:
```
validated 2 hypotheses — 0 bad (all good)
```
(exit 0)

Booking 1 (adopted `local/ideas/h_refine.md:81`, byte-verified identical) — format gate via `validate()` + novelty:
```
A len: 545
A gate: ([], [])
top similarities: H0002=0.331, H0001=0.318
NOVEL (top sim 0.331 < 0.82)
```
(exit 0; no errors, no warnings, NOVEL)

Booking 2 (new, authored below) — format gate via `validate()` + novelty:
```
B len: 630
B gate: ([], [])
top similarities: H0001=0.418, H0002=0.292
NOVEL (top sim 0.418 < 0.82)
```
(exit 0; no errors, no warnings, NOVEL, no structural twin flagged)

**(b) Budget:** 2 bookings total = 1 adoption (`h_refine.md`) + 1 new ≤ K_SYNTH=2. Attributable to this node; nothing else booked here.

**(c) No duplicates of H0001/H0002:** Booking 1 is the decoder-side logit-refinement mechanism (survey F5 family; nearest ledger sim 0.331). Booking 2 is the shuffled-exemplar control (nearest ledger sim 0.418, no structural twin of H0001: different IF choice, different IN scope/switch, different THEN direction and falsifier). Neither re-encodes H0001's simprior-readout claim or H0002's signed-GCA claim; Booking 2 is explicitly NOT a retry of H0001 (control design, new falsifier, per AGENTS §11).

**(d) Remap-or-discard:** nothing misattributed (§1); no discard needed.

### Booking 1 of 2 — ADOPTED from `local/ideas/h_refine.md:81` (not re-authored; exact line, 545 chars)

IF appending a zero-init decoder-side density-logit residual refinement (use_refine) IN a solo single-switch child of N0001_champion with use_refine=true, THEN final val MAE falls to 22.99 or lower versus live parent 23.293 at seed 20260830 under the canonical 32ep/1800s protocol, BECAUSE the residual pass over joint pre-softplus logits plus fine and cond context corrects leftover mass misallocation and look-alike peak placement that a single decoder pass leaves behind, DISPROVED IF final val MAE exceeds 22.99 under the canonical protocol.

- **Mechanism summary:** a zero-init post-decoder residual over joint (logits, fine, cond) state performs signed per-cell defect correction (subtract look-alike peaks, fill dropped mass) that the single-pass decoder and the non-negative global GCA path cannot express.
- **Falsifier bar vs the NEW live parent:** the card's literal numbers (≤22.99 vs 23.293) bind to the superseded parent; at booking/verdict time per AGENTS §6 semantics the bar re-instantiates to the live parent **22.5641**, i.e. confirm iff child final val MAE **≤22.26** (a further ≥0.30 gain) — see §5. Hypothesis text itself is never edited.

### Booking 2 of 2 — NEW (causal feedback §3 separating test; 630 chars; generic scope, no H/N ids)

IF exemplar-shuffled simprior control readout (use_shufsim) IN a solo single-switch child of the simprior-bearing parent with exemplar rows permuted across the batch at seed 20260830 under the canonical 32ep/1800s protocol, THEN final val MAE degrades at least 0.30 above live parent 22.5641 to 22.86 or higher, BECAUSE permuting exemplar identity destroys per-cell cosine match content while preserving the additive gradient-path regularization so lost gain implicates true matching signal and retained gain implicates regularization, DISPROVED IF final val MAE at seed 20260830 under the canonical protocol is at or below 22.86.

- **Mechanism summary:** shuffling exemplar rows across the batch keeps the SimPrior architecture, parameter count, and gradient path identical while destroying match content, so the outcome separates the matching-signal story (gain vanishes → confirm) from the gradient-regularization confounder (gain persists → refute).
- **Falsifier bar bound to the NEW live parent 22.5641:** confirm (match-content story) iff child MAE ≥22.86 (+0.30 degradation); refute iff ≤22.86. For reference, an improvement-direction bar off the same live parent would be ≤22.26 (a further 0.30 gain) — stated explicitly per the booking contract.

## 3. Calibration (pasted full output of `python3 scripts/discovery.py calibration`)

```
=== Hypothesis Prediction Calibration (eta=0.20) ===
conf@test        N confirm    rate  pred_conf
[0.25,0.50)      0       0       -          -
[0.50,0.75)      0       0       -          -
[0.75,1.00)      0       0       -          -
<0.25            0       0       -          -

=== Current Standings ===
hyp       confclass       tests
H0001    0.500uncertain       0
H0002    0.500uncertain       0
```

Reading: no tested hypotheses yet (both ledger entries at init confidence 0.50, 0 tests) — the bins are empty because the H0001 evidence event in §4 has not been recorded yet (the Lead records it; this subagent does not). `prove H0001` confirms: `conf=0.5000 (uncertain)`, `posterior ~ Beta(1.00, 1.00)`.

## 4. Recommended evidence event (for the Lead to record via CLI — NOT recorded by this subagent)

Strength justification: w=0.85 — near-maximal because the quantitative clearance is decisive (0.729 margin = 2.43× the 0.30 bar, bar cleared at ep18 and held 15/15 remaining epochs, clean 32/32 pass with zero anomalies) AND the mechanism is directly supported by trained-weight evidence (trust-projection norm 0.5874 off a zero init, both evidence channels used equally, learned temperature sharpening 0.07→0.0393); docked from 1.0 for single-run scope and the one named residual confounder (gradient-path regularization, now covered by Booking 2).

```
python3 scripts/discovery.py evidence H0001 --type supports --strength 0.85 --node N0002_h0001 --note "N0002_h0001 child 22.5641 @ep30 vs live parent N0001 23.293 @ep26, margin 0.729 (2.43x the 0.30 bar; bar 22.99 cleared ep18, held to ep32); clean 32/32 canonical pass, budget_hit false, no anomalies; mechanism-supported: simprior out-proj norm 0.5874 off zero-init with both evidence channels used equally, temp 0.07->0.0393; one residual confounder named (gradient-path regularization) with separating test booked."
```

## 5. Live-parent reference change (explicit)

**N0002_h0001 (22.5641 @ep30) is the new live-parent reference — all future falsifier bars bind to it, not 23.293.** Concretely: any further-improvement hypothesis confirms iff child final val MAE **≤22.26** (≥0.30 below 22.5641) under the canonical protocol at seed 20260830. **Flag:** the adopted `h_refine.md` card's literal falsifier line (≤22.99 vs old parent 23.293) will need its numeric line re-instantiated to ≤22.26 vs 22.5641 at booking/verdict time per AGENTS §6 semantics (both numbers stated in the evidence note; hypothesis text never edited).

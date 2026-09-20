# Synthesis — N0004_h0003 (H0003 `use_padapt`) — Synthesis subagent, node N0004_h0003

- **Node:** N0004_h0003 · **Parent:** N0002_h0001 · **Hypothesis:** H0003 `use_padapt` · **Switch:** `use_padapt` (child `true`; parent default `false`) · **Composition:** SOLO single-switch.
- **Pre-registered falsifier (`idea.md:4-6`, verbatim):** confirm iff child final val MAE **≤ 22.26** at seed 20260830 under the canonical 32ep/1800s protocol (live parent N0002_h0001 = 22.5641 @ep30; ≥0.30 lower).
- **Artifacts read this session:** local `idea.md`, `model.py`, `feedback/{quant,qual,causal}.md`, `config.toml`; server `N0004_h0003/result.json` + `info.json` (read live over `ssh cac-server`); `memory/hypotheses.jsonl` + `memory/index.json`; `N0002_h0001/synthesis.md`; `N0003_h0002/synthesis.md`; parent `N0002_h0001/result.json`; CLI `prove H0003`, `validate --all`, `calibration`.

## 1. Consolidated verdict: REFUTE H0003

All three feedbacks agree and are deduped below into one authoritative record (child `result.json` + `info.json` on the server; parent `result.json`; per-epoch TB `val/mae`/`val/rmse`/`train/loss_epoch`; `best.pth` tensor probe — no re-collection between files).

| Metric | Parent N0002_h0001 | Child N0004_h0003 | Δ (child−parent) |
|---|---:|---:|---:|
| Best val MAE | **22.5641 @ep30** | **25.5661 @ep13** | **+3.0020 (worse)** |
| Final val MAE (ep32) | 22.5705 | 26.1668 | +3.5963 |
| Pre-registered bar ≤22.26 | — | missed by **+3.3061** | — |
| Best-vs-best margin | — | **+3.0020** = ~10× the 0.30 bar | — |
| Epochs / budget | 32/32, `budget_hit: false` | 32/32, `budget_hit: false` | 0 |
| `elapsed_s` (wall) | 1755.0 | 1773.3 | +18.3 |
| Train loss @ep13 / @ep32 | 5.2261 / 2.6183 | 5.5819 / 5.5889 | +0.3558 / +2.9706 |
| Best-ep val RMSE | 89.0939 | 96.0988 | +7.0049 |
| Params (`best.pth` Σ numel) | 31,349,540 | 31,411,748 | +62,208 |
| `config_sha256` / `model_sha256` | 4d4c9baec459f657 / 611be2a9d70768b5 | 769ddfa05d25dfa9 / 93c221209869fe69 | differ |
| `head.padapt.out.weight` norm | n/a (0 `padapt` keys) | **2.4946** (bias 0.2905) | adapter live |

Child best exact `25.566089630126953` (`result.json:5`), parent best exact `22.564146041870117`. Val is **strictly monotone rising** after the peak: 25.5661 @ep13 → 26.1668 @ep32 (+0.6007 over 19 epochs) while the parent descends to 22.5641 @ep30 and holds 22.57 to ep32. Child beat parent on only ep1 (−0.5039) and ep2 (−0.8339) warm-up noise; worse on 30/32 epochs and every epoch from ep3.

**Mechanism-failure attribution (causal, consolidated + deduped).** Genuine mechanism failure, not an ignored branch and not overfitting:
1. **Adapter was active and harmful.** `head.padapt.out` is zero-init (`model.py:251-252`), so the residual path only exists if that gate moves; at the ep13 best checkpoint ‖`out.weight`‖ = **2.4946** (bias 0.2905) — decisively off zero. The `idea.md:172` "out ≈ 0 ⇒ discarded" escape clause does not apply; the optimizer used the adapter and it lost.
2. **Underfit-and-degrade, not overfit.** Child train loss bottoms at 5.563 @ep12 and is flat (5.56–5.59) to ep32, while the parent continues down to 2.6183; child is worse than the parent on **train** from ep2 onward. A classic overfit (train ↓ / val ↑) is absent. Timing rules out late overfit too: +1.659 of the 3.002 gap is already present at ep13 vs the parent's same-epoch 23.9073; the post-ep13 val drift contributes only +0.60.
3. **Dominant cause.** The unmodulated softmax cross-attention over all **2304 h2 tokens** (`model.py:256-261`) injects an image-content residual into the Condenser's K/V, which are exactly the **3** prototype vectors that define the class (`model.py:123`). This corrupts the matching geometry — the survey's named F3 failure mode (`survey_mechanisms.md:84-87`; priced in at `idea.md:165,173`). LOCA's own ablations say modulation is load-bearing (no-att 10.24→11.99; plain-sum for the first OPE attention +5% MAE, `idea.md:49,173`); our variant lacks any foreground/objectness normalization, so it hallucinates background into the prototype. The optimization-level reading (moving K/V target, worse basin) and the content-level reading (background hallucination) are not separately identified in one run; both are the same measured failure. No LOCA objectness modulation was run.

**Artifact ruled out (deduped).** `diff parent/model.py child/model.py` = the `PrototypeAdapter` class, the `use_padapt` flag (`model.py:162`), the `e_cond` insertion (`model.py:172-175`), and the post-SimPrior construction (`model.py:289-291`); `diff config.toml` = exactly one line, `use_padapt = true` (`config.toml:28`). Flag-off is a no-op by construction (`e_cond = e`, `if` skipped) and parent init draws keep their RNG positions (append-only order, AGENTS §5 rule 13; AdamW skips `None` grads), so the parent run is the flag-off reference. Run integrity: 32/32, `budget_hit false`, 1773.3 s ≤ 1800 s, no NaN/Inf/traceback/`_BudgetStop`; the differing shas plus the live `out` norm prove the intended child ran rather than silently falling back. **Harness bug ruled out.**

**No misattribution:** nothing to remap or discard; XScale/GCA/FineFuser/ExemplarEncoder/Condenser/decoder are byte-identical and the confirmed H0001 SimPrior path is untouched (reads original `e`).

## 2. Quality gate (AGENTS step 8; K_SYNTH=2) — 0 bookings

**Bookings: 0.** Nothing genuinely new survives the gates; this is stated explicitly rather than padded, per the causal recommendation and AGENTS §11.

**(a) Format / novelty gate for the single considered candidate — outputs pasted verbatim.** The only survivor of the three feedbacks is the causal agent's objectness-modulated adaptation variant (mask/weight the adaptation keys by a frozen-similarity foreground estimate). Its draft re-encode line (669 chars) was run through both gates:

`validate()` (`PYTHONPATH=src`, `cac.expt.hypothesis.validate`):
```
LEN 669
GATE ([], [])
```
(exit 0; no errors, no warnings; markers IF→IN→THEN→BECAUSE→DISPROVED in order)

`python3 scripts/novelty_check.py "<draft>"`:
```
top similarities: H0003=0.605, H0004=0.436, H0001=0.318
NOVEL (top sim 0.605 < 0.82)
```
(exit 0; no structural twin flagged)

**(b) Why this candidate dies anyway (AGENTS §11).** It is a re-encode of the refuted H0003 mechanism family (same solo-switch pre-Condenser interface, same pooled-K=3 `e`, same target edge `e_cond → Condenser`; only the attention weighting changes). §11 requires such a re-encode to carry a **genuinely NEW falsifier**. Its only discriminating, lab-valid test is the *identical* registered criterion — final val MAE ≤ 22.26 vs the live parent 22.5641 at seed 20260830 under the canonical protocol — i.e. the same intervention-independent prediction H0003 already made and failed. Changing the attention mask changes the intervention but not the testable prediction, so there is no new falsifier and the candidate **dies at the §11 gate**. (Novelty alone is not sufficient: the pass above is TF-IDF/structural, not the new-falsifier requirement.)

**(c) Spend recommendation (causal, adopted).** The pre-Condenser prototype-adaptation family has now failed once with a measured-active adapter that worsened both train and val. Per `feedback/causal.md:211-218` and `survey_mechanisms.md:10-17,174-175`, the portfolio order is F1 (dense similarity prior, already H0004) → F4 → F6 (negative-prototype suppression, `local/ideas/negsup.md`), ahead of any retry of this family. No node should be spent on the adaptation family now unless a modulated variant is explicitly wanted; and per §11 that variant cannot be booked here without a new falsifier.

**(d) Budget / duplicates.** 0 bookings ≤ K_SYNTH=2. Because nothing is booked, there are no duplicates to audit; for completeness, the rejected candidate's nearest ledger sims (H0003 0.605, H0004 0.436, H0001 0.318) are well below 0.82 and it does not twin the pending unbooked candidates — `use_refine` / `use_shufsim` (`N0002_h0001/synthesis.md`), `use_simbank` (now booked H0004) and the oracle per-image shift ceiling (`N0003_h0002/synthesis.md`), nor the `use_negsup` F6 card (`local/ideas/negsup.md`).

**(e) Remap-or-discard:** nothing booked and nothing misattributed; no remap required. The qual and causal readings were already deduped in §1.

## 3. Live-parent reference — unchanged

**N0002_h0001 = 22.5641 @ep30 REMAINS the live parent. H0003's refutation does not affect it.** N0004_h0003's 25.5661 is worse than the live parent and clears no bar; per quality × availability selection the live reference stays 22.5641, and all future falsifier bars still bind to it (next improvement bar = **≤22.26**).

## 4. Calibration (full output of `python3 scripts/discovery.py calibration`, run this session)

```
=== Hypothesis Prediction Calibration (eta=0.20) ===
conf@test        N confirm    rate  pred_conf
[0.25,0.50)      0       0       -          -
[0.50,0.75)      2       1     50%       0.50
[0.75,1.00)      0       0       -          -
<0.25            0       0       -          -
overall          2       1     50%      -
reliability error (weighted |rate − pred_conf|): 0.000

=== Current Standings ===
hyp       confclass       tests
H0001    0.585uncertain       1
H0002    0.420uncertain       1
H0003    0.500uncertain       0
H0004    0.500uncertain       0
```

Reading: two scored tests (H0001 supports → confirm; H0002 contradicts → non-confirm), both scored at their pre-event confidence 0.50, giving 50% observed against pred_conf 0.50 → reliability error 0.000 (no drift warning). H0003 stands at init 0.50 with 0 tests because the evidence event in §5 is **not yet recorded**; H0004 (simbank) is likewise untested at 0.50. `prove H0003` confirms `conf=0.5000 (uncertain)`.

## 5. Recommended evidence event (for the Lead to record via CLI — NOT recorded by this subagent)

**Strength `w=0.95`.** Near-maximal: the paired refutation is decisive on the pre-registered reading (child best 25.5661 is **+3.0020 worse** than the live parent 22.5641 — ~10× the 0.30 bar — and misses the 22.26 bar by **+3.3061**), the mechanism is directly measured active-but-harmful (`head.padapt.out.weight` norm **2.4946** off a zero-init gate, bias 0.2905), the run is a clean 32/32 canonical pass with `budget_hit false` and the predicted +62,208 param delta, and the only competing explanation (harness/artifact) is ruled out by the one-switch config diff, the flag-off no-op, the differing shas, and the live adapter norm. Docked from 1.0 only for single-run scope and the fact that (b) vs (c) in the causal reading are not separately identified.

```
python3 scripts/discovery.py evidence H0003 --type contradicts --strength 0.95 --node N0004_h0003 --note "N0004_h0003 child best 25.5661 @ep13 vs live parent N0002_h0001 22.5641 @ep30, +3.0020 worse (~10x the 0.30 bar); pre-registered bar 22.26 missed by +3.3061; final ep32 26.1668 vs parent 22.5705 (+3.5963), val monotone rising +0.6007 ep13->32; adapter active and harmful (head.padapt.out.weight norm 2.4946 off zero-init, bias 0.2905) -> mechanism failure not discarded; underfit-and-degrade: train loss plateaus 5.56 vs parent 2.6183, child worse from ep3; clean 32/32 canonical pass, budget_hit false, param delta +62,208 as predicted; single-switch config diff, flag-off reproduces parent, artifact ruled out."
```

**Expected confidence after this one event (Eq.1, η=0.20, c=0.50; advisory only):** `c' = c − η·w·c = 0.50 − 0.20·0.95·0.50 = **0.4050**` — still `uncertain` (above the <0.25 refuted threshold), so H0003 is not yet classified refuted; a further independent contradiction event would be needed to cross it. The Lead records the event; the ledger derives the stored confidence. **This synthesis does not record it and does not edit any ledger or confidence value.**

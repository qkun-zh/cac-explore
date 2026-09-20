# Synthesis — N0003_h0002 (H0002 `use_gca_cal`) — Synthesis subagent, node N0003_h0002

- **Node:** N0003_h0002 · **Parent:** N0001_champion · **Hypothesis:** H0002 · **Switch:** `use_gca_cal` (child `true`; parent default `false`) · **Composition:** SOLO single-switch.
- **Pre-registered falsifier (`idea.md:4,110,115`):** confirm iff child final EMA val MAE ≤ **22.99** at seed 20260830 under the canonical 32ep/1800s protocol (live parent 23.293 @ep26; ≥0.30 lower).
- **Lead-verified context (do not re-litigate):** H0002 REFUTED by its pre-registered falsifier. Clean 32/32 pass, `budget_hit false`. Live best is the sibling **N0002_h0001 = 22.5641 @ep30**.

## 1. Consolidated verdict: REFUTE H0002

All three feedbacks agree; numbers below are the single authoritative record, deduped from `feedback/{quant,qual,causal}.md` and traced to the two `result.json` files, the TB `val/mae` series, and `best.pth` inspection.

| Metric | Parent N0001_champion | Child N0003_h0002 | Source |
|---|---|---|---|
| Best val MAE | 23.293153762817383 @ep26 | **23.20878791809082 @ep32** | both `result.json:5-6` |
| Best-vs-best margin | — | **0.08436584** = 28.1% of the 0.30 bar | result.json |
| Final-epoch (ep32) val MAE | 23.33643341064453 | 23.20878791809082 | TB `val/mae` |
| Matched-epoch (ep32) margin | — | 0.12764549 | TB `val/mae` |
| Registered bar ≤22.99 — miss | — | **+0.218787** (best never within 0.21 of bar) | result.json |
| Epochs / budget | 32/32, `budget_hit false`, 1694.1 s | 32/32, `budget_hit false`, 1698.95 s | result.json |
| `config_sha256` / `model_sha256` | 679f37c26bc72103 / eaf2a8b93c280cba | daf2b13a125900ad / 633e362a482b3420 | result.json |
| Status / tested | baseline | `done`, `["H0002"]`, `code: ok` | `info.json` |

**Live cross-reference (context, not the direct parent):** new live best **N0002_h0001 = 22.564146041870117 @ep30**; child is **0.6446 worse**. Re-instantiated 0.30 bar off 22.5641 (AGENTS §6) = **22.2641**, missed by **+0.9447**. The node's own booked bar stays 22.99.

### 1a. Measured parameters — the "moved-but-missed" branch, NOT the null

Stored EMA weights that produced the val number (`best.pth`, via `best_checkpoint.py:21-29` / `calls/ema.py:46-63`), read by the causal probe:

| param | init | learned (EMA) | null box (`idea.md:123`) | null? |
|---|---|---|---|---|
| `gca.s_pos` | 0.02 | **0.00839800** (−0.011602, −58.0%) | `\|s−0.02\|/0.02 < 0.10` → measured **0.5801** | **FAIL (moved)** |
| `gca.s_lin` | 0.0 | **0.02147696** | `\|s_lin\| < 1e-3` | **FAIL (moved)** |
| `gca.b_cal` | 0.0 | **−0.11962026** | `\|b_cal\| < 0.5` | pass |

Trunk norms (`gca.gca.2` = last GCA layer): child weight 0.151678 / bias 0.072668 vs parent weight **0.233494** / bias 0.085947.

**Why "moved-but-missed" ≠ null, and still REFUTE.** Two of three null sub-conditions fail: the optimizer actively used the signed degree of freedom (halved the softplus branch, added a nonzero linear term, negative offset). This is not "coupling was already optimal." But the only quantity in count units, `b_cal = −0.12` counts, is **~0.03%** of the idea's own arithmetic that the 0.30 bar = **386 counts** (`idea.md:46`; Σ|Δ| ≈ 29,955 over 1286 val images). The tail margin is a consistent **real but small** paired effect — 10/10 epochs ep23–32 same direction, mean −0.0937, magnitudes 0.0646–0.1276, all above the ~0.02 paired precision and both plateau spreads (child ep24–32 range 0.0510, ep25–32 range 0.0182; parent ep26–32 range 0.0433) — yet **0.0844 < 0.30**. Under AGENTS rule 1 (no outcome shopping) the registered bar governs: **DISPROVED**. Per `idea.md:125` this is explicitly the moved-but-missed branch, evidence for a follow-up, not a rescue.

**No undertraining rescue within the frozen protocol.** Best is the final epoch but the curve is a flat plateau: child ep25→ep32 = −0.008 over 7 epochs (~0.0011 MAE/epoch); closing 0.2188 to 22.99 at that slope needs ~34–192 more epochs, and cosine LR is at its floor (η_min 1e-6). Protocol is frozen (32ep/1800s, seed 20260830 forever, no `repro_run --set seed=`).

**Anomalies (deduped, none changes the verdict):** (1) TB `train/mae` is `nan` for all 32 epochs in child **and** parent — torchmetrics `MeanMetric.compute`-before-`update` logging artifact; `train/loss_epoch` and `val/mae` are finite (no training NaN). (2) `result.json` carries disagreeing `elapsed` (1698.95) vs `elapsed_s` (1704.2); same in parent. (3) Child ep1 val MAE 164.2032 vs parent 146.9976 — every node cold-starts its 3.5 M head (`runner.py` has no parent-weight warm start), so "init-identical forward" is a parameter-init statement invisible to the epoch-0 metric; the idea's "epoch-0 parent-level" premise is not supported (qual's remap).

## 2. Quality gate (AGENTS step 8; K_SYNTH=2)

**(a) Format gate — outputs pasted verbatim:**

`python3 scripts/discovery.py validate --all`:
```
validated 2 hypotheses — 0 bad (all good)
```
(exit 0)

`validate()` + `novelty_check` on each proposed booking line:
```
=== A len=642 ===
gate: ([], [])
top similarities: H0001=0.349, H0002=0.349
NOVEL (top sim 0.349 < 0.82)
novelty EXIT 0

=== B len=663 ===
gate: ([], [])
top similarities: H0002=0.327, H0001=0.218
NOVEL (top sim 0.327 < 0.82)
novelty EXIT 0
```

**(b) Budget:** exactly 2 bookings = **K_SYNTH=2**, both new and attributable to this node; nothing else booked here.

**(c) No duplicates.** Cross-checked against the live ledger AND `N0002_h0001/synthesis.md` (read; its two pending batch-2 candidates are the `use_refine` decoder-logit residual card and the `use_shufsim` shuffled-exemplar control). TF-IDF similarity of the two proposed lines against ledger + those two sibling texts (combined corpus):
```
A | sib_shuffled=0.402, H0001=0.310, H0002=0.293, sib_refine=0.273   (no structural twin)
B | H0002=0.303, sib_shuffled=0.248, sib_refine=0.187, H0001=0.186   (no structural twin)
```
All ≪ 0.82. Booking A differs from H0001 in bank resolution/representation (per-token 1/8 keys vs pooled 1/16 `e`), from the sibling refine card (exemplar-similarity readout vs post-decoder logit residual), and from the sibling shuffled control (a new bank vs a destructive permutation of the confirmed readout). Booking B is a no-training diagnostic and matches none of them. Neither re-encodes H0002's signed global coupling, so no `contradicts`-on-existing-id framing is required.

**(d) Remap-or-discard:** one remap recorded, verdict unchanged. `idea.md:31,33,46` reasoned as if the 0.02 path were "near-detached" with gradient only through the fixed 0.02 scale; the measured parent `gca.gca.2` weight norm **0.233 ≠ 0** shows the parent's `z` was already learned and image-dependent — only the scale `0.02` was fixed. The "no gradient / no learned signal" framing is remapped to "fixed scale on a learned logit"; the verdict is unaffected. Qual's epoch-0 premise is likewise remapped (explicit). No feedback claim needed discarding.

### Booking 1 of 2 — NEW (direction i; 642 chars; generic scope, no H/N ids)

```
IF augmenting the live simprior readout with a finer 1/8 exemplar-token similarity bank via use_simbank IN a single-switch extension of the simprior-bearing parent node trained under the canonical protocol at seed 20260830 for 32ep/1800s, THEN final EMA val MAE falls at least 0.30 below the live parent 22.5641 to 22.26 or lower, BECAUSE per-token 1/8 ROI keys keep part-level exemplar geometry that 1/16 softmax-pooling averages into a background-dominated vector for small objects, so a max-over-token match recovers small-instance evidence the pooled prior drops, DISPROVED IF final EMA val MAE exceeds 22.26 under the canonical protocol.
```

- **Mechanism (one line):** the confirmed similarity prior reads `e` — a single 256-d vector per exemplar softmax-pooled from h3@1/16, which for small FSC147 objects is mostly padding/background; a 1/8 (h2) ROI token bank keeps K·49 per-part keys and a max-over-token similarity recovers the small-instance match evidence the pooled vector averages away.
- **Bar binding:** the literal ≤22.26 is bound to the **live parent 22.5641 at verdict time** per AGENTS §6 semantics; if the live parent number changes before this node runs, re-instantiate the numeric bar from the then-live parent and state both numbers in the evidence note. The hypothesis text is never edited.

### Booking 2 of 2 — NEW (direction ii, the causal agent's Diagnostic D; 663 chars; generic scope, no H/N ids)

```
IF an offline per-image signed count-error diagnostic on the canonically trained live-parent val pass IN a no-training val-set dump of per-image predictions and ground-truth counts, THEN the oracle per-image uniform additive count correction lowers val MAE by less than 0.30 below the live parent 22.5641, BECAUSE count error equals the scored quantity so a single per-image signed constant is the strongest possible member of the global-calibration family and its oracle ceiling decides whether that family can ever clear the 0.30 bar, DISPROVED IF the oracle per-image constant shift lowers val MAE by at least 0.30 to 22.26 or lower under the same predictions.
```

- **Mechanism (one line):** with `e_i = pred_i − gt_i` and MAE = mean`|e_i|`, the oracle `Σ_i min_δ |pred_i + δ − gt_i|` upper-bounds what *any* global signed-count coupling can achieve, so this diagnostic certifies the ceiling of the family that H0002 failed before another training run is spent.
- **Bar binding:** the literal ≤22.26 (and the 0.30 reduction) is bound to the **live parent 22.5641 at verdict time**; re-instantiate from the then-live parent and state both numbers if it changes. No training, no protocol change, seed irrelevant to the dump.
- **Gate consequence:** if the oracle reduces MAE by <0.30 → retire the global-calibration family (contradicts-evidence on H0002, AGENTS §11); if ≥0.30 → exploitable structure exists and a fresh per-image/exemplar-conditioned signed-calibration card is justified, with its own pre-registered falsifier. H0002 stays refuted either way.

## 3. Live-parent reference change (explicit)

**N0003_h0002 (23.2088 @ep32) does NOT become the live parent.** Its result is worse than the sibling and does not clear its bar; per quality × availability selection the live reference remains **N0002_h0001 = 22.5641 @ep30**. All future falsifier bars bind to **22.5641** (next improvement bar = **≤22.26**), not 23.293 and not 23.2088.

## 4. Calibration (full output of `python3 scripts/discovery.py calibration`, run this session)

```
=== Hypothesis Prediction Calibration (eta=0.20) ===
conf@test        N confirm    rate  pred_conf
[0.25,0.50)      0       0       -          -
[0.50,0.75)      1       1    100%       0.50
[0.75,1.00)      0       0       -          -
<0.25            0       0       -          -
overall          1       1    100%      -
reliability error (weighted |rate − pred_conf|): 0.500
WARNING: reliability error > 0.20 — confidence at test is drifting

=== Current Standings ===
hyp       confclass       tests
H0001    0.585uncertain       1
H0002    0.500uncertain       0
```

Reading: the one bin entry is H0001's recorded support event, scored at its pre-event confidence 0.50 → 100% observed confirm rate but pred_conf 0.50, hence the drift warning; the H0002 evidence recommended in §5 is **not yet recorded**, so H0002 still stands at init 0.50 with 0 tests.

## 5. Recommended evidence event (for the Lead to record via CLI — NOT recorded by this subagent)

**Strength justification:** `w=0.85` — near-maximal because the refutation is decisive on **both** registered readings (absolute bar missed by 0.2188; paired margin 0.0844 = 28% of the 0.30 bar), the run is a clean 32/32 canonical pass, and the mechanism is directly measured (null box fails 2/3); docked from 1.0 for single-run scope and because the effect is a small real gain rather than a catastrophic failure.

```
python3 scripts/discovery.py evidence H0002 --type contradicts --strength 0.85 --node N0003_h0002 --note "N0003_h0002 child 23.2088 @ep32 vs live parent N0001 23.293 @ep26, margin 0.0844 vs 0.30 bar (28.1%); absolute bar 22.99 missed by 0.2188; re-instantiated bar off live 22.5641 = 22.2641 missed by 0.9447; params moved off null: s_pos 0.0084 (-58%), s_lin 0.0215, b_cal -0.1196 counts (~0.03% of the 386-count bar); moved-but-missed, clean 32/32 canonical pass, budget_hit false."
```

**Expected confidence after this one event (Eq.1, η=0.20, c=0.50):** `c' = c − η·w·c = 0.50 − 0.20·0.85·0.50 = **0.4150**` (still `uncertain`; above the <0.25 refuted threshold, so H0002 is not yet classified refuted — a further independent contradiction, e.g. the §2 Booking 2 diagnostic, is required to cross it). These are advisory derivations only; the Lead records the event and the ledger derives the stored confidence.

**Evidence base (read this session):** `feedback/{quant,qual,causal}.md`; `result.json` (child) + `tree/N0001_champion/result.json` (parent) + `tree/N0001_champion/N0002_h0001/result.json` (live best); `idea.md`; `memory/index.json` + `memory/hypotheses.jsonl`; `N0002_h0001/synthesis.md`; `local/research/{survey_mechanisms,first_principles}.md`; CLI `prove H0002`, `validate --all`, `calibration`.

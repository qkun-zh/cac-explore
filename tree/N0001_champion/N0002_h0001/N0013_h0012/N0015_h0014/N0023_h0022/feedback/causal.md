# Causal feedback — N0023_h0022 (H0022 `use_margin`, MarginCal)

Causal agent, 2026-09-22. Inputs read: AGENTS.md §3/§5/§6, STATE.md, NODE
`model.py`/`idea.md`/`config.toml`/`result.json`/`val_result.json`/`val_perimage.json`,
parent N0015 `model.py`, parent dump `/home/qkun/cac_backup/N0015_val_perimage.json`
(+ `N0015_val_result.json`), `local/research/perimage_diagnostic.md` appendix,
NODE tensorboard `run/latest/tb`. No commit, no ssh. Every number below was
recomputed from those files this session.

Labels: **CONFIRMED** = directly file-backed (code line or recomputed dump
number) · **PLAUSIBLE** = mechanism inference consistent with the files but not
ablated · **SPECULATIVE** = consistent story, no direct file support.

---

## 0. Verified outcome and accounting (CONFIRMED)

- `result.json`: `futility_hit=true`, `n_epochs_done=24` of 32, best
  **22.1632 @ep24**, bar 19.0431 (missed by +3.12). `val_result.json`:
  n=1286, MAE 22.163878. Parent live = **19.3431** (STATE); parent backup dump
  reproduces 19.3259 — the same EMA/dump nuance the appendix already documented
  (19.326 vs 19.3431).
- Dumps verified: both `val_perimage.json` have **n=1286, identical `ids` and
  identical `gt`** → paired contrast is exact. Total AE: parent **24853.1**
  (dump) / 24875.2 (at reported MAE), child **28502.7** → **ΔAE = +3649.7**
  (dump-pair) or **+3627.5** (reported-pair ≈ the handoff's ≈3613; ΔMAE
  +2.8208). Slice partition recomputed as sparse<50 / mid 50–500 / dense>500 =
  **895 / 374 / 17**, exactly matching `val_result.json`.
- Child curve (local tb, file-backed): ep1 146.1 → ep14 23.48 → ep23 22.27 →
  ep24 **22.1632** (monotone to halt). The crossover claim (child ahead through
  ep14, behind +0.94 by ep23) is a **lead-provided verified datum** — parent tb
  is not local, not independently re-derived here (no ssh allowed).
- Same-seed noise datum (appendix, CONFIRMED as a recorded file fact): exact
  parent retrain +2.25, 3-draw spread {19.34, 20.70, 21.59}. Magnitude of the
  +2.82 is therefore draw-contaminated; the *structural* links below are
  draw-independent and are what the mechanism verdict rests on.

## ROOT — the margin branch is call-site exclusive; cellcal/peakcal never ran (CONFIRMED)

- `config.toml:28-30`: `use_cellcal=true`, `use_peakcal=true`,
  `use_margin=true` — all three on.
- Child `model.py:177-178`: `if self.use_margin:` is the **first** branch and
  calls `self.simprior(fine, e, use_margin=True, w_m=self.margincal.w_m)` —
  it never passes `use_cellcal`/`use_peakcal`/`w_c`/`w_p`.
- Child `model.py:237-244` (`SimPrior.forward`): the `if use_margin:` block
  **returns at line 244**, before `if use_cellcal:` (:245) and
  `if use_peakcal:` (:251). `w_c` is never even passed → stays zero-init,
  receives no gradient, never ran. The cellcal gain **did not execute on any
  image, wipe or healthy**.
- Parent N0015 `model.py:176-185, 234-251`: runs cellcal
  (`gain=exp(w_c·log1p(mhat))`, sign-blind, multiplies **both channels, both
  signs**, trained w_c≈0.148 per STATE) **and** peakcal
  (`gain=exp(clamp(w_p·f,−3,3)).clamp(0.7,1)`); peakcal was dead in the parent
  (w_p≈0.0017 → gain≈1, STATE H0014 verdict) so **peakcal removal is a null
  delta**; the entire actuator change is:

  | | parent N0015 | child N0023 |
  |---|---|---|
  | gain formula | `exp(w_c·log1p(mhat))`, w_c≈0.148 (≈×1.108 at mhat=1) | `exp(clamp(w_m·log1p(mhat),−3,3))`, w_m=**+0.2666** (≈×1.203 at mhat=1) |
  | applied to | ALL cells, both ev channels, sign-blind | only `ev>0` cells (elementwise per channel) |
  | `ev≤0` cells | multiplied by gain>1 (deepened) | **identity** (`torch.where` else-branch) |
  | peakcal | on, dead (≈identity) | never called (≈identity) |

  This deletion is not wipe-scoped: it is a **global forward-diff vs the
  parent for every image**.

---

## (a) Why the wipes could not be rescued by positive-only gain (CONFIRMED)

1. **The mask literally never touches anti-phase cells.**
   `model.py:242-243`:
   `pos = (ev > 0).float().detach()` then
   `ev = torch.where(pos > 0.5, ev * gain, ev)` — elementwise per channel.
   Any cell with `ev≤0` takes the identity branch: not amplified, not
   suppressed, not sign-flipped. There is no code path by which the margin
   gain can add mass where the evidence is negative.
2. **The wipes live almost entirely in that untouched branch.** Diagnostic
   appendix (CONFIRMED dump): wipe group top1_mean **−0.258**, neg_ev_frac
   **0.667**, ev_sum **−4924** — two-thirds of cells and the whole net
   evidence mass are negative and pass through identity unchanged. The
   appendix already bounded the relief available from merely *stopping
   cellcal's deepening* at **~377 ev-units** (−4924→−5301), an order of
   magnitude short of the counts the wipes lack (935 alone lacks 1782 under
   the child).
3. **The gain that does run is bounded and targets the wrong minority.**
   At w_m=+0.2666, mhat≈1: `gain = exp(0.2666·ln2) ≈ 1.20` — a ~20% boost on
   the ~1/3 positive cells, which on anti-phase images are by construction the
   minority (largely spurious) matches. Amplifying them cannot restructure
   the −4924 evidence field.
4. **Outcome (CONFIRMED, recomputed):** **0/11 rescued** (child ratio ≥0.5),
   9/11 worse. 935 0.40→**0.15** (+532 AE), 3428 0.23→0.09 (+62), 3488
   0.19→0.09 (+43), 3487 0.31→0.22 (+28); 840/851/865/7656/3481/3482/3484
   flat-to-slightly-changed (−18…+8). Wipe-group ΔAE **+678.3 (18.6% of the
   total delta)**. R2 falsifier dead; the card's own pre-registered F3
   ("wipe-not-gain-caused → exemplar-side successor") is the surviving
   explanation: the anti-phase field is set upstream at the exemplar-encoding
   interface, and no `ev`-level sign mask can reach it.

*Why 9/11 got actively worse despite removing the −377 deepening:*
**PLAUSIBLE** — the parent's equilibrium (sign-blind gain co-trained with
q/k/out for 32 epochs) sat at mean ratio 0.241; the child retrained from
scratch under a different gradient field (link b) and halted at ep24, landing
below that equilibrium. The single dominant term is 935 (+532 of +678), the
corpus's noisiest image (gt 2092) — its magnitude specifically is
**SPECULATIVE** to attribute to mechanism rather than draw (same-seed spread
datum).

## (b) Why dense got WORSE than parent (+127.88) (numbers CONFIRMED, mechanism PLAUSIBLE)

**Accounting (CONFIRMED, recomputed):** dense>500: 330.81 → 458.69 =
**+127.88 MAE = +2173.9 AE = 59.6% of the +3649.7 total**, signed undercount
deepening −5624 → −7641 (−2017 counts). Split:

- dense **non-wipe** (13): **+1647.9 AE (45.2%)**, 10/13 worse, signed −1491.
  Dominated by **new collapses**: 3425 ratio **0.90→0.13** (+683), 949
  0.86→0.64 (+241), 3437 0.86→0.48 (+191), 3433 0.66→0.38 (+150), plus
  957/830/1936/5860 small.
- dense wipe-members (840/865/935/7656): +526.1 (14.4%), almost all 935.
- Same collapse mode spills into mid: 3436 **1.13→0.09** (+346), 3434
  0.63→0.37 (+77). Across gt≥250, **8 images flipped parent-ratio≥0.5 →
  child<0.5, summing +1547 AE ≈ 42% of the entire +3650**.

**Why the exclusive branch did this to ALL images:**

1. The child deleted the parent's sign-blind cellcal gain from every forward
   (ROOT, CONFIRMED). On dense-KEEP images that gain was a **positive-mass
   floor**: appendix KEEP ev_sum +2455 → evg_sum +3022 (**+566 net mass**,
   CONFIRMED).
2. The same gain also acted as an **implicit dense-scene loss weight**: mhat
   is detached but `ev` is not, so parent gradients into S→qproj/kproj/out
   were scaled by `gain>1` on *all* high-mhat cells — including the negative
   half of dense/anti-phase grids where the child now routes gradient at
   scale 1 (`model.py:243`). The child trains its dense-cell evidence with
   roughly a `1/g ≈ 0.9×` relative discount vs the parent. (Code-path fact:
   CONFIRMED; consequence for learned dense structure: PLAUSIBLE.)
3. The mask **disengages exactly where density is highest**: the appendix
   records that evidence sign *flips with density* — as a scene drifts toward
   anti-phase, `ev>0` coverage collapses and the child's only actuator goes
   idle on precisely the images that need mass (this is how 3425/3436, fine
   under the parent, collapsed under the child).
4. **Static math cannot explain the sign of the outcome — and that is the
   point.** Holding base ev fixed:
   `child_ev − parent_ev = pos·(g_m−g_c) + neg·(1−g_c)` with g_m>g_c>1 and
   neg<0 → **both terms positive**: the child's pre-out evidence is
   pointwise *larger* (more mass-adding) than the parent's. Observed is the
   opposite (mass −1491 on dense-keep). Therefore the dense regression is not
   the mask arithmetic itself; it runs through (i) the deleted mass floor +
   deleted dense gradient-focusing, (ii) co-adaptation of q/k/out under the
   new gradient field over 24 epochs, (iii) futility truncation at 24/32, and
   (iv) draw noise — **PLAUSIBLE** as a bundle; the collapse identities and
   magnitudes are **CONFIRMED** from the dumps.
5. Peakcal's removal contributed nothing (dead in parent, CONFIRMED).

## (c) Why the healthy majority lost late-phase suppression (numbers CONFIRMED, reading PLAUSIBLE)

- Parent mechanism reading (**PLAUSIBLE**): sign-blind `ev·g_c` deepened
  negative background evidence → deeper negative cond residual → decoder
  co-adapts suppression; this matured late as w_c grew (0.148) alongside 32
  epochs. Child: negatives at identity (`model.py:243`) — that suppression
  can only be re-learned from unscaled ev, at ~0.9× gradient on those cells
  (link b.2), within 24 epochs.
- What the broad groups actually show (CONFIRMED, recomputed):
  - sparse<50 (n=895): MAE 5.814→**5.703 (−0.11)**, ΔAE −99.3, signed
    +138 (slightly *more* mass), 470 better/425 worse — i.e. sparse
    *improved*, consistent with also having lost cellcal's texture-FP
    amplification (the STATE H0012 sparse caveat).
  - mid 50–250 non-wipe (n=337): MAE 27.08→28.51 (**+1.43**), ΔAE +481.7
    (13.2%), signed **+228** (slightly more mass), 167 better/170 worse —
    a diffuse, near-parity slip whose direction (more counts) matches
    "lost suppression".
- So (c) is real but **secondary**: the healthy majority contributes ~10% of
  the +2.82 net (sparse −99, mid50-250 +482), and its signed drift is toward
  *less* undercount, exactly as suppression-loss predicts. It did **not**
  produce the verdict. The headline damage is link (b): undercount
  *collapses* on the 250+ band — a different phenomenon from the diffuse
  suppression loss, and it must not be conflated with it.

## (d) Where the mid-train advantage came from (PLAUSIBLE / partial CONFIRMED)

- CONFIRMED: child's own tb curve descends smoothly ep1→24 (ep14 23.48,
  ep23 22.27, ep24 22.163); `result.json` futility HALT at 24/32 while the
  curve was still descending (just too slow vs the 19.0431 bar slope). The
  "ahead through ep14, behind +0.94 by ep23" crossover is a lead-provided
  datum (parent tb not local).
- PLAUSIBLE mechanism for the early lead: at start, `out` is zero-init and
  q/k evidence is weak/unstructured (`SimPrior` docstring, `model.py:224-227`).
  The margin path amplifies only the mass-adding (positive) component —
  useful in a globally undercounted regime — while routing the not-yet-trained
  negative evidence through identity, i.e. it **avoids the parent's early
  amplification of structured-noise negatives** (parent multiplies whatever
  the random-ish q/k produce, both signs). Cleaner early residual → faster
  early descent; w_m engaged positively (R1 datum: +0.2666).
- After crossover (~ep15-16, lead datum), the parent's w_c grew into its
  sign-blind equilibrium and its late-phase suppression/mass-floor matured
  over epochs 16–32; the child never entered that phase (HALT ep24). The
  mid-epoch advantage was thus a *transient of the early gradient field*,
  not a better fixed point — the same actuator deletion that helped early
  (identity-on-weak-negatives) is what failed late (identity-on-dense-
  anti-phase + no dense gradient focusing).
- SPECULATIVE: the specific claim "early gain ≈ useful because evidence is
  weak" was not ablated (no no-gain child arm); it is the natural reading of
  the sign-mask + zero-init facts, not a measured decomposition.

## (e) Quantified attribution of +2.82 (all CONFIRMED, recomputed from the two dumps)

ΔAE total **+3649.7** (dump-pair; +3627.5 reported-pair) on **n=1286**,
ΔMAE +2.838 / +2.821:

| group | n | parent MAE | child MAE | ΔMAE | ΔAE | share of ΔAE |
|---|---:|---:|---:|---:|---:|---:|
| **dense >500** | 17 | 330.81 | 458.69 | **+127.88** | **+2173.9** | **59.6%** |
| — dense non-wipe | 13 | 177.72 | 304.48 | +126.76 | +1647.9 | 45.2% |
| — dense wipe members (840/865/935/7656) | 4 | — | — | +131.5 | +526.1 | 14.4% |
| **mid 50–500** | 374 | 37.50 | 41.71 | **+4.21** | **+1575.1** | **43.2%** |
| — mid wipe members (851/3428/3481/3482/3484/3487/3488) | 7 | — | — | +21.8 | +152.3 | 4.2% |
| — **mid 250–500 non-wipe** | 30 | 92.50 | 123.87 | **+31.37** | **+941.2** | **25.8%** |
| — mid 50–250 non-wipe | 337 | 27.08 | 28.51 | +1.43 | +481.7 | 13.2% |
| **sparse <50** | 895 | 5.814 | 5.703 | **−0.111** | **−99.3** | **−2.7%** |
| *memo:* the 11 wipes (cross-slice) | 11 | 494.3 | 556.0 | +61.7 | +678.3 | 18.6% |
| *memo:* 8 new collapses gt≥250 (parent≥0.5→child<0.5) | 8 | — | — | — | +1547 | 42.4% |

Reading: **~60% dense (≈45pp of it new dense-keep collapses + 14pp wipe
deepening), ≈26pp the mid 250–500 non-wipe band (itself concentrated: top-3
images = 60% of that band, led by 3436's fresh collapse), ≈13pp diffuse mid
50–250, sparse slightly better.** The +2.82 is a **high-count collapse event**
(new + deepened ratio<0.5 images, gt≥250), not a broad degradation and not
the wipes' failure to rescue per se (wipes alone = 18.6%).

---

## The single most load-bearing causal statement

MarginCal's `pos=(ev>0)` mask makes the actuator structurally incapable of
rescuing the wipes — it never touches the net-negative anti-phase field
(ev_sum −4924, 67% of cells) that defines them, so 0/11 rescue was decided
at code-write time — while the call-site-exclusive branch simultaneously
deleted cellcal's sign-blind gain from every image, removing the mass floor
and dense-cell gradient focusing that held the 250+ band, where new collapses
(3425 0.90→0.13, 3436 1.13→0.09) supply the majority of the +3627 AE.

*Noise caveat carried, not hidden: the +2.82 magnitude overlaps the known
+2.25 same-seed draw spread (appendix), but the two structural findings above
(mask-incapability, exclusive deletion) are draw-independent code facts.*

# Qualitative feedback — N0023_h0022 (`use_margin` MarginCal, child of N0015_h0014)

Text-log analysis per AGENTS §9. Sources read directly (all local, no ssh):
node `idea.md` / `model.py` / `config.toml` / `result.json` / `val_result.json` /
`val_perimage.json`, parent per-image dump `/home/qkun/cac_backup/N0015_val_perimage.json`
(+ `N0015_val_result.json`), child TensorBoard scalars
(`run/latest/tb/t/events.out.tfevents.1790049213…`), `journal/events.jsonl`
(entries 28/58/59), `memory/hypotheses.jsonl` (create/evidence for H0022),
`local/research/perimage_diagnostic.md` (Appendix wipe table), and the booked
hypothesis text in `memory/hypotheses.jsonl` / `idea.md:9`.
Mechanism scalars (w_m, w_c, temp) and the parent-side trajectory points are
Lead-verified and cited to `journal/events.jsonl:59` (no local checkpoint for
N0023 exists; ssh forbidden this session).

---

## 1. Did the BECAUSE clause hold as written?

**No — the premise half held (and was already measured pre-run), the remedy half
failed decisively.**

The booked BECAUSE (`idea.md:9`, `memory/hypotheses.jsonl` create line) has two
joints:

1. **Diagnosis:** cellcal's sign-blind gain `exp(w_c·log1p(mhat))` multiplies
   predominantly-negative similarity evidence on high-energy dense cells, so
   gain>1 deepens a negative cond contribution instead of adding mass.
   — *Supported by the pre-run dump* (`local/research/perimage_diagnostic.md:215–222`):
   wipes carry top1_mean −0.258, ev_sum −4924 → −5301 post-gain, gain_on_neg
   1.098; KEEP/MID net positive and gain adds mass (+566/+455). Sign of top1
   predicts wipe at 71% precision in gt[250,500) (`perimage_diagnostic.md`,
   Appendix; also `h0023a_idea_DRAFT.md:40`).

2. **Remedy:** "margin-sign-selective amplification restores the mass-adding
   invariant … by never amplifying negative margins while keeping the dense
   energy gain on positive margins" → then the THEN-bars (rescue wipes, val
   −0.30, dense <330.81, sparse ≤5.45).
   — **Failed as written.**

What the mechanism *actually* did, given w_m=+0.2666 and cellcal bypassed
(mechanism read: `journal/events.jsonl:59`):

- The margin branch is call-site exclusive: `CountingHead.forward` takes
  `if self.use_margin:` first (`model.py:177–178`) and never reaches the
  `elif self.use_cellcal:` arm (`model.py:179–183`), so `w_c` stayed at its
  zero-init (0) and `CellCal` (`model.py:266–277`) was dead code this run —
  exactly the booked "replacement / call-site-exclusive margin path"
  (`idea.md:3`, `idea.md:21`).
- Inside `SimPrior.forward`: `gain = exp(clamp(w_m·log1p(mhat), −3, 3))`,
  hard stop-grad mask `pos=(ev>0).detach()`, `ev = where(pos, ev·gain, ev)`,
  early `return self.out(ev)` (`model.py:237–244`). So per cell/channel:
  positives got a **stronger-than-parent** energy gain (w_m +0.2666 vs parent
  cellcal w_c +0.148; at mhat=1: ≈1.20× vs ≈1.11×), negatives were routed at
  **exact identity** (gain ≡ 1 by construction, not learned), and neither
  cellcal nor peakcal ever ran.
- Temp-pin engaged via `if self.head.use_peakcal or self.head.use_margin`
  (`model.py:341–342`); temp held at 0.07 (read: `events.jsonl:59`).

Outcome vs the remedy's predictions (`result.json:5–10`: best 22.163 @ep24,
`n_epochs_done` 24, `futility_hit` true; need 0.403/ep vs threshold 0.12 —
`events.jsonl:59`):

| Claim | Booked | Measured | Source |
|---|---|---|---|
| val MAE ≤19.0431 | −0.30 vs 19.3431 | **22.163 (+2.82)** | `val_result.json:8`; parent `idea.md:3` |
| dense gt>500 <330.81 | rescue | **458.69 (+127.9, WORSE)** | `val_result.json:19,33` |
| sparse gt<50 ≤5.45 | pass | 5.703 (improved from 5.814, still over) | `val_result.json:23`; parent from `N0015_val_perimage.json` recompute |
| ≥7/11 wipes ratio ≥0.65 | rescue | **0/11; 9/11 deepened; mean ratio 0.240→0.193** | paired `val_perimage.json` × `cac_backup/N0015_val_perimage.json` |
| R1 engage (w_m ≥ +0.10) | engage | **w_m=+0.2666, temp 0.07, w_c=0** | `events.jsonl:59` |

Per-image wipe pairs (pred→pred, same ids; both dumps, computed this session):
935: 0.403→0.148; 3428: 0.227→0.091; 3488: 0.192→0.092; 3487: 0.310→0.224;
851: 0.453→0.428; only 840 (0.281→0.285) and 865 (0.260→0.278) ticked up, both
still ≪0.65. The 11-image list is the booked set (`idea.md:15`).

**Reading:** the BECAUSE conflated "deepening is harmful" with "deepening is the
cause." The dump only ever supported the first; the appendix itself pre-priced
the gap: on wipes the damaged term is ~377 AE-units of deepening, and "stopping
it does not by itself supply the hundreds of counts the wipes lack"
(`perimage_diagnostic.md:226`). Identity-routing deletes the ~377 and leaves the
anti-phase substrate (ev_sum −4924) fully intact. Mechanism engaged, remedy
insufficient, causal claim as written **false**.

## 2. Honest design assessment: was the sign-mask the right operator?

**The mask was the right *local* operator for the measured harm and the wrong
*global* operator for the wipe — and 0/11 + dense-worse says the booked claim
over-extended a true local fact into a false causal theory.**

- What the mask got right: by construction it can never deepen a negative
  (`model.py:242–243`), it is mass-adding-only, zero suppression (explicitly
  not a ReLU/neg−pos form), one scalar, append-only RNG (`model.py:338`),
  step-0 identity via zero-init `out` (`model.py:224–226`). As a *repair of the
  deepening term* it is provably correct — R1's identity-on-negatives read is
  true by construction.
- What it could not do: anti-phase evidence is **negative-valued signal**, not
  "signal that was wrongly amplified." Routing it at identity converts
  `ev⁻·gain` back to `ev⁻` — still negative, still fed to `out`, still
  anti-correlated with mass on wipes (66.7% of wipe cells negative,
  `perimage_diagnostic.md:215`). The mask changes the *magnitude* of the poison,
  never its *sign* or its *presence*. 0/11 rescue with every wipe staying or
  deepening is the direct empirical statement of that.
- Dense +127.9 makes it worse than "insufficient": the operator also **damaged
  healthy dense scenes**. New collapses on images outside the booked wipe set,
  paired dumps: 3425 (gt 885) 795→112, 3436 (gt 443) 502→39, 949: 938→697,
  975: 529→405. Dense-slice AE 5624→7798 (+2174), 12/17 worse; mid AE
  14026→15601 (+1575), 199/374 worse; sparse AE 5203→5104 (−99, 470/895 better).
  So the claim "keeping the dense energy gain on positive margins" would preserve
  the winner's dense behavior is also false — dense got the worst of both worlds.
- Verdict on the claim: this is a **clean mechanistic refutation, not a quiet
  null** (contrast H0014 peakcal). w_m engaged positive and large, the mask
  did exactly what booked, and the world still moved the wrong way on all three
  bars plus R2. The falsifier worked exactly as a falsifier should.

## 3. Paired trajectory shape: mid-train gain, late loss

Trajectory (child TB `val/mae`, 24 points: ep14 23.4832, ep16 23.2461, ep23
22.2686, ep24 22.1632; parent points/gap Lead-verified `events.jsonl:59`):
child ahead of parent through ~ep14–15 (23.48 vs 23.92), crossover ~ep15–16,
+0.94 behind by ep23; child late slope ep18→24 = **−0.156/ep** (endpoint of TB
points) vs parent **−0.28/ep** in the same window; futility HALT at ep24
(`result.json:8–10`).

Mechanistic reading of the crossover, in three stacked costs:

1. **Early gain = stronger positive gain in an undercount regime.** Parent's
   confirmed cellcal ran at w_c=+0.148; the child replaced it with w_m=+0.2666
   on the positive half only (`model.py:241`, reads `events.jsonl:59`). Bias is
   strongly negative everywhere (child bias_mean_delta −14.58,
   `val_result.json:12`), so amplifying positives ~1.20× at mhat=1 while leaving
   negatives alone is directionally right early: the child leads through ~ep14.

2. **Identity-routing negatives removed a load-bearing term on the healthy
   majority.** KEEP/MID groups are net-positive but only ~48%/35% of their cells
   are negative (`perimage_diagnostic.md:215–217` table); the parent's
   sign-blind gain scaled *both* signs by the same spatial energy field
   `ev·gain` (`model.py:249–250`), preserving top1/cons contrast structure while
   reweighting by energy. The mask instead applies an *asymmetric* per-channel
   transform (channel scaled iff that channel >0, `model.py:242–243`): mixed-sign
   cells (top1>0>cons) see their two channels rescaled differently, so the
   input distribution to the zero-init `out` conv (`model.py:224`, 2→64) changes
   character everywhere, not only on wipes. The journal's own reading: "removing
   negative-gain on the healthy majority lost late-phase suppression" without
   converting any anti-phase evidence (`events.jsonl:59`). Measured cost lands
   mid+dense: mid AE +1575, dense AE +2174, with dispersion up on both sides
   (sum_over 4450→4875, sum_under 20403→23627; `cac_backup/N0015_val_result.json`
   vs `val_result.json:15–16`) — not a uniform shift the loss can absorb, a
   spatial-contrast degradation that produces new scene-level collapses
   (3425/3436 above).

3. **Late slope gap = one scalar cannot re-earn what the mask forbids.** Parent
   keeps two degrees of freedom (gain on both signs, balanced by the loss);
   child's only trainable is w_m, and its gradient sees positives only
   (`∂ev/∂w_m = pos·ev·log1p(mhat)·gain`, draft §2; `model.py:241–243`).
   Once the early positive-gain headroom is spent, the parent continues at
   −0.28/ep mining balanced-gain refinements while the child is capped at
   −0.156/ep polishing an operator that is wrong-signed on wipes and
   contrast-distorting on healthy dense — the +0.94 gap at ep23 is the integral
   of that slope difference after crossover, and futility correctly kills it
   (need 0.403/ep ≫ 0.12, `events.jsonl:59`).

One-line causal summary: **the mask bought the easy half (stop deepening) with
the wrong currency (lose negative-field contrast on the majority), and the
half that mattered (convert anti-phase evidence to mass) was never touched.**

## 4. Design-quality notes

**Falsifier well-posedness.** The three numeric disjuncts (val ≤19.0431, dense
<330.81, sparse ≤5.45) are specific, pre-registered, bound to the live parent
(`idea.md:9`), and all three failed on mechanism-level magnitudes far beyond the
+2.25 same-seed level noise the handoff tells us to ignore
(`perimage_diagnostic.md:229–231`; `events.jsonl:59`). The fourth disjunct —
"the negative-margin amplification fraction does not collapse to identity
(gain ≈ 1) on the wiped images" (`idea.md:9`) — is **vacuous as written**: the
hard `torch.where` mask makes gain≡1 on `ev<0` true by construction for any
trained weights (`model.py:242–243`), so it can never fire as a DISPROVED
condition; it is an R1-style engagement read misfiled as a falsifier (and its
polarity — "DISPROVED IF it does *not* collapse" — states the success condition
as the failure condition). Booking-quality lesson: mechanism-by-construction
reads must not be sold as independent falsifiers; the bars + R2 (0/11) did all
the real work here.

**Booked text vs implemented code** (checked line-by-line): **match.**
- Call-site exclusivity + cellcal/peakcal flags staying true in config:
  booked `idea.md:3,21`; implemented `model.py:177–178` (margin first,
  `elif` cellcal at 179), `config.toml:28–30` (`use_cellcal=true`,
  `use_peakcal=true`, `use_margin=true`). Intentional dead-code bypass, same
  precedent as N0015's dead peakcal.
- Gain/mask math: booked snippet `idea.md:24–26` ≡ `model.py:241–243`
  (clamp −3..3 included).
- Early return skipping parent cellcal/peakcal blocks: `model.py:244`
  before 245/251 — consistent with "margin path is call-site exclusive"
  (`idea.md:21`).
- `MarginCal` one zero-init scalar, constructed last: `model.py:295–306,338`;
  temp-pin condition extended: `model.py:341–342` ≡ `idea.md:30`.
- No second grad path / no shared projection / `e` read-only: `model.py:237`
  uses `fine.detach()`; qproj/kproj untouched (`model.py:221–222`). Clean on
  the §11 bans and the second-gradient-path family rule
  (`AGENTS.md:254–259`, STATE batch-3).
- Pluggability: single switch `use_margin` (`config.toml:30`,
  `model.py:164`) — passes the standing-regime ablation rule.

**Process nits (not verdict-affecting).**
- `idea.md` lacks the runner-parseable line `1. **H0022** — <booked text>`
  (STATE gotcha 5), and `info.json` accordingly carries
  `tested_hypotheses: []` despite a clean booking (journal 58). Repair via the
  API, not by hand (AGENTS §5.3).
- `info.json` `"status": "timeout"` on a futility HALT is *by design*
  (`events.jsonl:28`: "run_node maps futility_hit to timeout status") — not a
  bug, but easy to misread in feedback.
- The sparse ≤5.45 disjunct was a bar the live parent itself never met
  (parent sparse 5.814, `STATE.md:164`); the child improved it to 5.703 and
  still "failed" that limb. Binding anyway per §6 (bars are semantics, not
  literals of attainability), but the *decisive* failures here are val +2.82,
  dense +127.9, and 0/11 — sparse is a footnote.
- Handoff noise directive honored: verdict written as a single-draw paired
  contrast judged on mechanism-level moves (dense +127.9, 0/11, futility slope),
  all ≥ an order of magnitude above the ±2.25 level-noise datum
  (`perimage_diagnostic.md:229`).

## 5. What this rules in / rules out for the next card (H0023 substrate swap)

**Rules in.** H0022 walked the causal ladder one rung and falsified the lower
one *with the mechanism fully engaged*: with negatives provably never amplified
(w_m=+0.2666, mask by construction, cellcal bypassed), all 11 wipes stayed or
deepened and dense got worse — so the wipe is not carried by the gain's sign
domain, it is carried by the **substrate** (anti-phase similarity evidence
itself, ev_sum −4924, top1 −0.258). That is precisely the slot H0023
(`N0024_h0023`, `use_antimatch`) occupies: on the anti-match minority, stop
feeding `ev` to `out` at all and substitute the in-phase energy field `mhat`
before the *confirmed* cellcal gain runs (`h0023a_idea_DRAFT.md:46–64`,
booked one-liner `N0024_h0023/idea.md:9`). H0022 also supplies H0023's must-have
read design: R1 must show the routed/post-swap evidence sum *turn non-negative*
— not merely stop getting more negative — because H0022 proves magnitude-repair
without sign-conversion rescues nothing (0/11). And it rules in treating
F3-of-H0022 (wipe-not-gain-caused → exemplar-side / substrate successor,
`idea.md:34`) as the confirmed branch: the appendix's frontier call
(`perimage_diagnostic.md:226`) now has its confirmation experiment.

**Rules out.** (a) Any further gain-*domain* mask on the readout — sign-selective
or otherwise: form was closed by H0018, domain by H0022 (contradicts w=0.85,
`hypotheses.jsonl` evidence line). (b) "Amplify positives harder" as a wipe
cure: w_m 0.2666 ≫ parent 0.148 still ended +2.82 val / +127.9 dense — the
positive-gain lever is spent at the parent's operating point. (c) The strong
version of the booking's own theory, "cellcal's sign-blind deepening is *the*
wipe cause": at best a ~377-AE amplifier of a pre-existing anti-phase collapse
(`perimage_diagnostic.md:226`), not the source. (d) Quiet readings of this
refutation: engagement was real, so no successor may argue H0022 "failed to
optimize." (e) A softer version of the mask in any re-tuned form (soft gates,
partial identity, learned thresholds on margin sign) — same operator family,
would need a genuinely new falsifier per §11 rule 11 or it dies on contact.
Residual caution to carry into H0023: H0022's early lead (ahead through ~ep14)
shows the healthy majority *is* reachable through the simprior interface, but
H0023's swap must fire only on the ~2% anti-match minority (route read R1) —
the mid-train crossover here is the priced cost of touching the majority's
evidence distribution at all.

# Causal feedback — N0024_h0023 (H0023 `use_antimatch`, AntiMatch substrate-swap)

Causal agent, 2026-09-22 (rewrite: full file-backed pass). Inputs read: AGENTS.md
§3/§5/§6, NODE `val_attr.json` (1286 recs, dump_antimatch.py output — NOW LOCAL),
`val_result.json`, `val_perimage.json`, `result.json`, `model.py`, `idea.md`,
`config.toml`, `local/dump_antimatch.py` (field meanings), parent dumps
`/home/qkun/cac_backup/N0015_val_perimage.json` + `/tmp/opencode/val_attr.json`
(parent N0015 attribution, proven: `w_c=0.147648`, no `routed` key), sibling gold
template `../N0023_h0022/feedback/causal.md`, parallel fleet files
`feedback/{quant,qual,diagnostic}.md`, draft booking
`local/research/h0023a_idea_DRAFT.md` §4, ledger `memory/hypotheses.jsonl:44`.
No commit, no ssh.

Labels: **CONFIRMED** = recomputed from a local file this session · **CONSTRUCTION**
= forward-logic identity from `idea.md`/`model.py` (true by arithmetic, not learned) ·
**PLAUSIBLE** = mechanism inference consistent with the files, not ablated.

**Input-status correction vs earlier draft:** the prior `causal.md` ran with the
node dumps absent and marked child-side numbers **[lead-quoted]** / per-image cells
**BLOCKED**. All four artifacts (`result.json`, `val_result.json`,
`val_perimage.json`, `val_attr.json`) are now present in the node dir; every child
number below is recomputed from them this session. Corrections to the earlier draft
are listed in §7.

---

## 0. Verified outcome (CONFIRMED, files now local)

- `result.json`: `futility_hit=true`, `n_epochs_done=24`/32, best **23.0618 @ep24**,
  bar 19.0431. `val_result.json`: n=1286, MAE **23.0572**, RMSE 83.25; slices
  sparse **5.9717** (n=895) / mid **42.6369** (n=374) / dense **491.8111** (n=17,
  `dense_threshold=500`). Parent live 19.3431 (booked); parent local dump recompute
  19.3259, dense 330.8131 — so **dense Δ = +160.998 (+48.7%)**, mid Δ = +5.134,
  sparse Δ = +0.158, whole-val ΔMAE +3.731 (dump-pair) / +3.719 (vs booked live).
- Child AE (val_perimage) **29651.61** vs parent **24853.06** → **ΔAE = +4798.55**;
  ids/gt identical across the two dumps → paired partition exact.
- Wipes at the dump's WIPE definition (`dump_antimatch.py:51,167`, gt≥300 ∧
  ratio<0.5): child **n=19** vs parent **n=11** → **8 new wipes** (ids now known:
  1919, 1956, 3425, 3433, 3434, 3436, 3437, 5744; 6/8 routed; ΔAE +1674.8).
  R3 (no new wipes) **FAIL**.
- Mechanism scalars from `val_attr.json` (unique over all 1286 recs):
  **`w_c=0.02822391`**, **`w_p=0.00643087`**, **`temp=0.07`**.

---

## 1. Construction claims vs measured val reads (explicit split)

### CONSTRUCTION (from idea.md forward logic + `model.py` — true before any run)

| # | claim | code anchor |
|---|---|---|
| C1 | Hard per-image gate: `anti = (whole-image top1 mean < 0)`, stop-gradient | `model.py:241` |
| C2 | Routed substrate: `ev → cat(mhat, mhat)`, `mhat = (m/mbar).clamp(0,10)` is **mean-1, ≥0** | `model.py:238-243` |
| C3 | ⇒ pre-gain routed `∑ev = 2·96²·1 = 18432`, **scene-count-independent** | follows from C2 |
| C4 | ⇒ post-gain routed `evg_sum = ∑(mhat·gain) ≥ 0` for any `w_c` (`gain=exp(·)>0`) | follows from C2 + `model.py:248-249` |
| C5 | Predicted routed `evg_sum` at learned `w_c`: `18432·exp(w_c·ln2)` (at mhat≡1) | C3 + cellcal formula |
| C6 | Non-routed path byte-identical; route decision detached (no q/k/fine grad from routed imgs) | `model.py:241,243` |
| C7 | Peakcal skipped iff `|w_p| < 0.01` (reported, not applied) | `dump_antimatch.py:127-135` |
| C8 | Zero new parameters; `out` zero-init; `w_c`/`w_p`/`temp` parent-owned | `model.py:224-226`, `idea.md:3` |

### MEASURED (val_attr.json / val_result.json / parent dumps — recomputed this session)

Every quantity in §2–§4 is a measured read. Where a construction identity (C3–C5) is
checked against the dump, both sides are stated.

---

## 2. Booked mechanism reads — PASS/FAIL table (all fields from `val_attr.json`)

Dispatch booked reads for H0023: R1 route fire + construction `evg_sum` +18.8k,
`w_c`, `gain_on_neg`, temp pin, `w_p`; R2 wipe pred/gt vs parent; failure-mode
numbers (531/1286=41%, KEEP precision 0.63). Each verified against the file:

| read | field(s) | claim / bar | measured (val_attr.json) | verdict |
|---|---|---|---|---|
| **R1a** route fires, post-swap `evg_sum` non-negative on ≥90% of routed | `routed`, `evg_sum` | ≥90% of routed have `evg_sum≥0`; booked level +18.8k | **531/531 = 100%** with `evg_sum≥0`; routed mean **+18804.2**, median **+18800.4**, min **+18796.8**, max +18903.9 | **PASS** (100% ≥ 90%; +18.8k confirmed per-image) |
| **R1b** construction identity check | `evg_sum` vs C5 | `18432·exp(0.02822391·ln2)` | predicted **18796.1**; observed min **18796.83**, mean **18804.23** → match **≤0.04%** | **PASS** (construction-true to file precision) |
| **R1c** route fires on the anti-match *minority* | `routed` | booked "~2–3% of val" (`draft:134`) | **531/1286 = 41.29%** of all val | **FAIL** (20× over the booked minority rate — the *scope* half of R1) |
| **R1d** route fires on ≥90% of the booked 11 wipes | `routed` on booked-11 | ≥90% (draft R1) | **7/11 = 63.6%** (unrouted: 840, 935, 3482, 3487) | **FAIL** |
| **w_c** dispatch-fix (alive, ~+0.15 not 0) | `w_c` | ≠0; parent band 0.148–0.166 | **0.02822391** (constant) | **PASS as alive (≠0)**; **FAIL vs ~0.15** — 5.2× weaker than parent `w_c=0.147648` |
| **gain_on_neg** | `gain_on_neg` | KEEP sanity ≈1.08 (`dump_antimatch.py:12`) | overall mean **1.0186** / med 1.0191; KEEP mean **1.0138**, WIPE mean **1.0167** | **FAIL vs 1.08** (parent means: KEEP 1.0758 / WIPE 1.0979) — gain active (>1) but ~5× shallower |
| **temp pin** | `temp` | stays 0.07, untouched by swap | **0.07** (unique over 1286) | **PASS** |
| **w_p** | `w_p` | dead peakcal (parent ≈0.0017, null delta) | **0.00643087** < 0.01 ⇒ peakcal **skipped** at dump (C7) | **PASS** (still dead; null delta as booked) |
| **R2** wipe pred/gt vs parent | `ratio` on booked-11 | ≥7/11 at ≥0.65; mean ratio 0.241→≥0.55 | **0/11 at ≥0.65**; **0/11 even at ≥0.5**; ratios 0.052–0.478, mean **0.2002** (parent **0.2410** — *worsened*) | **FAIL** (deciding read) |
| **F-numbers (dispatch "F3")** route-gate too coarse | `routed` by bucket | 531/1286 = 41%; KEEP precision 0.63 | **531/1286 = 0.4129 ✓**; KEEP (gt≥300, ratio≥0.5) **12/19 = 0.6316 ✓**; also WIPE 13/19 = 0.6842, MID 194/353 = 0.5496, sparse 312/895 = 0.3486 | **VERIFIED — both numbers match the file exactly** |

Label note: the dispatch parenthesizes these coarse-gate numbers under "F3".
`idea.md:35` books them as **F2 (route-too-coarse)** with **F3 = KEEP degradation**
(the 0.63 KEEP fire rate *is* that degradation read); `draft:127` / `qual.md` /
`diagnostic.md` use **F3 = wipe-not-substrate / MSE-veto** (the 0/11-under-+18.8k
signature). All three number sets are verified below (§5); no number depends on the
label.

**Booked bar summary:** R1a/R1b PASS · R1c/R1d FAIL · R2 FAIL · R3 (new wipes) FAIL ·
triple val/dense/sparse bars 0/3 · temp/w_p PASS · w_c/gain_on_neg alive-but-weak.
The decisive signature stands: **the swap fired construction-true (+18.8k) and the
count bars still all failed.**

---

## 3. WIPE vs KEEP mechanism fields (mean / median over wipes vs KEEP)

Group defs per `dump_antimatch.py:167-168` (dense=300 default): WIPE = gt≥300 ∧
child ratio<0.5 (n=19); KEEP = gt≥300 ∧ ratio≥0.5 (n=19); MID = 50≤gt<300 (n=353).
All values recomputed from `val_attr.json`:

| field | WIPE mean | WIPE median | KEEP mean | KEEP median | reading |
|---|---:|---:|---:|---:|---|
| `routed` | 0.6842 | 1 | 0.6316 | 1 | gate barely separates wipes from keeps (Δ only 0.05) |
| `top1_mean` | −0.1598 | −0.1799 | −0.0887 | −0.1200 | KEEP also sits near/below zero → mass over-fire |
| `ev_sum_pswap` | −3110.0 | −3460.2 | −1783.0 | −2398.9 | pre-swap anti-phase extends into healthy KEEP (12/19 routed) |
| `evg_sum` | +13403.3 | +18808.1 | +13002.8 | +18828.0 | post-gain mass is + and ~equal across wipe/keep — no wipe-specific signal |
| `gain_on_neg` | 1.0167 | 1.0177 | 1.0138 | 1.0138 | floor active but ≈1.01–1.02 everywhere (parent: 1.098/1.076) |
| `neg_ev_frac` | 0.7013 | 0.7431 | 0.5822 | 0.6290 | wipes more negative pre-swap (as booked) yet both groups route ~2/3 |
| `mhat_mean` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | mean-1 identity (C2) holds on every image (global mean 0.9999968) |
| `ratio` | 0.2540 | 0.2871 | 0.7044 | 0.6781 | outcome gap untouched by the swap |

Cross-checks with construction: routed-only rows — `ev_sum_pswap` mean **−2774.0**
(anti-phase in) → `evg_sum` mean **+18804.2** (positive out), 0/531 negative (R1a).
Unrouted rows keep parent arithmetic: `ev_sum_pswap` mean +2604.9 → `evg_sum` mean
+2666.2 (gain-only, no swap). Global `mhat_mean` ≈ 1.0 confirms C2/C3 on all 1286.

---

## 4. The causal chain — every arrow with a measured number

**Thesis (matches the dispatch):** substrate swap construction-true (+18.8k) yet
0/11 rescue + dense +161 ⇒ consumption-interface veto (F3), with the coarse gate
(F2) as co-fired amplifier.

### Arrow A → B — the swap executed exactly as constructed (CONFIRMED)

- On the **routed booked-7** (851, 865, 3428, 3481, 3484, 3488, 7656): pre-swap
  `ev_sum_pswap` = −5718, −4924, −6524, −3112, −4701, −458, −12843 (all anti-phase)
  → post-swap `evg_sum` = **+18816.7, +18843.7, +18812.1, +18815.8, +18808.1,
  +18826.9, +18839.2** (mean **+18823.2**). The same 7 ids at parent weights carried
  `evg_sum` sum **−62236** (parent attr: −11843…−15153, +5913 for 7656).
  **Sign flip delivered: ≈ +81k swing in evidence sum.**
- Construction identity (C5): predicted 18796.1 vs observed min 18796.83 — the
  +18.8k is the architecture's constant (18432·gain), not a learned count signal.
- **R1a PASS**: 100% of 531 routed images have `evg_sum ≥ 0` (bar ≥90%).

### Arrow B → C — despite that, zero counts moved on the target wipes (CONFIRMED)

- Booked-11 `ratio` (val_attr): 0.315, 0.478, 0.318, 0.139, 0.105, 0.052, 0.090,
  0.053, 0.244, 0.123, 0.287 — **0/11 ≥ 0.65 (bar ≥7/11), 0/11 ≥ 0.5**;
  mean **0.2002** vs parent **0.2410** (worsened by −0.041). The 7 that *did* route
  with +18.8k each still sit at ratio 0.052–0.478 (max routed = 851 at 0.478).
- The 4 unrouted wipes (`top1_mean` **+0.035 / +0.103 / +0.060 / +0.097** for
  840/935/3482/3487) also stayed — 935 even fell 0.403→0.139.
- **R2 FAIL**: a phase-corrected, construction-positive evidence tensor produced
  zero counts. This arrow is within-checkpoint (same dump, same weights) and
  draw-invariant.

### Arrow C → D — the cost landed on dense and the healthy majority (CONFIRMED)

- dense **491.8111 − 330.8131 = +160.998** (bar was <330.81) — the "+161" arrow;
  wipes **19 vs 11** (8 new, ΔAE +1674.8, 6/8 routed — the swap ran on most of the
  new collapses); mid +5.134; sparse +0.158 (parent 5.814 → 5.972, bar 5.45).
- Route scope measured: **531/1286 = 41.29%** all-val; **KEEP 12/19 = 63.16%**;
  WIPE 13/19 = 68.42% (KEEP-vs-WIPE route gap only 0.05 → gate precision ≈ base
  rate); MID 194/353 = 54.96%; sparse 312/895 = 34.86%. On those 531 the healthy
  images lost both the working parent cosine path (C6: byte-swap + stop-grad) and
  received the count-free 18432-integral field (C3).
- Actuator weakened in the same run: `w_c` 0.147648 → **0.02822391**,
  `gain_on_neg` 1.076–1.098 → **1.0186** — the "identical confirmed gain" of the
  BECAUSE held only at source level, not at runtime depth (PLAUSIBLE cause:
  two-regime gradient stats on shared `out`/`w_c` under 41% routing; not ablated).

### Arrow D → conclusion — consumption-interface veto = F3 (CONFIRMED pattern; locus PLAUSIBLE)

- What the chain establishes by elimination, all file-backed: evidence **sign** was
  fixed (A: −2774 → +18804); the gate **did** fire on the targets (7/11); the gain
  was **alive** (`w_c≠0`, `gain_on_neg>1`); counts **did not move** (B: 0/11) and
  the slices went the wrong way (C: +161/+5.13/+0.16). The failure therefore sits
  **below the evidence/gain layer** — at `out` (2→64 zero-init, `model.py:224-226,
  262`) → `cond_map` add (`:181-188`) → `decoder` with GroupNorm (`:128-140,189`):
  a mean-1, channel-duplicated, scene-constant-integral field (C2/C3) carries no
  count scale for that path to consume. This is the draft's own pre-registered
  **F3** ("route fires, evidence in-phase, wipes stay wiped — out maps energy to
  unsharp mass the MSE vetoes", `draft:127`), with the H0016/H0017 quiet-null
  precedent for energy-mass paths. The GN-uniform-shift step is **PLAUSIBLE**
  (code structure CONFIRMED, not ablated).
- Co-fired amplifier (idea.md **F2** / the dispatch's coarse-gate numbers, both
  verified §2): 41.29% over-fire with KEEP 0.6316 explains the +8 new wipes, the
  sparse/mid slip, and why this run is worse than sibling N0023 — but it cannot
  explain B: fixing the gate would leave the correctly-routed 7 still at 0/11 under
  +18.8k. **F3 is primary and gate-independent; F2 is what made the damage large.**
  F1 (route-ineffective) did **not** fire: `evg_sum` never went negative.

---

## 5. Failure-mode number verification (dispatch asks: verify each)

| number in dispatch | field source | measured | match? |
|---|---|---|---|
| route fire 531/1286 = 41% | `sum(routed)` / n | 531/1286 = **0.412908** → 41.29% | **yes** (41% at 2 d.p.) |
| KEEP precision 0.63 | KEEP bucket routed frac (`gt≥300, ratio≥0.5`) | 12/19 = **0.631579** → 0.63 | **yes** (exact at 2 d.p.) |
| construction `evg_sum` +18.8k | routed `evg_sum` | mean +18804.2, med +18800.4, booked-7 mean +18823.2 | **yes** (per-image level; see §7 correction on "aggregate") |
| R2 wipe pred/gt vs parent | `ratio` booked-11 vs parent dump | 0/11 ≥0.65; child mean 0.2002 vs parent 0.2410 | **yes — FAIL as booked** |
| `w_c` cellcal weight | `w_c` | 0.02822391 (alive, ≠0; 5.2× below parent band) | **yes** (0.0282) |
| `gain_on_neg` | `gain_on_neg` | overall 1.0186 (KEEP 1.0138 / WIPE 1.0167) | **yes at WIPE≈1.017; overall corrects to 1.0186** (§7) |
| temp pin | `temp` | 0.07 | **yes** |
| `w_p` | `w_p` | 0.00643087 (<0.01 ⇒ peakcal skipped) | **yes** |

No dispatch number failed verification; two got precision/scope corrections (§7).

---

## 6. The single most load-bearing causal statement

**Combined with N0023's** — MarginCal's `pos=(ev>0)` mask never touched the
net-negative anti-phase field (parent WIPE `ev_sum` mean −4924, 67% negative cells)
so 0/11 was decided at code-write time (`N0023/feedback/causal.md` headline) —
**N0024 closes the same layer from the opposite side and still gets 0/11:** a
correctly-fired substrate swap that lifted routed post-gain evidence from ≈−62k
(parent same-7 sum) to **+18804 mean (0/531 negative, construction identity matched
to ≤0.04%)** moved **zero counts** (booked-11 mean ratio 0.2410→**0.2002**, 0/11 ≥
0.65) while the coarse gate simultaneously routed **41.29%** of all val
(**KEEP 0.6316**) and pushed dense to **+160.998** with **8 new wipes**. Joint
localization, every arrow file-backed: the dense wipe lives **below the
evidence/gain layer** — in the evidence's missing count-bearing absolute magnitude
and its consumption at `out→cond→decoder` (F3) — not in evidence sign, phase, or
gain; and a whole-image sign gate at 41% prevalence cannot scope-select the minority
(F2). Anti-phase is a correlate of the wipe, not its operative substrate.

*Noise note carried, not hidden: +3.72 ≈ +1.2 parent-draw-luck + ~2.5 child
deviation (~2.3σ, `diagnostic.md:36-47`). The statement above rests only on
within-checkpoint mechanism reads (0/11 under construction-true +18.8k; identities
C2–C5; bucket counts) that a draw cannot produce.*

---

## 7. Corrections vs the earlier (file-missing) draft

| earlier draft claim | file-measured correction |
|---|---|
| `gain_on_neg 1.017` **[lead-quoted]** | overall mean **1.0186** (1.0167 is the WIPE-group mean specifically; KEEP 1.0138) |
| routed `evg_sum +18.8k` "each set-level" / quant's "aggregate" | **per-image**: routed mean +18804.2, booked-7 mean +18823.2; the sum over routed-7 is **+131762**, not 18.8k — 18.8k is the per-image construction constant |
| `18432·exp(0.0282·ln2)≈18796` matches "+18.8k to ~0.5%" | with exact `w_c=0.02822391`: predicted **18796.07** vs observed min **18796.83** / mean **18804.23** → match to **≤0.04%** (0.5% was loose) |
| MID routed "193–195/353" range | exact **194/353 = 0.5496** |
| sparse routed "≈311–313" residual | exact **312/895 = 0.3486** |
| booked-11 child mean ratio "0.2004" (derived from 3-dp quotes) | **0.20021** from full-precision `ratio` (parent 0.24095) |
| new-wipe ids **BLOCKED** | now computed: **1919, 1956, 3425, 3433, 3434, 3436, 3437, 5744** (ΔAE +1674.81); 6/8 routed |
| §d partition rows (i)/(iii)/(iv) BLOCKED | child `val_perimage` local: total AE 29651.61, ids==parent ids — full paired partition now computable (done for new-wipes and booked-11 ΔAE +605.29) |
| child scalars [lead-quoted] | exact constants: `w_c=0.02822391`, `w_p=0.00643087`, `temp=0.07` |
| unrouted booked top1 +0.035/+0.060/+0.097/+0.103 | **unchanged — verified exact** |
| KEEP 0.63 / 531 / 41% | **unchanged — verified** (0.6316 / 531 / 0.4129) |
| parent refs (19.3259, WIPE −4924/−5301/1.098, KEEP +2455/+3022/1.076, `w_c` 0.147648, gate sim 322/1286) | **unchanged — re-verified from parent dumps this session** |

---

## Source citation per section

| § | claims | source |
|---|---|---|
| 0 | endpoint, slices, AE, wipe ids, scalars | NODE `result.json`, `val_result.json`, `val_perimage.json`; parent `N0015_val_perimage.json` |
| 1 C1–C8 | construction identities | `idea.md:19-35`, `model.py:224-226,237-249,262`, `dump_antimatch.py:127` |
| 2 | every PASS/FAIL row | NODE `val_attr.json` (recomputed); parent `w_c`/`gain_on_neg` from `/tmp/opencode/val_attr.json`; bars from `idea.md:9,35` + `draft:114-129` + dispatch |
| 3 | WIPE/KEEP mean-median table | NODE `val_attr.json`, group defs `dump_antimatch.py:167-169` |
| 4 A–D | chain arrows | `val_attr.json` per-image rows (booked-11, routed buckets) + `val_result.json` slices + parent attr routed-7 `evg_sum` + `model.py` lines as cited |
| 5 | dispatch-number verification | `val_attr.json` + `dump_antimatch.py` bucket logic |
| 7 | corrections | this session's recompute vs the earlier draft text |

*Files written: this document only —
`tree/…/N0024_h0023/feedback/causal.md`. No commit, no ssh, no ledger/info.json
edits.*

# Causal feedback — N0022_h0021 (H0021 `use_exkern`, REFUTED quiet-null)

Causal agent, BACKFILL, 2026-09-22. Inputs read: AGENTS.md §3/§5/§6, NODE
`model.py`/`idea.md`/`config.toml`/`result.json`/`val_result.json`/
`val_perimage.json`/`run/latest/tb`, parent N0015 `model.py`, parent dump
`/home/qkun/cac_backup/N0015_val_perimage.json` (+`N0015_val_result.json`),
`src/cac/models/pl_module.py`, `src/cac/engine/runner.py`, ledger
`memory/hypotheses.jsonl` (H0021 evidence), `journal/events.jsonl` (precision
draws), STATE.md. No commit, no ssh. Every number below was recomputed from
those files this session.

Labels: **CONFIRMED** = directly file-backed (code line or recomputed dump
number) · **PLAUSIBLE** = mechanism inference consistent with the files but not
ablated · **SPECULATIVE** = consistent story, no direct file support.

---

## 0. Verified outcome and accounting (CONFIRMED)

- `result.json`: `futility_hit=true`, `n_epochs_done=24` of 32, best
  **21.9284 @ep24**, bar 19.0431 (missed by +2.89). Config delta vs parent:
  **exactly one line**, `use_exkern = true` (diff of the two `config.toml`s);
  run overrides `augment=true`, `futility_bar=19.0431`.
- `val_result.json`: n=1286, MAE 21.9275, dense(gt≥500) 454.451, sparse
  5.7913, mid 40.8822, bias −15.14 (mean_pred 48.40 vs mean_gt 63.54).
  Parent backup dump: MAE 19.3259, dense 330.813, bias −12.41 — the live
  booked parent number is **19.3431** (STATE/journal; 0.017 dump-vs-record
  EMA nuance, same as N0023's accounting).
- Ledger contrast (H0021 evidence, hypotheses.jsonl:40): val 21.928 vs
  19.3431 = **+2.59**; dense **+123.6**; mid 40.9 vs 37.5; sparse flat;
  **w_x = −0.0398, never engaged positive**; futility need **0.367/ep**.
  Arithmetic check from local tb: best-before-ep24 = 21.9796@ep23, so
  (21.9796−19.0431)/8 = **0.36706** — the ledger number reproduces exactly
  (runner.py:185–187 prints `need X/ep vs ref 0.06×2.0`).
- w_x provenance: endpoint read from server `best.pth` at run close, recorded
  only in the ledger note (checkpoint not local, no ssh in this backfill) →
  file-backed-secondary. **w_x trajectory is not logged**: local tb scalar tags
  are {hp_metric, lr-AdamW, train/loss, epoch, val/mae, val/rmse,
  train/loss_epoch, train/mae} — no mechanism scalars, so trajectory claims
  below are labeled accordingly.
- Child val curve (tb, CONFIRMED): 158.6@ep0 → 22.74@ep16 → 21.98@ep23 →
  21.9284@ep24 HALT; realized late slope ep16→ep24 ≈ **0.101/ep**, against a
  required 0.367/ep → halt correct, and the run was still ~2.9 MAE from the
  bar with 8 epochs left. The halt is not the cause of failure; the bars were
  already unreachable.

## 1. ROOT — why the optimizer never engaged the path

Ledger claim under test: *“joint MSE/L1 loss never rewards the path →
gradient ~0 → learned fusion weight ignores it.”* Audited link by link:

| # | Link | Verdict | Evidence |
|---|---|---|---|
| A | There is **no direct reward** for `w_x` in the loss | **CONFIRMED** | Loss is only `MSE(density) + 0.4·L1(Σdensity, count)` on `out["density"]` (`pl_module.py:20–32`); nothing supervises `ch_cal`/`w_x`. |
| B | There **is** a structural gradient path (so “never rewards” ≠ “gradient structurally zero”) | **CONFIRMED** | `cond_map = cond_map + w_x · ch_cal` (`model.py:248`) is live: ∂L/∂w_x = Σ (∂L/∂cond_map)·ch_cal (1-channel broadcast over 64 cond channels), no detach on `w_x` or on the product. |
| C | External feature is **stop-gradient** | **CONFIRMED** | `fd = fine.detach()` (:37), kernel `Kbar…​.detach()` (:62), calibration pool `chd = ch.detach()` (:66) — the whole `ch_cal` field has `requires_grad=False`; only `w_x` is live. Detach blocks grad into the fuser via this path, **not** into `w_x`. |
| D | Zero-init sits on a **saturating nonlinearity** (dead-at-init)? | **CONFIRMED — NO** | Fusion is a plain multiply-add straight into the decoder conv stack (`model.py:248→249`); no sigmoid/tanh/ReLU gate at the fusion point. Dead-gradient-at-init is ruled out as the quiet-null cause. |
| E | Scalar is in the optimizer with wd | **CONFIRMED** | `filter(p.requires_grad, model.parameters())` → AdamW, lr 1e-3, **wd 0.08** (`pl_module.py:56–58`, `config.toml:42`); `w_x` is `requires_grad=True` (`model.py:359`). |
| F | “gradient ~0” | **~0 as in zero-mean/noise-dominated: PLAUSIBLE · identically zero: REFUTED by the endpoint itself** | AdamW with an identically-zero gradient leaves `w_x` at exactly 0 (decoupled wd multiplies 0 → 0). Observed **w_x = −0.0398 ≠ 0** (ledger) proves nonzero gradient mass reached the scalar at some steps — “never rewarded” is right about *systematic* reward, wrong as *structurally no gradient*. Sign is slightly **negative**, i.e. the density loss weakly preferred *removing* the field, not “ignoring” it in the strict sense. |
| G | Why the correlation’s expectation ≈ 0 (**the actual root**) | **PLAUSIBLE** | `ch_cal` is a deterministic function of `(fine, boxes)` — `fine` is already concatenated into the decoder input (`model.py:249`) and box-position information already reaches `cond_map` through cross-attn over `e` (RoI’d from h3 at the same boxes, `:229–231`). The added field carries **no information the trunk lacks**, so the density loss has no reason to buy or sell it consistently → ∂L/∂w_x is a near-zero-mean correlation → AdamW noise-walk + wd parks `|w_x| ~ 0.04`. Redundancy components are code-CONFIRMED; the zero-mean⇒drift step is unmeasured (no grad/w_x logging). |
| H | “fusion weight ignores it” (path inert at eval) | **CONFIRMED** (endpoint ledger-backed; arithmetic local) | Direct forward effect is `w_x·ch_cal` ≈ −0.04·O(1) added to 64 of 192 decoder input channels ≈ **~2% scale** — the path is effectively off at inference. |

**Root cause (one paragraph):** the quiet-null is not a blocked gradient, not
a dead/saturated init, and not an excluded parameter — all three are ruled out
by code (B, D, E). It is **zero expected reward**: the only training signal is
the density MSE/L1, the sole coupling to `w_x` is its correlation with the
density-error gradient, and `ch_cal` is an information-redundant, fully
detached re-encoding of `(fine, boxes)` the trunk already consumes (G). That
correlation is zero-mean noise, so the scalar never receives a consistent
push, AdamW+wd parks it at −0.0398, and the CountingDINO transfer stays
disengaged — exactly failure mode F1 as pre-registered in `idea.md:46`.

## 2. Causal chain to +2.59 / dense +123.6

| # | Link | Label |
|---|---|---|
| C1 | Config delta is one line (`use_exkern=true`); model delta is the exkern function + flag + forward branch + `ExKern` scalar (parent `model.py` diffed — no other change); step-0 forward byte-identical at `w_x=0` | CONFIRMED |
| C2 | Endpoint `w_x = −0.0398`, never positive (trajectory unlogged) | CONFIRMED (ledger read) |
| C3 | Direct inference effect of the path ≈ 2% scale → **mechanism contributed ~nothing to predictions directly** | CONFIRMED (arithmetic) |
| C4 | But `w_x ≠ 0` ⇒ forward no longer byte-identical to parent ⇒ shared-module training trajectory diverges from N0015’s; level outcomes are draw-dominated: same-seed draws **{19.34, 20.70, 21.59}** (N0015, N0021 dead-code-clean retrain, exact N0015 repro — journal events 55/57) with mean 20.54, sd 1.13 | CONFIRMED (journal) |
| C5 | Futility HALT ep24 (need 0.367/ep vs ref 0.06×2.0; realized 0.101/ep) — verdict already sealed: +2.89 from the bar with 8 epochs left | CONFIRMED (tb + runner arithmetic) |
| C6 | Outcome shape (§3) is dense/mid undercount collapse — wipe phenotype — with sparse flat | CONFIRMED (dumps) |
| C7 | Sibling N0023 (MarginCal, mechanism **engaged** w_m=+0.2666) produced nearly the same signature (+2.82, dense 458.7) while N0022’s mechanism was **disengaged** → the outcome magnitude is mechanism-insensitive | PLAUSIBLE (journal event 59 vs ledger 40) |

### Noise decomposition of the +2.59 (assumptions stated)

Assumptions: (a) same-seed draws are exchangeable across these near-identical
forwards (N0021’s delta was dead-code only; N0022’s delta is a ~2%-scale
residual); (b) identical EMA/eval protocol for all draws; (c) n=3 → σ≈1.13 is
a very noisy estimate; (d) any structural effect of `use_exkern` enters the
level additively on top of the draw.

| Method | Structural share | Draw-contaminated share |
|---|---|---|
| A — center on draw mean: 21.9284 − 20.5443 | **+1.384** (≈54%) | +1.201 (parent’s lucky draw: 20.5443 − 19.3431) ≈46% |
| B — closest pairing (exact same-seed N0015 repro 21.5928): 21.9284 − 21.5928 | **+0.336** (≈13%) | +2.250 (21.5928 − 19.3431) ≈87% |

**Range: structural +0.34…+1.38 of the +2.59 → at least ~46% of the headline
is draw-contaminated under either method; the booked parent 19.3431 is itself
the lucky tail of the draw set.** Dense cannot be decomposed the same way (no
same-config repro per-image dump locally) — its structural share is
SPECULATIVE, bounded indirectly in §3 (level-fit residual ≈ −2). Per AGENTS
§6 and the handoff: **the mechanism verdict rests on the w_x read (−0.0398,
never positive) and the absence of engagement, not on the level.**

## 3. AE attribution (two val dumps, common ids, n = 1286 exact match)

Dump-pair totals: parent AE **24853.1**, child AE **28198.8**, **ΔAE = +3345.7**
(ΔMAE +2.6017; ledger’s +2.59 uses the 19.3431 record instead of the 19.3259
dump — 0.017 difference). Buckets = `val_result` partition (sparse<50 /
mid 50–500 / dense≥500 = 895/374/17, recomputed exact).

| Bucket | n | child MAE | parent MAE | slice ΔMAE | contribution to total ΔMAE | share |
|---|---:|---:|---:|---:|---:|---:|
| sparse (gt<50) | 895 | 5.791 | 5.814 | −0.022 | **−0.016** | −0.6% |
| mid (50–500) | 374 | 40.882 | 37.503 | +3.379 | **+0.983** | +37.8% |
| dense (≥500) | 17 | 454.451 | 330.813 | +123.638 | **+1.634** | +62.8% |
| **total** | 1286 | 21.928 | 19.326 | +2.602 | +2.602 | 100% |

- Mean signed pred shift child−parent: sparse **+0.15**, mid **−4.13**, dense
  **−123.6** — the whole regression is deeper undercount (child n_under 754 /
  sum_under 23834.8 vs parent 762 / 20403.3; sum_over slightly *down*
  4449.8→4364.0). Sparse is flat vs parent (+0.02 better): **no dense-bias
  leak through the new path** — while the absolute sparse bar ≤5.45 was
  missed (5.791), exactly as the parent itself missed it (5.814).
- Concentration: worse 660 ids contribute +3.943, better 626 ids −1.341;
  **top-10 delta ids alone = +2.080 (80% of the total)**, top-17 = +2.370.
- Worst movers: 3425 (+662.7), 935 (+579.1), 3436 (+351.8), 6969 (+246.8),
  3437 (+242.4), 949 (+149.4), 975 (+145.4), 3433 (+124.9). Best movers:
  865 (−80.8), 1956 (−79.7), 3665 (−53.6).

### Shape vs the known wipe phenomenon

- **Phenotype matches**: dense slice is pure undercount (child
  `mean_signed_delta = −454.45` in `val_result`), ratio collapses
  `pred/gt < 0.5` exactly as in the wipe corpus; wiped ids parent n=71 →
  child n=78.
- **Carriers differ**: the historical 11-wipe set (935, 7656, 865, 3481,
  3482, 840, 3484, 851, 3428, 3487, 3488) contributes only
  **+558.5 AE (16.7%)** and is mixed — 6/11 deepened (935 0.40→0.13
  dominates), 865 actually *improved* (0.26→0.34). The vehicle is **14 new
  ratio collapses** (parent ratio ≥0.5 → child <0.5): **+1605.3 AE =
  +1.248 ΔMAE ≈ 48% of the total**, headlined by
  3425 (0.90→0.15, pred 795→132), 3436 (1.13→0.07), 3437 (0.86→0.38),
  3433 (0.66→0.43) — those four alone = **+1381.8 AE = 41.3% of ΔAE**.
  So: same wipe *shape*, mostly **newly wiped previously-healthy dense/mid
  images**, not a worsening of the old wipe group.

### Dense blowup is level-coupled, not exkern-specific (PLAUSIBLE)

Lineage observational fit (journal/STATE file-backed points, child excluded):
`dense ≈ −526.3 + 44.82 · val_level` predicts child dense = **456.4** vs
observed **454.45 → residual −2.0**. Cross-checks: N0013 (20.89) dense
421.1; baseline-v2 (21.44) dense 423.4; N0023 (22.16) 458.7; N0016 (22.19)
≈485.8; N0018 (23.38) ≈510.4. Dense MAE tracks val level at ~45 dense-MAE
per 1 val-MAE in this regime — the +123.6 is what +2.59 of level already
buys, consistent with draw/trajectory divergence (C4) rather than a
path-specific dense effect. Heterogeneous configs → PLAUSIBLE, not proven.

## 4. The single most load-bearing causal statement

**Because `ch_cal` is a fully-detached, information-redundant function of
`(fine, boxes)` that the decoder and cross-attn condenser already consume, and
the density MSE/L1 couples to `w_x` only through its correlation with the
density error, ∂L/∂w_x stays zero-mean noise that AdamW+wd parks at −0.0398 —
the path never engages (and not for lack of gradient plumbing: the scalar is
live, in the optimizer, and off no saturating nonlinearity). The +2.59 /
dense +123.6 is therefore draw-level divergence wearing the wipe phenotype,
not an exkern effect: same-seed N0015 draws already span 19.34–21.59, the
dense blowup is fully level-predicted (fit residual −2.0), and 14 new ratio
collapses on previously-healthy images carry 48% of the ΔAE while sparse stays
flat.**

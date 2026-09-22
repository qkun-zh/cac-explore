# Causal feedback — N0025_h0024 (H0024 `use_canchor`, CountAnchor post-`out` residual α scale)

Causal agent, 2026-09-22. Inputs read: AGENTS.md §3/§5/§6, NODE `model.py` / `idea.md` / `config.toml` / `result.json` / `info.json` / `run/latest/tb` (`val/mae`, `hparams.yaml`), parent dumps `/home/qkun/cac_backup/N0015_val_perimage.json` + `N0015_val_result.json`, sibling gold templates `../N0023_h0022/feedback/causal.md` and `../N0024_h0023/feedback/causal.md`, leverage-probe premises in `../N0026_h0025/idea.md` + `journal/events.jsonl:65`, ledger `memory/hypotheses.jsonl:45`. No commit, no ssh.

Labels: **CONFIRMED** = file-backed this session · **CONSTRUCTION** = forward-logic identity from `idea.md`/`model.py` (true by arithmetic) · **PLAUSIBLE** = mechanism inference consistent with the files, not ablated.

**Artifact status (load-bearing):** under the node there is **no** `val_result.json`, **no** `val_perimage.json`, **no** `val_attr.json` / alpha dump, **no** test dump, **no** local train log. Only `result.json`, `info.json`, `idea.md`, `model.py`, `config.toml`, and `run/latest/tb/…10646.0`. Every child-side mechanism read that would require per-image or alpha fields is reported **not in dump**.

---

## 0. Verified outcome (CONFIRMED)

- `result.json`: `futility_hit=true`, `n_epochs_done=24`/32, best **22.22052001953125 @ep24**, bar `futility_bar=19.0431`, `budget_hit=false`, elapsed_s 1333.6, shas `config f30d9a561f34090d` / `model 8599b71f9a5644dd`, overrides `{augment: true, futility_bar: 19.0431}`.
- tb `val/mae` @ep24 = 22.22052001953125 **exactly** = `best_mae` (monotone 141.692 → 22.221 over 24 pts). Futility arithmetic (threshold 0.12): ep16 best 24.6253 need 0.3489 → WARN; ep24 need 0.3972 > 0.12 → **HALT**, ceiling 20.0031, gap +2.2174. Booked gate “best > 20.00 at ep24 ⇒ HALT”; 22.2205 is **above the (20.0, 21.6) draw-contaminated band** (`idea.md:132,139`).
- Parent live 19.3431 (booked); parent dump recompute 19.3259, dense 330.8131 — so measurable val **Δ = +2.8774 vs live / +2.8947 vs dump**. Dense/sparse/mid child slices: **not in dump**.
- `config.toml` delta vs parent = exactly `use_canchor = true` (filtered diff one line); `use_simprior/cellcal/peakcal` all true. `info.json` records `status: "timeout"` while `result.json` has `futility_hit=true` — status label as written by the runner, not re-interpreted here.

---

## 1. Construction claims vs measured reads

### CONSTRUCTION (from `model.py` + `idea.md` — true before any run)

| # | claim | anchor |
|---|---|---|
| C1 | Flag `use_canchor` threaded through all four `simprior(...)` call sites; cellcal/peakcal cascade unchanged (no call-site-exclusive bug like N0023) | `model.py:165,177–187` |
| C2 | Anchor block only after cellcal/peakcal; substrate `ev` byte-identical to parent before `out` | `model.py:237–255` |
| C3 | `t = top1.detach().clamp_min(0)`; spatial minmax → `[0,1]`; `z` = mean ROI-align mean over K boxes; `alpha = clamp(1/(z+1e-6), 0.25, 4.0)` | `model.py:258–266` |
| C4 | Return `self.out(ev) * alpha` — scales the **post-`out` residual**, bias included; flag off → `return self.out(ev)` parent line | `model.py:267–268` |
| C5 | Zero params, zero modules, zero RNG; stop-grad everywhere (`top1.detach()`, no grad through `alpha`/`bboxes`); temp untouched (pin path intact, `use_peakcal=true`) | `idea.md:4,91–100`; `config.toml:27–30` |
| C6 | Step-0 identity: `out` zero-init ⇒ `0·alpha = 0` | `model.py:223–225`, `idea.md:98` |
| C7 | Gradient to `out`: `∂res/∂θ_out = alpha · ∂out/∂θ_out` — live reweighting of the existing residual path, not a new path | `idea.md:74` |

### MEASURED (only what the files contain)

| read | field | bar (`idea.md:121`) | measured | verdict |
|---|---|---|---|---|
| **R1a** val `alpha` CV | alpha dump | ≥0.05 | **not in dump** | **unmeasurable** (cannot PASS; booked as mechanism falsifier if CV<0.05 once dumped) |
| **R1b** median α wipe-11 ≥1.3× median α KEEP-27 | alpha dump + per-image ids | ≥1.3× | **not in dump** | **unmeasurable** |
| **R1c** wipe residual L2 ≥ +20% vs parent | residual norms | ≥+20% | **not in dump** | **unmeasurable** |
| **R2** wipe pred/gt movement | `val_perimage` ratios | mean 0.241→≥0.40; ≥4/11 ≥0.50 | **not in dump** | **unmeasurable** |
| **R3** sparse / KEEP guard | `val_perimage` | sparse ≤5.45; no KEEP 0.8→<0.6 | **not in dump** | **unmeasurable** |
| **R4 val** | `result.json` / tb | ≤19.0431 | **22.2205** | **FAIL** (+3.1774) |
| **R4 dense / sparse** | `val_result` slices | <330.81 / ≤5.45 | **not in dump** | **unmeasurable** |
| **Futility** | `result.json` + rule | ep24 need ≤0.12 | need **0.3972** → HALT | **CONFIRMED HALT** |
| Config one-line delta | filtered config diff | exactly `use_canchor=true` | confirmed | **PASS** |
| Engagement pre-smoke (journal:64) | SMOKE25_OK (server, not local) | alpha active per-image, temp 0.07 pin, params unchanged | **lead-quoted / not re-verifiable from local files** | carried as external claim only |

**Booked bar summary:** R4-val FAIL · R4-dense/sparse/R1/R2/R3 **not in dump** · futility HALT CONFIRMED · construction C1–C7 CONFIRMED from code. **Verdict: REFUTED via futility-HALT** on the measurable val bar; mechanism engagement neither confirmed nor cleared locally.

---

## 2. Residual path vs parent (structural contrast)

- **Parent path (CONFIRMED structure):** `cond_map = cond + SimPrior(...)` then `decoder(cat[fine, cond_map])` — the residual is one of two decoder inputs (`model.py:175–188`).
- **Child path:** identical except the returned residual is multiplied by per-image `alpha ∈ [0.25,4]` (`model.py:267`). Cellcal gain, similarity substrate, sign of evidence, peakcal/temp-pin: byte-identical on every image (C1–C2) — **unlike N0023**, which exclusive-dispatched cellcal away.
- **What α can and cannot do (CONSTRUCTION):** it moves **only** the amplitude of the already-trained `out(ev)` term (64-ch residual added into `cond_map`). It cannot change spatial phase of the residual, cannot add a new tensor, cannot touch `fine` (the generator path). If the residual is net-suppressive and low-leverage (see §3), rescaling it is bounded by that leverage — at most ×4 up / ×0.25 down on a path the decoder barely uses for mass generation.
- **Child residual content after training: not in dump** (no weight dump, no per-image residual norms, no `w_c`/alpha series). Parent residual leverage numbers below come from the leverage probe quoted in `N0026_h0025/idea.md:8,47` and `journal/events.jsonl:65`, not from N0025 artifacts.

---

## 3. Leverage-probe context — was α-on-residual structurally under-powered?

**Probe premises (CONFIRMED as recorded file facts, not re-run here):** trained parent best.pth ablations — `fine=0` → count **−91.6%**; `cond_map=0` → count **+23.8%**; **remove SimPrior residual → count +29.9%** (i.e. residual is **net-suppressive**); `cond×2` → count −17.6%; channel sensitivity **fine mean 15.2 vs cond mean 1.6 (~10×)** (`N0026_h0025/idea.md:8,47`; `journal/events.jsonl:65`).

**Reading for H0024 (PLAUSIBLE, mechanism-consistent):**

1. **Wrong-lever by construction.** Count mass lives on `fine` (−91.6% when zeroed, 10× channel sens). The SimPrior residual is a **weak, net-suppressive** cond-path term (removing it *raises* count +29.9%). Scaling that term by α (even at the clamp max 4×) multiplies a suppressive, low-gain actuator — it cannot inject mass onto the generator path the decoder actually integrates. Six prior residual/evidence cards (H0016/17 quiet-null; H0018/22/23 wipe 0/11) already died on this path; H0024 tests **absolute magnitude on the same low-leverage path** (the magnitude fork of the two-card closure OR), not a new path.
2. **Under-powered even if α engaged.** R1 demanded only CV≥0.05 and 1.3× wipe/KEEP median separation — engagement ≠ count movement. With leverage ~1.6/15.2 ≈ 0.106 relative to fine, a 1.3–4× residual scale is effectively a **≤~0.1–0.4 fine-equivalent** perturbation before decoder co-adaptation; F3 in the card (“scale without shape” / consumption-interface fork) is the pre-registered name for this outcome (`idea.md:130`).
3. **Directional risk if α<1 on healthy images (F2):** starving an already-suppressive residual would *raise* counts on healthy images (probe: removing residual +29.9% ⇒ suppressing it further ⇒ fewer counts / higher AE) — the priced mid/sparse regression. **Cannot check without alpha/per-image dumps.**
4. **Honest status:** the level result (22.22, HALT outside draw band) is consistent with under-powered residual-scale, but **without alpha stats we do not distinguish F1 quiet-null (α≈1) from F2/F3 (α moved, counts didn’t)**. Any synthesis claiming a specific failure mode must first pull `val_result.json` / `val_perimage.json` / an alpha dump from the server (`run/latest/best.pth` still exists for a post-hoc dump per `idea.md:139`).

---

## 4. Causal chain — what is file-backed vs blocked

**Thesis (measurable half):** futility HALT at ep24 with best 22.2205 (need 0.3972/ep vs 0.12) and val bar miss +2.8774 vs live parent → **REFUTED as booked on R4-val alone**; trajectory sits at parity with sibling N0023 (+0.0574) and better than N0024 (−0.8413), i.e. CountAnchor did **not** deliver the booked −0.30 improvement.

| arrow | status | numbers |
|---|---|---|
| A. Flag wired, one-line config, construction C1–C7 | **CONFIRMED** | `model.py:165–187,255–268`; config diff `use_canchor=true` only |
| B. α engagement / wipe loading (R1) | **not in dump** | no alpha, no CV, no wipe/KEEP medians |
| C. Wipe/KEEP/sparse/dense count movement (R2/R3) | **not in dump** | no child per-image |
| D. Level outcome | **CONFIRMED** | 22.2205 @ep24, futility HALT, val +2.8774/+3.1774 |
| E. Residual is low-leverage suppressive; α structurally weak actuator | **PLAUSIBLE** (probe premises CONFIRMED elsewhere; not re-ablated on N0025) | fine sens 10×; remove residual +29.9% count; cond sens 1.6 |

Elimination available **without** alpha dumps: the card needed **both** R4-val AND R1; R4-val fails decisively outside the draw-contaminated band, so the card is refuted regardless of B. What cannot be closed locally is *which* failure mode (F1 vs F2 vs F3) and whether dense/sparse moved like N0023/N0024.

---

## 5. Single most load-bearing causal statement

**H0024 put the closure’s absolute-magnitude fork on the one path the leverage probe measures as weak and net-suppressive (fine sens 10×, residual removal +29.9% count) — a post-`out` scalar on a 64-ch suppressive residual cannot move integrated mass the way `fine` does — and the run confirms no booked gain: futility HALT at ep24 best 22.2205 (need 0.3972/ep > 0.12, +2.2174 above ceiling, +2.8774 vs live 19.3431), outside the card’s own (20.0, 21.6) draw-contaminated band; alpha engagement and slice/AE reads are not in the local dump and must not be claimed either way.**

*Noise note carried: same-seed draw spread {19.34, 20.70, 21.59} sd≈1.13; 22.22 is above the worst historical draw. The refutation rests on the pre-registered val bar + futility rule, not on a mechanism read we do not have.*

---

## Source citation per section

| § | claims | source |
|---|---|---|
| 0 | endpoint, futility, parent bars | NODE `result.json`, `info.json`, tb `…10646.0`; parent `/home/qkun/cac_backup/N0015_val_{result,perimage}.json`; rule `src/cac/engine/runner.py:151–191` |
| 1 C1–C7 | construction | NODE `model.py:165–187,223–225,237–268`, `idea.md`, `config.toml` (filtered diff) |
| 1 measured rows | R1–R4 / futility | bars `idea.md:121–124,132,139`; child values from `result.json`/tb or **not in dump** |
| 2 | residual path | NODE `model.py:175–188,255–268` |
| 3 | leverage premises | `../N0026_h0025/idea.md:8,47,118`, `journal/events.jsonl:65` (recorded probe; not re-run) |
| 4–5 | chain + headline | all of the above |

*Files written: this document only — `feedback/causal.md` (paired with `feedback/quant.md` same session). No `qual`/`diagnostic`/`synthesis`. No commit, no ssh, no ledger/`info.json` edits.*

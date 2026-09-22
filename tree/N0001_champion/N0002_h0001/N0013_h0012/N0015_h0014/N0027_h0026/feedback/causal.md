# Causal feedback — N0027_h0026 (H0026 `use_geme`, GemeCal geometry ME scale on SimPrior ev)

Causal agent, 2026-09-22. Inputs read this session (all local, no ssh, no commit): AGENTS.md §3/§5/§6/§11, STATE.md, NODE `model.py` / `idea.md` / `config.toml` / `info.json` / `result.json` / `val_result.json` / `test_result.json` / `val_perimage.json`, parent `../config.toml` + `/home/qkun/cac_backup/N0015_val_{result,perimage}.json`, draft `local/research/h0026_idea_DRAFT.md`, leverage/S-curve premises `local/research/reexamination_20260922.md` + `../N0026_h0025/idea.md:8,47` + `journal/events.jsonl:65`, sibling gold `../N0025_h0024/feedback/{quant,causal}.md` + `../N0026_h0025/feedback/{quant,causal}.md`, sibling endpoints `../N002{3,4,5,6}*/result.json`, draw stats `local/research/perimage_diagnostic.md:229`, futility rule `src/cac/engine/runner.py:151–191`.

Labels: **CONFIRMED** = file-backed this session · **CONSTRUCTION** = forward-logic identity from `idea.md`/`model.py` (true by arithmetic) · **PLAUSIBLE** = mechanism inference consistent with the files, not ablated.

**Artifact status (load-bearing):** under the node there is **no** `w_g` scalar dump, **no** per-image `f` field, **no** corr(log me, gt) readout, **no** local `run/` tb. Present: `result.json`, `val_result.json`, `test_result.json`, `val_perimage.json`, `info.json` (unsynced: `status: "proposed"`, `tested_hypotheses: []`), `idea.md`, `model.py`, `config.toml`. Every child-side mechanism read requiring `w_g`/`f` is reported **not in dump**.

---

## 0. Verified outcome (CONFIRMED)

- `result.json`: `futility_hit=true`, `n_epochs_done=24`/32, best **22.087779998779297 @ep24**, bar `futility_bar=19.0431`, `budget_hit=false`, elapsed_s 1316.5, shas `config c48cbec8d36450f4` / `model f8b0470a4542fd92`, overrides `{augment: true, futility_bar: 19.0431}`.
- `val_result.json` (ckpt_epoch 24): mae **22.0730057964058**, dense **445.0527** (n=17), mid **41.6080** (n=374), sparse **5.8755** (n=895), bias −15.36. `test_result.json`: mae **19.8044** (n=1190).
- Parent live 19.3431 (booked); dump recompute 19.3259, dense 330.8131, sparse 5.8136. **Δ = +2.7299 vs live / +3.0299 vs bar.** All three booked bars **FAIL** (19.0431 / 330.81 / 5.45).
- Futility arithmetic (threshold 0.12): ep16 WARN best **24.0491** → need 0.3129/ep > 0.12 → WARN; ep24 ceiling **20.0031**, need **0.3975/ep > 0.12** → **HALT**, best **+2.085 above ceiling**; outside the card's own draw-contaminated band (20.0, 21.6) by **+0.49** (`idea.md:176`). (Post-update convention from final best: 0.3806/ep — HALT under either ordering.)
- Config delta vs parent = exactly `use_geme = true` (one added line); `use_simprior/cellcal/peakcal` all true. `info.json` `status: "proposed"` is the runner-sync gap (rule 3: no hand-edit).

---

## 1. Construction claims vs measured reads

### CONSTRUCTION (from `model.py` + `idea.md` — true before any run)

| # | claim | anchor |
|---|---|---|
| C1 | Flag `use_geme` threaded through **all four** `simprior(...)` call sites; cellcal/peakcal cascade unchanged (no N0023 exclusive-dispatch) | `model.py:167,179–194` |
| C2 | `GemeCal`: `f = exp(w_g·(log(me)−log 32)).clamp(0.25,4)`; `me = mean_K(S²/area)` from annotation boxes, clamped [1,1e4]; stop-grad geometry (no learned input path) | `model.py:295–310` |
| C3 | Multiply `ev ← ev · f` **after** cellcal/peakcal gains, **before** `return self.out(ev)` — substrate/gain forms byte-identical to parent before `out` | `model.py:261–263` vs parent `:243–252` |
| C4 | `w_g` zero-init ⇒ f=1 exact ⇒ step-0 identity; additionally zero-init `out` ⇒ residual 0 (double identity) | `model.py:302,309`; `idea.md:135` |
| C5 | +1 param only; `torch.zeros(())` consumes **zero RNG**; GemeCal constructed after PeakCal (append-only, rule 13); flag non-module | `model.py:167,346–347`; `idea.md:136` |
| C6 | Temp-pin path intact (`use_peakcal=true` → `temp.requires_grad_(False)`); no shared projection with simprior/qproj/kproj (H0006/7 family clean) | `model.py:349–351` |
| C7 | Gradient nonzero at init: `∂L/∂w_g = ∂L/∂ev · ev · log(me/32)` — generically nonzero whenever me≠32 (not a structurally dead gate, H0009 lesson) | `idea.md:105` |
| C8 | Config one-line delta, pluggable by single switch `use_geme` | `config.toml:30` filtered diff |

### MEASURED (only what the local files contain)

| read | bar (`idea.md:156–159`) | measured | verdict |
|---|---|---|---|
| **R0** corr(log me, log(1+gt)) on val | ≥0.20 | **not in dump** | **unmeasurable** (dataset-level disjunct undecidable) |
| **R1** CV of per-image `f` (+ `w_g`, clamp histogram) | ≥0.05 | **not in dump** | **unmeasurable** (F1 detector undecidable) |
| **R2** mean f\|gt≥300 ≤0.90× mean f\|gt<50 | ≤0.90× | **not in dump** | **unmeasurable** (direction commit undecidable) |
| **R3 val** | ≤19.0431 | **22.0730** | **FAIL** (+3.0299) |
| **R3 dense** | <330.81 | **445.0527** | **FAIL** (+114.24) |
| **R3 sparse** | ≤5.45 | **5.8755** | **FAIL** (+0.4255 vs bar; +0.062 vs parent) |
| **Futility** ep16 / ep24 | WARN / need ≤0.12 | WARN @24.0491; need **0.3975** → HALT | **CONFIRMED HALT** |
| Config one-line delta | exactly `use_geme=true` | confirmed | **PASS** |
| Launch gates (smoke, shas) | `result.json` shas + journal Lead-record | shas present in `result.json`; smoke Lead-recorded | shas **CONFIRMED** locally; smoke not re-verifiable from local run artifacts |

**Booked bar summary:** R3 val/dense/sparse **all FAIL** · R0/R1/R2 **not in dump** · futility HALT CONFIRMED · construction C1–C8 CONFIRMED from code. **Verdict: REFUTED via futility-HALT** on the measurable triple bar; mechanism engagement neither confirmed nor cleared locally.

---

## 2. Residual path vs parent (structural contrast)

- **Parent path (CONFIRMED structure):** `cond_map = cond + SimPrior(...)` then `decoder(cat[fine, cond_map])` — the SimPrior residual is one of two decoder inputs (`model.py:184–195`).
- **Child path:** identical except that inside `SimPrior.forward` the evidence tensor `ev` is multiplied by per-image `f ∈ [0.25, 4]` before the zero-init `out` projection (`model.py:261–263`). Cellcal gain, similarity substrate, sign handling, peakcal, temp-pin: byte-identical on every image (C1–C3).
- **What `f` can and cannot do (CONSTRUCTION):** it moves **only** the amplitude of the pre-`out` evidence — a single positive scalar broadcast over both evidence channels. It cannot change spatial phase, cannot add a tensor, cannot touch `fine` (the generator path), cannot bypass the zero-init `out`. Bounded by the residual's own leverage: probe removes residual → count **+29.9%** (net-suppressive), cond sens ~10× below fine (`N0026_h0025/idea.md:8,47`, `journal:65` — recorded premises, not re-ablated here).
- **Child residual content / `w_g` after training: not in dump** (no weight scalar pulled, no per-image `f`, no residual norms).

---

## 3. Leverage-probe context — was geometry-on-ev structurally under-powered?

**Probe premises (CONFIRMED as recorded file facts, not re-run here):** `fine=0` → count **−91.6%**; `cond_map=0` → **+23.8%**; remove SimPrior residual → **+29.9%**; `cond×2` → −17.6%; channel sens fine **15.2** vs cond **1.6** (~10×).

**Reading for H0026 (PLAUSIBLE, mechanism-consistent):**

1. **Right covariate, still the wrong-lever path.** Exemplar-box `me` is a genuine order-of-magnitude covariate (CACViT-published) and varies by construction across the S-curve — unlike H0024's flat response-z. But the actuator still multiplies the **weak, net-suppressive** residual: even a correctly-signed capacity loading reweights a term the decoder barely integrates for mass generation. The trilogy's three loci (post-out α, decoder γ, pre-out f) all scale or condition low-leverage/suppressive-adjacent paths relative to `fine`'s 10× sensitivity — explaining a shared +2.7…+3.9 envelope despite different sources and operators.
2. **R2's committed direction was attenuate-on-dense (f ≤0.90× on gt≥300).** If `f` engaged with the wrong sign (amplify suppression on dense), dense would worsen — observed dense **445.05 (+114.2)** is consistent with either wrong-direction load or engaged-but-ineffective plus draw; **cannot check without the f dump.**
3. **F1 quiet-null remains the priced dominant risk** (`idea.md:163`): MSE on a 69%-sparse training set may simply never reward geometry scaling, leaving `w_g ≈ 0` and the child as parent + inert scalar. Level +2.73 sits above every observed draw (+0.48 over worst), which strains a pure quiet-null reading but does not exclude it (single draw, sd≈1.13).
4. **Honest status:** without `w_g`/`f`/corr we do not distinguish F1 (quiet-null), F2 (dead covariate, corr<0.20), or F3 (wrong-direction load). Any synthesis claiming a specific failure mode must first pull those reads from `best.pth` (`idea.md:157–158` registers them as eval-time reads).

---

## 4. Causal chain — what is file-backed vs blocked

**Thesis (measurable half):** futility HALT at ep24 with best 22.0878 (need 0.3975/ep vs 0.12, +2.085 above ceiling 20.0031, outside the card's own (20.0, 21.6) band) and triple-bar miss (val +2.73 / dense +114.2 / sparse over 5.45) → **REFUTED as booked on R3 alone**; corridor position **+2.73, best of N0023–N0027** but still 0/3 bars.

| arrow | status | numbers |
|---|---|---|
| A. Flag wired, one-line config, construction C1–C8 | **CONFIRMED** | `model.py:167,179–194,261–263,295–310,346–347`; config diff `use_geme=true` only |
| B. `w_g`/`f` engagement, corr, loading direction (R0–R2) | **not in dump** | no w_g, no f histogram, no corr |
| C. Level + slices | **CONFIRMED** | val 22.0730, dense 445.05, mid 41.61, sparse 5.876, test 19.804 |
| D. Futility HALT | **CONFIRMED** | `result.json`; ep16 WARN 24.0491; need 0.3975/ep |
| E. Residual is low-leverage suppressive; any pre/post-out amplitude structurally weak | **PLAUSIBLE** (probe premises CONFIRMED elsewhere; not re-ablated on N0027) | fine sens 10×; remove residual +29.9% count |

Elimination available **without** the dump: the card needed **R3 ∧ R0–R2**; R3 fails 0/3 decisively outside the draw band, so the card is refuted regardless of B. What cannot be closed locally is *which* failure mode (F1 vs F2 vs F3) fired.

---

## 5. Single most load-bearing causal statement

**H0026 put the trilogy's last untested amplitude cell (geometry ME, published CACViT covariate, learnable sign) on the same measured weak-suppressive SimPrior residual the lineage has now bracketed three times — a pre-`out` scalar on evidence that removes to +29.9% count cannot move integrated mass the way `fine` does — and the run confirms no booked gain: futility HALT ep24 best 22.0878 (need 0.3975/ep ≫ 0.12, +2.085 above ceiling, +2.73 vs live 19.3431), outside the card's own draw-contaminated band, all three bars failed (dense +114.2, sparse over 5.45); `w_g`/`f`/corr engagement reads are not in the local dump and must not be claimed either way.**

*Noise note carried: same-seed draws {19.34, 20.70, 21.59} sd≈1.13; 22.07 is above the worst historical draw by +0.48. The refutation rests on the pre-registered triple bar + futility rule, not on a mechanism read we do not have.*

---

## Source citation per section

| § | claims | source |
|---|---|---|
| 0 | endpoint, slices, futility, shas | NODE `result.json`, `val_result.json`, `test_result.json`; rule `src/cac/engine/runner.py:151–191`; parent `/home/qkun/cac_backup/N0015_val_{result,perimage}.json` |
| 1 C1–C8 | construction | NODE `model.py:167,179–194,261–263,295–310,346–351`, `idea.md`, `config.toml` (filtered diff) |
| 1 measured rows | R0–R3 / futility | bars `idea.md:156–159,176`; child values from `result.json`/`val_result.json` or **not in dump** |
| 2 | residual path | NODE `model.py:184–195,234–263` vs parent structure |
| 3 | leverage premises | `../N0026_h0025/idea.md:8,47`, `journal/events.jsonl:65`, `local/research/reexamination_20260922.md` (recorded; not re-run) |
| 4–5 | chain + headline | all of the above; corridor `../N002{3,4,5,6}*/result.json` |

*Files written: this document only — `feedback/causal.md`. No `qual`/`diagnostic`/`synthesis`. No commit, no ssh, no ledger/`info.json` edits.*

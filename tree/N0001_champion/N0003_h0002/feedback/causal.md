# Causal feedback — N0003_h0002 (H0002, learnable signed global-count coupling)

**Verdict of record:** REFUTED at the registered bar (`result.json:5-6`: best_mae
**23.20878791809082** @ep32 vs bar **≤22.99**, live parent N0001 **23.293153762817383**
@ep26, `tree/N0001_champion/result.json:5-6`). This file does **not** re-litigate the
verdict; it attributes *why*, from measured parameters and curves.

**Attribution: partially-learned with a small effect.** The coupling moved decisively
(null box fails), it produced a consistent late-epoch paired gain, but the gain is
~0.08–0.13 MAE — 28–43% of the 0.30 bar — and the learned global offset is
arithmetically negligible in count units. It is not "mechanism-not-learned" and not
"GCA already locally optimal."

---

## (1) MEASURED parameters — null criterion does NOT hold

Probe: `/tmp/probe_gca.py` (16 lines, `<30`), run on server CPU with
`/data/miniconda/envs/cac/bin/python`, reading `ckpt["model"]` from
`/data/cac/tree/N0001_champion/N0003_h0002/run/latest/best.pth`. The stored weights
are the **EMA shadow** that produced the val number (`best_checkpoint.py:21-29` via
`calls/ema.py:46-63`, per `idea.md:117-126`).

| key | init (`model.py:179-181`) | trained EMA | Δ | idea.md:123 null cond | holds? |
|---|---|---|---|---|---|
| `gca.s_pos` | 0.02 | **0.00839800** | −0.011602 (−58.0%) | `\|s−0.02\|/0.02 < 0.10` → measured **0.5801** | **FAIL** |
| `gca.s_lin` | 0.0 | **0.02147696** | +0.021477 | `\|s_lin\| < 1e-3` | **FAIL** |
| `gca.b_cal` | 0.0 | **−0.11962026** | −0.119620 | `\|b_cal\| < 0.5` | PASS |

Trunk norms (last layer `gca.gca[-1]` = `gca.gca.2`):

| tensor | child | parent (same probe) |
|---|---|---|
| `gca.gca.2.weight` norm | **0.151678** (init 0, `model.py:176`) | **0.233494** |
| `gca.gca.2.bias` norm | **0.072668** (init 0, `model.py:177`) | **0.085947** |
| `gca.gca.0.weight` norm | 5.862320 | 7.023124 |
| `gca.gca.0.bias` norm | 0.307195 | 0.347718 |

**Null verdict: does NOT hold** — 2 of 3 sub-conditions fail; `s_pos` moved 58% of
its init, `s_lin` moved off exactly zero. The coupling was actively optimized.
The parent measurement also falsifies the card's premise that the 0.02 path is
"near-detached": parent `gca.gca.2` weight norm is 0.233 (not zero), so the parent's
global path already carried a learned, image-dependent `z` — the fixed 0.02 only
scaled it.

---

## (2) Interpretation — moved, and helped slightly; the achievable correction is tiny

- **Moved.** Null box fails (above); `gca.gca.2` moved off its zero init in both
  parent and child.
- **Direction is interpretable.** `s_pos` was **halved** (0.02→0.0084): the optimizer
  *reduced* reliance on the non-negative softplus branch, and routed the signed
  degree of freedom through `s_lin` (+0.0215) with a negative offset
  `b_cal = −0.12` counts. Large-`z` net gain ≈ 0.0084+0.0215 ≈ 0.030; negative-`z`
  contribution becomes subtractive (`model.py:190-191`). This is a genuine signed
  recoupling, exactly the mechanism the idea targeted (`idea.md:33-40`).
- **But the count-unit magnitude is negligible.** `b_cal` is the only quantity
  directly in count units: **−0.12 counts**, versus the idea's own arithmetic that the
  0.30 bar = **386 counts** (`idea.md:46`). The learned global offset is ~0.03% of the
  required correction. A head that reads only `GAP(fine)+e_mean` (`model.py:184-185`)
  cannot localize the few heavy-tail images the card wanted to fix
  (`idea.md` Open Risk 1, line 128).
- Therefore the mechanism was **tested and learned**, and it did move MAE the right
  way (see §3), but its effect ceiling is small. This is "partially-learned with small
  effect," not "no gain" (there is a consistent gain) and not "already locally
  optimal" (params moved 58%).

---

## (3) Margin 0.0844 — small **real** effect, not noise, but sub-bar

Same-seed paired precision is ~0.02 (`AGENTS.md §6`; `idea.md:115`). Per-epoch
`val/mae` from tensorboard (`tb/t`, child vs parent) for the tail:

| ep | child | parent | child−parent |
|---|---|---|---|
| 23 | 23.32327 | 23.38791 | −0.06464 |
| 24 | 23.25979 | 23.33070 | −0.07091 |
| 25 | 23.21677 | 23.32571 | −0.10894 |
| 26 | 23.22702 | 23.29315 | −0.06613 |
| 27 | 23.22401 | 23.31939 | −0.09538 |
| 28 | 23.21566 | 23.30751 | −0.09185 |
| 29 | 23.22187 | 23.30696 | −0.08509 |
| 30 | 23.21650 | 23.32215 | −0.10565 |
| 31 | 23.21338 | 23.33410 | −0.12072 |
| 32 | 23.20879 | 23.33643 | −0.12764 |

10/10 epochs same direction; mean gap **−0.0937**, magnitudes 0.0646–0.1276, all above
the ~0.02 precision. Within-run plateau spreads: child ep25–32 = 0.01823; parent
ep26–32 = 0.04328. A consistent 10-epoch same-direction gap exceeding both the stated
precision and the plateau spread makes the pure-noise explanation implausible — the
effect reads as **small but real**.

It is nonetheless **below bar**: best-vs-best margin 23.29315−23.20879 = **0.0844**
(= 28% of the 0.30 bar); matched-epoch-32 margin = 0.12764 (= 43%). Real ≠ sufficient;
the registered bar is 0.30 and is not met.

---

## (4) Confounder: best at ep32 is a flat plateau, not a descending curve

`best_epoch = 32` is the last epoch (`result.json:5-6`; log
`chain_run.log:557-558` `"best_mae": 23.20878791809082, "best_epoch": 32`; :550
`max_epochs=32 reached`; `budget_hit: false`, `result.json:9`). But the curve is flat,
not still steeply descending: child ep25→ep32 = 23.21677→23.20879 = **−0.008 over 7
epochs** (~0.0011/epoch). Reaching 22.99 from 23.20879 (Δ = 0.2188) at that rate needs
~190 more epochs. The parent is *worse* by this metric: best ep26 = 23.29315 then
**rises** to 23.33643 at ep32.

So "undertrained rather than wrong" does not survive: (a) the flat plateau says extra
epochs cannot close 0.22; (b) the protocol is frozen at 32ep/1800s, seed 20260830
forever, with `repro_run --set seed=` forbidden and no outcome shopping
(`AGENTS.md §6, §7, rule 1`). Under the frozen protocol the verdict is refuted
regardless. **Residual confound:** the 3-param change also perturbs the shared-weight
trajectory — the child is *worse* than the parent through ~ep21 and crosses only at
~ep22 — so part of the late gain could be a generic landscape perturbation rather than
signed calibration per se. The harness cannot separate these (same-seed, but the loss
diverges after step 1, so shared weights are not sample-paired); this is a caveat on
the *size* attribution, not a rescue of the bar.

---

## (5) Follow-up: diagnostic under a NEW falsifier; no silent retry

Per `AGENTS.md §11` / rule 11, H0002 is not retried silently. Any re-encoding of the
signed global-count coupling must be booked as `contradicts`-evidence on H0002 **and
carry a new falsifier**. Given the measured `b_cal = −0.12` counts vs a 386-count bar,
the honest next move is a **diagnostic** (no training, no protocol change) that tests
the card's own Open Risk 1:

- **Diagnostic D (new falsifier, numeric):** on the val set, compute per-image signed
  count error `e_i = pred_i − gt_i` and the oracle uniform additive correction
  `Σ_i min_δ |pred_i + δ − gt_i|`. **IF** the oracle per-image constant shift lowers val
  MAE by **< 0.30**, **THEN** no global-count-coupling variant can pass the 0.30 bar and
  the global-calibration family should be retired (evidence on H0002); **DISPROVED IF**
  it lowers MAE by **≥ 0.30** (then exploitable structure exists and a fresh card is
  justified).
- **Only if** the oracle clears 0.30 may a **new** card be booked (e.g. a
  per-image/exemplar-conditioned signed calibration on the decoder side — a live,
  untouched direction per `AGENTS.md §11`) with its own pre-registered falsifier. H0002
  stays refuted either way.

---

**Evidence base:** probe values above; `run/latest/best.pth`; tb `val/mae`/`val/rmse`
(child + parent); `chain_run.log:550,557-558`; `result.json:5-6,9`;
`tree/N0001_champion/result.json:5-6`; `idea.md:33-40,46,115,117-126,128`;
`model.py:176-177,179-181,184-191`; `config.toml:24`.

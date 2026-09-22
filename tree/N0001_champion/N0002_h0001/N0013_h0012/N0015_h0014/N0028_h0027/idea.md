# H0027 — FineScale: zero-init geometry capacity scale on the decoder-bound fine channels (SOLO card; similarity-side H0026 already landed its verdict on N0027)

- Parent: `tree/N0001_champion/N0002_h0001/N0013_h0012/N0015_h0014` (LIVE parent N0015_h0014, val MAE 19.3431 @ep32 v2 protocol, seed 20260830).
- SOLO card: exactly one mechanism switch `use_finescale` (default false), shipped as a **single-line config delta** (`use_finescale=true`; every other key byte-identical). When the flag is on, `CountingHead.forward` computes a per-image geometry capacity factor `f = clamp((me/32)^w_f, 0.25, 4)` from the annotation exemplar boxes (`me = mean_K(S²/(w_k·h_k))`, CACViT order-of-magnitude capacity, stop-grad geometry) and multiplies the **fine channels of the decoder input only** (`dec_fine = fine * f` immediately before the unchanged `torch.cat([dec_fine, cond_map], 1)`). `w_f` is a **single zero-initialized learnable scalar** inside a new `FineScale` module constructed AFTER every parent module. cond_map, SimPrior, cellcal/peakcal gain forms, the Condenser, GCA, and every post-decoder line: byte-identical. **+1 parameter, +0 effective RNG draws** (`torch.zeros(())` consumes no RNG). Flag off → new branch never taken, executed lines byte-identical to parent.
- Regime: frozen backbone, head-only training. Decoder `in_ch` untouched (still `D + cond_dim` = 192); optimizer/loss/schedule invariant (AdamW, MSE+w_cnt·L1, cosine, AMP).
- Protocol: v2 (augment=true, EMA eval, seeded loaders), seed FIXED 20260830, 32ep/1800s. Same-seed paired contrast vs live parent N0015 only.
- **Triple conjunctive bars** (§6 semantics, bound to live parent 19.3431 under v2): THEN final EMA val MAE ≤ **19.0431** (≥0.30 below 19.3431) AND dense-tail gt>500 **< 330.81** AND sparse gt<50 **≤ 5.45**. Missing any one bar refutes the card. Launch carries `--set futility_bar=19.0431` (ep16 WARN / ep24 HALT, rule 16).
- **Engagement reads (pre-registered):** R1 `|w_f| ≥ 0.01` at best.pth **or** val CV of f ≥ 0.05 (collapse of both ⇒ quiet-null refute); R0 Pearson corr(log me, log(1+gt)) on val ≥ **0.20** (dataset covariate gate).
- **Why THIS card:** H0026 GemeCal (similarity-evidence site) landed its verdict on N0027 (REFUTED, futility 22.088) — the `me` covariate survives at a different site/operator/tensor. Offline probes show the only oracle members ≥2 MAE are gt-binned **scale (−2.27)** and gt-binned **affine (−7.47)** — per-image density-amplitude forms; `fine` is the dominant generator (leverage probe: fine=0 → −91.6% vs the weak residual/cond paths). This card puts the published order-of-magnitude geometry prior on the mainline feature amplitude at train time — the deployable-form count-conditional scale STATE.md:195 names — without touching SimPrior, cond_map, or anything post-decoder.
- **Honest expectation:** expected-negative (HANDOFF discipline: three-card +2.7…+3.9 arc, sd≈1.13 same-seed floor; draft §5 NO-GO overridden by booking directive). Mechanism reads weighted where they conflict with level bars.

## §0 — Bookable one-liner (verbatim booked text; runner-parse line)

1. **H0027** — IF a zero-init geometry capacity scale on the fine feature branch (`use_finescale`) that multiplies fine by f = clamp((me/32)^w_f, 0.25, 4) before the unchanged decoder concatenation, where me = mean_K(S²/(w_k·h_k)) is the exemplar-box order-of-magnitude capacity and w_f is a single zero-initialized learnable scalar, while the SimPrior readout, the cellcal and peakcal gain forms, the Condenser, the decoder weights, the GCA bias path, and every other parent line stay byte-identical, IN the live N0015 head on FSC147 v2 at seed 20260830, THEN final EMA val MAE drops at least 0.30 below 19.3431 (at or below 19.0431) with dense-tail gt>500 below 330.81 and sparse gt<50 at or below 5.45, BECAUSE the measured S-curve under-count is an absolute per-image amplitude error whose oracle gt-binned scale headroom is −2.27 MAE while the dominant generator branch fine carries −91.6 percent leverage versus the weak residual path, so exemplar-box geometry as a count-capacity covariate scaled onto fine at train time lets the loss learn the per-image density amplitude the decoder otherwise compresses toward the training mean. DISPROVED IF final EMA val MAE is not at least 0.30 below 19.3431 (above 19.0431), or dense-tail gt>500 is not below 330.81, or sparse gt<50 exceeds 5.45, or the engagement reads fail: abs(w_f) below 0.01 at best.pth and val coefficient of variation of f below 0.05, or Pearson corr of log me with log(1+gt) on val below 0.20.

## §1 — Exact mechanism (delta from parent N0015 model.py)

| # | Site | Change |
|---|---|---|
| 1 | imports | `import math` (for `math.log(32.0)`). |
| 2 | `CountingHead.__init__` (beside `use_peakcal`) | `self.use_finescale = _get(cfg, "use_finescale", False)` — non-module flag, no RNG. |
| 3 | `CountingHead.forward` pre-decoder | `dec_fine = fine`; if flag and `self.finescale` exists → `fscale = self.finescale(bboxes_in, self.S)`; `dec_fine = fine * fscale.view(-1,1,1,1)`; `dens = self.decoder(torch.cat([dec_fine, cond_map], 1))`. **Single site** — the fine channels of the decoder input. cond_map / SimPrior / fmap→Condenser / GCA all read the unscaled `fine`. |
| 4 | New class `FineScale` after `PeakCal` | `w_f = nn.Parameter(torch.zeros(()))`; `f = clamp((me/32)^w_f, 0.25, 4)` with `me = mean_K(S²/area).clamp(1,1e4)` from `bboxes_in`. |
| 5 | `Counter.__init__` after PeakCal attach (before temp-pin) | `if self.head.use_finescale: self.head.finescale = FineScale()` — AFTER every parent module (append-only RNG, rule 13); zero-init draws no RNG. |
| 6 | `config.toml` | one new line `use_finescale = true`. |

Untouched: `Backbone`, `FineFuser`, `ExemplarEncoder`, `Condenser`, `DensityDecoder`, `GCA`, `SimPrior` (entire class + all four call sites), cellcal/peakcal math, qproj/kproj/temp, temp-pin, post-decoder GCA bias add.

## §2 — Step-0 identity, params, guards

- **Flag off:** `dec_fine is fine` → executed cat/decoder lines are the parent's byte-for-byte.
- **Flag on at init:** `w_f=0` ⇒ `(me/32)^0 = 1` ⇒ `clamp(1, 0.25, 4) = 1` exact ⇒ `fine * 1.0 == fine` (IEEE exact) ⇒ parent-identical forward.
- **Params:** +1 (`w_f`). One non-module flag `use_finescale`. Constructed after PeakCal attach with zero RNG draws → parent init stream bit-identical (rule 13).
- **Guards:** `area.clamp_min(1)`, `me.clamp(1,1e4)` before log; `f.clamp(0.25, 4.0)` finite for any finite `w_f`; no new normalization, no softmax/temp, no sign test, no mask (train/eval agree).
- **Smoke (Coding Agent):** flag-off max-abs-diff vs parent == 0; flag-on-at-init == 0; param delta == +1; temp still 0.07 pinned.

## §3 — Reads and failure modes

- **R0 — covariate carries count (dataset gate):** Pearson corr(log me, log(1+gt)) on val ≥ **0.20**. Below ⇒ geometry is a dead covariate — honest mechanism refutation regardless of bars.
- **R1 — scale engages:** `|w_f| ≥ 0.01` at best.pth **or** val CV of per-image f ≥ **0.05**; report `w_f` trajectory and f histogram (clamp pile-up at 0.25/4.0 = F4). Both below ⇒ quiet-null REFUTE before bars are consulted.
- **R2 — bars (conjunctive):** final EMA val MAE ≤ **19.0431** AND dense-tail gt>500 **< 330.81** AND sparse gt<50 **≤ 5.45** (same-seed v2 pair vs live N0015 19.3431 / 330.81 / 5.814). Launch: `--set futility_bar=19.0431`.
- **Failure modes:** F1 quiet-null (w_f→0, expected mode); F2 R0 corr fail (dead covariate); F3 sparse blowup past 5.45 while dense improves; F4 clamp saturation (read histogram); F5 engage-but-bars-fail on sd≈1.13 floor (refute as booked, note mechanism-read status); F6 futility HALT ep24 — verdict = futility refutation + full eval_test + R0/R1 on best.pth; F7 never attribute against N0026/N0027 (this child is a direct child of N0015).

## §4 — Disambiguation (§11 bans / prior hypotheses)

| Constraint / prior hyp | Why H0027 does not re-encode it |
|---|---|
| H0026 GemeCal (REFUTED, N0027) | Site: decoder-bound `fine` feature tensor pre-cat vs post-gain SimPrior `ev` pre-out. Tensor: 128-ch generator features vs 2-ch suppressive residual. Path: dominant (−91.6% leverage) vs measured-suppressive (+29.9%). Falsifier: `|w_f|`/CV + corr≥0.20 vs H0026's CV + f-load direction. Same covariate (`me`), different operator target — new falsifier registered (rule 11). |
| H0025 CapFilm (REFUTED, N0026) | Site: fine channels at decoder INPUT vs decoder hidden post-GN. Operator: per-image scalar power law vs per-channel FiLM γ,β affine. Conditioner: `me` alone vs GAP(fine)+logME. No γ/β, no affine, no decoder-weight edit. |
| H0024 CountAnchor (REFUTED, N0025) | Response-mass fixed `1/z`, post-decoder residual. Here: annotation geometry, learnable exponent, pre-decoder feature multiply. |
| §11 post-decoder additive | Nothing after `decoder(...)`; the multiply is on the decoder's input channels, pre-cat. |
| §11 gain-domain masks (H0018/H0022) | No sign test, no `where`, no cell suppression; a uniform positive per-image scalar on feature channels. |
| §11 pre-condenser exemplar gating | Reads `bboxes` geometry only; `e`, ExemplarEncoder, Condenser untouched. cond_map path untouched (fmap taken from unscaled fine). |
| §11 transfer-by-learned-fusion | `me` = annotation box areas already in `bboxes_in`; no external evidence, no fusion module. |
| H0010/H0015/H0002 count/regime/global scalars | No predicted count, no regime split, no global constant — f varies per image from geometry; CV/OR engagement gate unsatisfiable by a constant. |
| H0006/H0007 second grad path / shared projections | `FineScale` owns its single Parameter; no alias to simprior/qproj/kproj/decoder weights. |
| Temp-pin hygiene / trainable temp | Temp line byte-identical, still pinned 0.07; card owns no temperature. |
| GCA / load-bearing readout | GCA reads the unscaled `fine` (parent path); frozen hs(2,3) + cross-attn Condenser byte-identical. |

## Verification record

- novelty gate: **PASS, exit 0** — top sim H0026=0.638, H0025=0.511, H0024=0.443 (all < 0.82); no structural twin. Raw: `local/research/h0028_finescale_novelty.json`
- format gate: **PASS** — `discovery validate` on the booked text: 0 errors (length warning only, same class as prior booked cards).
- booking: **DONE** — `python3 scripts/discovery.py hypo N0015_h0014 --new "<text>" --solo` → ledger `H0027`, node `N0028_h0027`.
- coding: parent + deltas above; static verify (config diff 1 line, ast.parse, zeros init, no post-decoder, no SimPrior edit, no sign mask).
- launch (Lead): `run_node N0028_h0027 --set augment=true --set futility_bar=19.0431` — NOT launched by this card's authoring pass.

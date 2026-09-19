# N0008_h0014 — Quantitative feedback (H0014 SOLO, SAFECount-style dual-normalized similarity + query enhancement)

Scope: same-seed paired contrast vs the CANONICAL parent N0001_champion under the
canonical protocol (noaug + EMA + seeded loaders + determinism, seed 20260830).
Child switch: `use_safe_enhance=true` (paper 2201.08959). Parent config has no
`use_safe_enhance` key (off path). This note is dissection + fragility analysis only;
it does not re-verdict the ledger entry (`supports w=1.0`, conf 0.600).

Sources read:
- Server canonical results: `/data/cac/tree/N0001_champion/result.json`,
  `/data/cac/tree/N0001_champion/N0008_h0014/result.json`
- Server configs: `/data/cac/tree/N0001_champion/config.toml`,
  `/data/cac/tree/N0001_champion/N0008_h0014/config.toml` (both `seed=20260830`,
  `deterministic=true`, `augment=false`)
- Server infos: `/data/cac/tree/N0001_champion/info.json` (canonical
  `best_metric=23.293153762817383`), `/data/cac/tree/N0001_champion/N0008_h0014/info.json`
- Per-epoch curves via EventAccumulator (server python) on:
  - parent `/data/cac/tree/N0001_champion/run/latest/tb/t/events.out.tfevents.1789812726.*` (tag `val/mae`, n=32)
  - child `/data/cac/tree/N0001_champion/N0008_h0014/run/latest/tb/t/events.out.tfevents.1789817882.*` (tag `val/mae`, n=32)
- Checkpoints: `/data/cac/.../run/latest/best.pth` (`torch.load` keys `epoch`, `best_mae`)
- Local `tree/N0001_champion/result.json` + `info.json` are STALE v1 artifacts
  (`best_mae=21.458858489990234`); all parent numbers below are the SERVER canonical run.
- Log note: no `exec.log` exists for either canonical run. Server parent
  `run/latest/run_node.log` (mtime Sep-19 14:05) is a stale v1 tail (ends with
  `val/mae 21.459`, `best_mae 21.4588...`); server child `run/latest/` contains only
  `best.pth` + `tb/`. TB scalars + `result.json` + `best.pth` headers are therefore
  the primary sources.

## 1. Final table + checksum / checkpoint cross-check

| Item | Parent N0001_champion (canonical, server) | Child N0008_h0014 (server) |
|---|---|---|
| `best_mae` (result.json) | 23.293153762817383 (@ep26) | 22.96969223022461 (@ep32) |
| `best_epoch` (result.json) | 26 | 32 |
| Last-epoch `val/mae` (TB step 7295) | 23.336433 (@ep32) | 22.969692 (@ep32; == best) |
| `epochs` / `n_epochs_done` | 32 / 32 | 32 / 32 |
| `budget_hit` | false | false |
| `elapsed_s` (result.json) | 1699.4 (`elapsed` 1694.1166) | 1787.2 (`elapsed` 1781.8784) |
| `config_sha256` | 679f37c26bc72103 | 7748200bd523e08f (differs: `use_safe_enhance=true` line) |
| `model_sha256` | eaf2a8b93c280cba | 650a7ba53b8b24be (differs: SafeEnhance block) |
| `best.pth` header | `epoch=26, best_mae=23.293153762817383` — MATCHES result.json | `epoch=32, best_mae=22.96969223022461` — MATCHES result.json |
| `best.pth` size | 125,383,139 B | 125,517,947 B (Δ +134,808 B ≈ +0.11%) |
| TB `val/mae` best | min 23.293154 @ep26 — MATCHES result.json to 6dp | min 22.969692 @ep32 — MATCHES result.json to 6dp |

Checkpoint cross-check: PASS for both nodes — `best.pth` (`epoch`, `best_mae`) agrees
exactly with `result.json` (`best_epoch`, `best_mae`), and the TB-curve minima agree
to 1e-6. Child best == last epoch (ep32), i.e. the curve was still improving at the
32-epoch wall; the "best" is truncated, not a converged interior minimum (see §3).

Size cross-check: +134,808 B / 4 B·param⁻¹ = 33,702 fp32-equivalents, consistent with
the +33,152-param SafeEnhance block (§4) plus pickle overhead.

## 2. Epoch-by-epoch delta vs canonical parent (child minus parent, ep1..32)

TB `val/mae` values are quoted at 6dp as stored; deltas computed from full-precision
stored values (rounded here to 4dp). Step index increments 228/epoch for both runs
(228 steps/epoch × 32 = 7296; same loader length — protocol match).

| ep | child | parent | Δ (C−P) | ep | child | parent | Δ (C−P) |
|---|---|---|---|---|---|---|---|
| 01 | 148.3846 | 146.9976 | +1.3870 | 17 | 23.3070 | 23.8016 | −0.4946 |
| 02 | 69.4386 | 69.0238 | +0.4148 | 18 | 23.1381 | 23.7184 | −0.5803 |
| 03 | 49.4351 | 49.4376 | −0.0024 | 19 | 23.0088 | 23.6163 | −0.6075 |
| 04 | 39.9981 | 39.5353 | +0.4628 | 20 | 23.0777 | 23.5276 | −0.4500 |
| 05 | 34.2627 | 34.1235 | +0.1392 | 21 | 23.1080 | 23.5090 | −0.4010 |
| 06 | 30.7833 | 30.6235 | +0.1598 | 22 | 23.1198 | 23.4600 | −0.3402 |
| 07 | 28.6531 | 28.5269 | +0.1262 | 23 | 23.0712 | 23.3879 | −0.3167 |
| 08 | 27.3236 | 27.1416 | +0.1819 | 24 | 23.0411 | 23.3307 | −0.2896 |
| 09 | 26.3341 | 26.0856 | +0.2485 | 25 | 23.0530 | 23.3257 | −0.2727 |
| 10 | 25.4349 | 25.3461 | +0.0888 | 26 | 23.0302 | 23.2932 | −0.2630 |
| 11 | 24.7540 | 24.7792 | −0.0252 | 27 | 22.9989 | 23.3194 | −0.3205 |
| 12 | 24.2804 | 24.3675 | −0.0871 | 28 | 22.9717 | 23.3075 | −0.3359 |
| 13 | 24.0339 | 24.2335 | −0.1996 | 29 | 22.9796 | 23.3070 | −0.3273 |
| 14 | 23.8099 | 24.0380 | −0.2281 | 30 | 22.9778 | 23.3221 | −0.3443 |
| 15 | 23.6534 | 23.8607 | −0.2073 | 31 | 22.9782 | 23.3341 | −0.3559 |
| 16 | 23.5197 | 23.9168 | −0.3970 | 32 | 22.9697 | 23.3364 | −0.3667 |

Sustained tail lead (ep24–32, individually as required):
ep24 −0.2896, ep25 −0.2727, ep26 −0.2630, ep27 −0.3205, ep28 −0.3359,
ep29 −0.3273, ep30 −0.3443, ep31 −0.3559, ep32 −0.3667.
Mean over ep24–32: −0.3195; range [−0.3667, −0.2630]; all 9 tail epochs clear the
−0.26 level and 6/9 clear −0.30 as single-epoch contrasts.

Is there any region where the child loses? YES — the early region. The child is
WORSE (Δ>0) in 9 of the first 10 epochs (ep1, ep2, ep4–ep10; worst +1.3870 at ep1,
+0.4628 at ep4), with only ep3 marginally better (−0.0024). The sign flips at ep11
(−0.0252) and the child then leads in ALL 22 remaining epochs ep11–ep32 without a
single reversal. Peak single-epoch lead is ep19 (−0.6075); the mid-run ep16–ep23
block (−0.40 to −0.61) is stronger than the ep24–32 tail. Shape reading: early
penalty (extra randomly-initialized block starts near-identity via zero-init `out`
but still perturbs optimization), then a sustained, never-reversing lead from ep11.

## 3. Verdict numbers + parent tail shape

- Best-vs-best: 22.96969223022461 − 23.293153762817383 = **−0.323461532592773 ≈ −0.3235**
  (child better). Exact same-seed paired contrast; both from `result.json`, confirmed
  by TB minima and `best.pth` headers.
- Anchor bar (0.30 below the LIVE canonical parent): 23.293153762817383 − 0.30 =
  **22.993153762817382 ≈ 22.9932**. Bar margin: 22.993153762817382 − 22.96969223022461
  = **+0.023461532592773 ≈ +0.0235 cleared** (child below the line by 0.0235).
  The margin is 1.23× the harness same-seed replicate noise (0.019) — see §5.
- Final-vs-final (ep32 vs ep32): 22.969692 − 23.336433 = **−0.366741 ≈ −0.3667**
  (child better by more than best-vs-best).
- Parent tail shape: parent best@26 (23.293154), then ep27 23.319393, ep28 23.307508,
  ep29 23.306957, ep30 23.322147, ep31 23.334095, ep32 23.336433 — a monotonic-ish
  drift of **+0.043279 ≈ +0.04** from best to ep32 (val plateaus then degrades while
  `train/loss_epoch` keeps falling 3.29→2.83, i.e. mild overfit past ep26).
  Child tail shape: child ep26 23.030201 → ep32 22.969692, i.e. **−0.0605 continued
  improvement** over the same span; child best is at the wall (ep32 == best).
- Does the child beat the parent partly because the parent drifted up? For
  BEST-vs-BEST: no — the +0.043 parent drift does not enter the −0.3235 (both are
  minima). Decomposition: −0.3235 = (child@26 − parent@26) + (child@32 − child@26)
  = −0.2630 + (−0.0605). So ~81% (−0.263) is already present as a paired lead AT the
  parent's best epoch, and ~19% (−0.060) comes from the child still improving to the
  wall while the parent had stopped improving. For FINAL-vs-FINAL (−0.3667): yes —
  that number IS inflated by the parent drift: −0.3667 = −0.3235 + (−0.0433). Anyone
  quoting final-vs-final must subtract the +0.04 parent overfit drift to get back to
  the verdict-relevant −0.32. Caveat in the other direction: the child's own best is
  truncated (still falling at ep32), so best-vs-best slightly UNDERSTATES what a
  longer budget would show for the child under this seed — but that is a statement
  about this seed's trajectory shape, not a license to extrapolate (n=1, §5).

## 4. Overhead: wall-clock vs new params / FLOPs (n=1 caveat)

- Wall-clock: child 1787.2 s vs parent 1699.4 s = **+87.8 s, +5.1665% ≈ +5.2%**,
  i.e. +2.74 s/epoch (55.85 vs 53.11 s/epoch). Both 32/32 epochs, `budget_hit=false`
  (1787.2 s < 1800 s wall with ~13 s headroom; parent ~101 s headroom).
- New params (exact from `model.py` SafeEnhance, d_fine=128, embed_dim=256, d_s=64):
  `q` 128×64+64 = 8,256; `kv` 256×64+64 = 16,448; `norm` LayerNorm(64) = 128;
  `out` 64×128+128 = 8,320; **total 33,152 ≈ ~33k** as stated. Independent check:
  `best.pth` size Δ +134,808 B ≈ 33.7k fp32-equivalents — consistent.
- FLOPs estimate (forward, per image; nominal K=3 exemplars, HW=96×96=9,216 query
  tokens at S=384 → fine at S/4): q-proj 9216·128·64 ≈ 75.5M MACs; kv-proj
  3·256·64 ≈ 0.05M; score einsum 9216·3·64 ≈ 1.77M; mix einsum ≈ 1.77M; out-proj
  9216·64·128 ≈ 75.5M. **Total ≈ 154.6M MACs ≈ 309M FLOPs (×2) per image**,
  ≈ 2.47G MACs per batch-16 step, ≈ 18.0G MACs extra over the run
  (228 steps/epoch × 32). That is a small dense surcharge on the head (frozen
  backbone dominates absolute FLOPs) and is directionally consistent with the
  observed +5.2% wall-clock, the remainder being the two extra einsum memory passes
  over the 9216×3 similarity map plus AMP/launch overhead.
- n=1 caveat: ONE paired run. The +87.8 s mixes the block's true cost with run-to-run
  system jitter (I/O, thermal, queue neighbor) — no variance estimate is available
  from a single timing pair, so +5.2% is a point observation, not a cost model. Same
  for the −0.32: exact as a same-seed contrast, unmeasured as a population effect.

## 5. Sensitivity reading — what this run licenses (no verdict change)

- Licensed statement (exact): under seed 20260830 with the canonical harness, adding
  the H0014 block moves best val MAE from 23.2932 to 22.9697, a same-seed paired
  contrast of **−0.3235**, clearing the pre-registered 0.30 bar by **+0.0235**.
- Fragility arithmetic: the +0.0235 clearance is essentially ONE noise unit above the
  harness same-seed replicate noise (0.019; ratio 1.23×). A same-seed rerun that comes
  out 0.024 worse on either side flips bar clearance. This is a knife-edge pass, not
  a robust clearance — it is compatible with BOTH "real +0.3-class effect" and "a
  +0.28 effect plus +0.04 lucky draw", which the ledger's `supports w=1.0` (not a
  confirmation) already reflects.
- Level-vs-contrast discipline (AGENTS §6): seed spread is ~±1.0 MAE, so the absolute
  levels (23.29, 22.97) carry NO cross-seed meaning at the 0.30 bar. Only the paired
  −0.32 contrast is interpretable, and only under this seed.
- What a confirmation run must show (same seed, same protocol, no other edits):
  (a) replicate the SIGN and ORDER of the tail lead — child ahead in the ep11+ block
  with no extended reversal, and a best-vs-best contrast ≤ −0.30 (i.e. child best at
  least 0.30 below the concurrent same-seed parent's best, recomputed from that run's
  live parent number per the anchor rule); (b) bar clearance with MARGIN worth trusting
  — given 0.019 replicate noise, a second clearance of ≥ +0.05–0.10 (or two
  independent same-seed replicates both clearing) is what would move this from
  "knife-edge pass" to "stable pass"; a single rerun landing at −0.28/−0.30 holds the
  current `supports` reading but does not strengthen it; (c) report BOTH best-vs-best
  and final-vs-final with the parent's ep26→ep32 drift quoted, so drift-inflation
  (≈ +0.04 here) cannot be mistaken for effect size. A different-seed run tests
  generality (level claim, needs multi-seed), NOT the 0.30 bar.

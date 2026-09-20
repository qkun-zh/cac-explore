# Causal feedback — N0010_h0009 (H0009 `use_densexp` DenseTailExpert)

Parent: N0002_h0001 (EMA val MAE 22.5641 @ep30). Child outcome: **24.0500 @ep32**
(`result.json` code ok, 32/32ep, `budget_hit` false, elapsed_s 1794.9).
Margin **+1.49** vs parent; falsifier bar 22.26 missed by **1.79**. Verdict on the
booked dual bar: **REFUTED (null-and-dragging)** — the expert never engaged AND
the joint run still degraded with the N0009-predicted readout signature. All
numbers below were measured off the server checkpoint / tb logs on 2026-09-20,
not recollected. TB series cross-checked against sibling quant.md (identical).

## 1. What was tested (one switch)

Post-decoder dense-tail expert `DenseTailExpert` (verified **4,322 params** in a
fresh `DenseTailExpert(cond_dim=64)` construction: exp1 2080 + exp2 2112 + out
65 + gate 65; node total ~31.349M ≤ 32M): `cond_map → exp1(64→32 1x1) → GELU →
exp2(32→64 1x1) → GELU → zero-init out(64→1 1x1)` → residual `r` (zeros at init);
per-image gate `g = sigmoid(Linear(GAP(cond_map).detach()))`, output
`dens = base + g*r`. GCA untouched, decoder `in_ch` untouched (192), no S
recompute, no shared projection, no temperature touch by construction.
Design claim under test: stop-gradient proxy + private params + zero-init keep
the confirmed simprior readout isolated, so a miss reads as mechanism rejection,
never readout corruption (`idea.md:13,18,25-26`). The Lens-3 cheap-falsification
branch (`idea.md:19`) pre-registers exactly this read: gate-never-opens or
expert-norm-stays-near-zero = mechanism rejection with the readout undisturbed.

## 2. Checkpoint probe (server `run/latest/best.pth`, epoch 32)

Expert — **functionally null, gate learned closed**:
- `densexp.out.weight` (1,64,1,1) norm **0.0090** (init 0), `out.bias` 0.0009.
  The trust projection never left zero: no residual direction was ever adopted.
- `densexp.gate.weight` (1,64) norm **0.6018** (init exactly 0 — verified
  `nn.init.zeros_` at `model.py:247-248`), `gate.bias` **−0.2381** (init 0).
  The gate head moved off init but settled into a closed equilibrium (§3).
- `densexp.exp1.weight` (32,64,1,1) **2.6888** vs fresh-init scale ~3.27 (4-seed
  mean 3.25–3.29); `exp2.weight` (64,32,1,1) **3.2833** vs fresh ~4.58
  (4.55–4.66). Both trunk convs sit **18% / 28% BELOW init scale** — weight
  decay ground them down; no learned feature. Biases 0.4432/0.4762 are init-
  scale residuals, not signal.
- Verdict on the branch: trunk decayed, gate closed, out at zero. The
  hypothesized conditional peak capacity never engaged.

Simprior readout — **perturbed despite the null branch**:
- Child: qproj **7.6474** / kproj **7.6315** / temp **0.004367** /
  `out.weight` **1.0573** / `out.bias` 0.0700.
- Parent `best.pth` (ep30, re-read): qproj **5.7314** / kproj **7.3547** / temp
  **0.03932** / `out.weight` 0.5874 / `out.bias` 0.0730 — reproduces the card's
  working point (5.73/7.35/0.0393).
- Drift: q **+33.4%**, k +3.8%, **temp collapsed 9.0× toward the 1e-3 clamp**
  (only 4.4× above floor), `simprior.out` re-amplified **+80.0%**. The K-softmax
  runs ~9× sharper than the confirmed operating point with the trust projection
  re-amplified around it — the exact failure signature N0009 synthesis §4.2/§5
  flagged for this card ("temp-to-floor trajectory as the predicted failure
  signature", temp guard "within 2× of parent through ep8": 9.0× violates it).
- Broad drift beyond the readout (contrast N0009's localized +3%/−1.6%):
  decoder agg 40.7599 vs 34.7235 (**+17.4%**), fuser 35.5434 vs 30.2311
  (**+17.6%**), GCA 7.6715 vs 6.1882 (**+24.0%**); cond +1.6%, exemplar +0.8%
  intact. The whole head drifted, not just temp/out.

## 3. Functional probe (real val batch, 5 images spanning gt 7 → 2092)

| img (val idx) | gt | g (gate) | r RMS | r max\|·\| | L std | base Σ | gated Σ |
|---|---|---|---|---|---|---|---|
| 576 | 7.0 | 0.0722 | 0.000036 | 0.00004 | 0.003 | 13.59 | 13.26 |
| 1025 | 11.0 | 0.0883 | 0.000042 | 0.00005 | 0.005 | 17.87 | 17.75 |
| 1015 | 27.0 | 0.0958 | 0.000044 | 0.00006 | 0.013 | 32.93 | 32.52 |
| 754 | 120.0 | 0.0806 | 0.000040 | 0.00005 | 0.038 | 90.97 | 90.60 |
| 229 | 2092.0 | 0.1450 | 0.000086 | 0.00014 | 0.079 | 301.61 | 304.97 |

- "Gate opened on dense images?" — **relatively yes, absolutely no.** g rises
  ~2× from sparsest (0.072) to densest (0.145), so the detached GAP proxy does
  carry dense-level signal as designed — but the gate never opens: ≤0.15
  everywhere, including gt=2092. The optimizer evaluated the branch and voted
  it off.
- r is **3 orders of magnitude below the operating range** (r RMS / L std
  ~0.001; max|·| 1.4e-4 vs L std up to 0.079). Base vs gated count sums differ
  <2% on all 5 images. The N0009 guard "r RMS ≪ L std" is satisfied to an
  extreme — the branch is a null — yet temp still collapsed 9× (§2).
- Tail anecdote (n=1, labeled as such): on the gt=2092 image the base predicts
  301.6 and the gated output 305.0 — still massively undercounted per the
  STATE diagnostic. The catastrophic tail was NOT rescued; the dense-tail
  conjunct (gt>500 mean |Δ| vs 511.7) stays unevaluated at scale (no per-image
  dump) but gets no support here.

## 4. Learning dynamics (tb scalars, child vs parent)

- Child `val/mae`: 270.56 (ep1) → 34.74 (ep8) → 25.30 (ep16) → **24.05**
  (ep32, best=final). Parent: 153.29 → 27.27 → 23.25 → 22.57. Gap opens at ep1
  (+117.27, early-transient chaos, both terrible) and narrows monotonically to
  **+1.49** — interference from the start, never a tracking window.
- Child `train/loss_epoch`: 19.84 (ep1, −18.18 BETTER than parent's 38.02)
  → 8.33 (ep5) → 5.83 (ep16) → flat **5.32–5.35 from ep17 to ep32** (15-epoch
  stall). Parent descends every late epoch to **2.6183**. Final train ratio
  **2.04×**. Train-worse AND val-worse at the end = underfit/optimization drag,
  same family as N0009 (2.3×) but milder. The ep1 train-better/val-worse split
  is an early transient (zero-init makes step-0 forward ≡ parent by
  construction, `model.py:243-248,177-179`); the load-bearing facts are the
  15-epoch train stall and the +1.49 val floor.
- Val RMSE is the one scalar that moves toward the parent: child best=final
  **84.1496 @ep32** vs parent best 88.9234 @ep28 / final 89.2400 (**−4.77
  best-best, −5.09 final-final**, monotone 32/32 descent, no N0008-style
  divergence). RMSE/MAE falls (3.95 → 3.50) while MAE degrades. With r≈0 the
  density maps are nearly base-identical at the probe, so this shape change
  reflects the re-tuned base path (sharper temp → fewer huge errors, more small
  biases is one consistent reading) — attribution uncertain without per-image
  dumps, recorded honestly. It cannot touch the MAE-bar verdict (bar missed by
  1.79, ~50× paired-contrast precision).
- Limitation: `run/latest/` holds only `best.pth` (no intermediate ckpts), so
  the N0009-proposed temp-trajectory guard cannot be evaluated temporally —
  WHEN temp collapsed (early deflection vs late runaway) is unknowable from
  this run's artifacts. The endpoint violation (9.0×) is certain; the path is
  not.

## 5. Harness ruled out

- Checksums re-verified on server (`sha256sum`): `config.toml`
  `8f8d14b1c6e058f0cf1e499a…` ✓, `model.py`
  `70410fe91dd6d346a036b4f…` ✓ — match `result.json`
  (`config_sha256 8f8d14b1c6e058f0`, `model_sha256 70410fe91dd6d346`).
- Server config diff vs parent is **exactly one switch**: `+ use_densexp = true`
  (`diff` of the two `config.toml`). `hparams.yaml` confirms canonical
  protocol on both: seed 20260830, augment false, deterministic true,
  AdamW/cosine/AMP/MSE+0.4-L1 identical.
- Run completed **32/32 epochs**, `budget_hit: false`, `elapsed_s` 1794.9
  (5.1 s UNDER τ_max), status done / code ok. No truncation confound:
  best=final @ep32 on a flat floor (last-5 range 0.0154, crawl ≈ −0.004/ep →
  ~460 epochs to close the 1.79 bar gap — quant §3 arithmetic, adopted).
- Construction verified append-only (`model.py:279-284`, after all parent
  modules; out/gate zero-init `model.py:243-248`; flag-off op-identical
  `model.py:178-179`): step-0 ≡ parent, shared-module init draws preserved
  (AGENTS rule 13). No NaN/Inf in any 32-pt series; `train/mae` all-NaN on
  both runs is the standing logger artifact.

## 6. Mechanism verdict

**H0009 REFUTED as null-and-dragging.** The idea.md Lens-3 falsification
branch fired cleanly: gate never opens (≤0.15 incl. gt=2092), expert trunk
decayed below init (−18%/−28%), out stayed at 0.009 — the conditional-capacity
mechanism was rejected by the joint optimizer itself. Yet hosting the dead
branch still cost **+1.49 MAE** with temp collapse 9×, `simprior.out` +80%,
broad head drift (+17–24% decoder/fuser/GCA), and a 2.04× train stall. The
readout was NOT left undisturbed.

Family update — N0009's ban is confirmed in outcome but its mechanism story is
too narrow. N0009 (engaged-suppressive, −19.5%, operating-point re-centering
via `softplus(L+r)`) cannot explain N0010, because here r≈0 means NO operating-
point shift — yet the same temp-to-floor + re-amplification + stall attractor
appeared. The sharpening runaway now spans four nodes (N0006 temp 0.0214,
N0008 0.0255, N0009 0.0033, N0010 0.0044 — every trainable head perturbation
of N0002 has collapsed temp below the confirmed 0.0393). Extended rule: **no
trainable post-decoder additive on the density path, however detached, however
private, however small — N0010 already had r≈0 with full detach and still paid
1.49.** "Smaller r" or "harder detach" cannot escape a ban whose null member
fails. Remaining plausible channel here is joint-optimization drag from step 1
(extra gradient path into cond_map through the trunk + AdamW/wd competition
deflecting the trajectory into a worse basin where temp sharpens to
compensate) — stated as inference, not measurement; the endpoint facts above
are measurement. Suggested evidence: `contradicts` on H0009, weight w=0.85
(clean full-schedule run, decisive miss ~50× precision, mechanism-measured
null + 9× temp-collapse signature; docked below the 0.90 catastrophic/NaN
class per the N0006 +1.02 precedent).

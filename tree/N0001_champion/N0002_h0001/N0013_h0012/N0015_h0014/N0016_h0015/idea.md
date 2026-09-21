# H0015 idea — two-regime per-cell gains routed by fixed spatial-extent cut (`use_dualcal`)

Parent N0015_h0014 (live: peakcal mechanism NULL w_p=0.0017, win carried by temp-pin hygiene vs N0013 temp 0.0413) · SOLO single-switch card (`use_dualcal`, default false; build-time assert it requires `use_simprior` + `use_cellcal`) · frozen backbone, head-only training · decoder in_ch untouched (192 = 128 + 64) · protocol v2 (augment=true via `--set augment=true`, 32ep/1800s, EMA eval) + seed FIXED 20260830 · triple bars (conjunctive): THEN final EMA val MAE <= 19.0431 (0.30 below live 19.3431) AND gt>500 dense-tail mean |del| below 330.81 AND gt<50 sparse slice at or below 5.45 — per §6 bars semantics these numbers re-instantiate from the live v2 parent at verdict time; the hypothesis text itself is never edited.

## §0 Booked-hypothesis placeholder (Lead: pass verbatim to `discovery hypo --new`)

IF two-regime per-cell gains routed by fixed spatial-extent cut (use_dualcal) IN the live N0015_h0014 head under v2 protocol seed 20260830, THEN final EMA val MAE falls at least 0.30 below 19.3431 AND gt>500 dense-tail mean |del| falls below 330.81 AND gt<50 sparse slice falls at or below 5.45, BECAUSE the self-normalized spatial-extent count nelev separates sparse from dense images where the global scalar surrogate cannot, and the split weights learn a sparse discount (w_lo < 0) against background-texture overcount while the dense weight keeps the cellcal gain (w_hi > 0), DISPROVED IF final EMA val MAE is not at least 0.30 below the live v2 parent or dense-tail mean |del| is not below the live dense baseline or sparse gt<50 slice exceeds 5.45 or the weights do not split with w_hi > 0 > w_lo.

## §1 Survey note (web-verified 2026-09-21, no closed-door derivation)

Claim grounded: one global per-cell gain cannot serve sparse and dense images at once — sparse images need a discount against background-texture overcount while dense images need the gain kept. The fix is regime-specific treatment, routed by a statistic that actually separates the regimes.

- L2HCount (Xu et al., arXiv:2503.12935 — verified: abs page resolves to "L2HCount: Generalizing Crowd Counting from Low to High Crowd Density via Density Simulation"; Dual-Density Memory Encoding Module with LDCM/HDCM learns low- and high-density patterns separately because diffuse and peaked regimes demand different treatment): precedent that low- vs high-density patterns must be handled by separate parameters. Our card is its frozen-head per-cell analog: two scalar gain weights instead of two memory banks, routed by a fixed spatial-extent cut instead of attention.
- CMTL (Sindagi et al., arXiv:1707.09605 — verified: abs page resolves to "CNN-based Cascaded Multi-task Learning of High-level Prior and Density Estimation for Crowd Counting"; count-classification prior coarsely estimates image count and feeds it into density estimation; code github.com/svishwa/crowdcount-cascaded-mtl): precedent that a coarse image-level count prior conditions the density map. Our card is its non-learned counterpart: the coarse prior is a fixed deterministic count (nelev), and it conditions the gain instead of the features.
- DecideNet (Liu et al., arXiv:1712.06679 — verified: abs page resolves to "DecideNet: Counting Varying Density Crowds Through Attention Guided Detection and Density Estimation", CVPR 2018; detection branch for sparse scenes, regression branch for dense scenes, routed by attention): precedent that sparse and dense regimes need different estimators with explicit routing. Our card keeps the same actuator on both sides (per-cell energy gain) and routes by a fixed cut — no learned gate, no detector, no attention.
- Deliberately NOT cited: temperature/entropy/matching literature (banned by verdicts for this lineage stage). This card computes no K-softmax, no entropy, no match volume — one count over detached energy plus the existing guarded gain ops.

## §2 Exact math + attach points (parent `tree/N0001_champion/N0002_h0001/N0013_h0012/N0015_h0014/model.py`)

Parent facts: backbone frozen (`requires_grad_(False)` model.py:35, `train()` keeps net eval :45-48); `CountingHead.__init__` (:147-163) reads non-module flags `use_simprior` (:159), `use_cellcal` (:161), `use_peakcal` (:163); `CountingHead.forward(h2, h3, bboxes_in)` (:165-188) computes `fine` (:167, (B,128,96,96)), `e` (:170), `cond_map` (B,64,96,96) (:172), SimPrior residual branch with trailing cellcal/peakcal kwargs (:175-185), decoder concat `torch.cat([fine, cond_map], 1)` (:186, 128+64=192). `SimPrior.forward` (:225-252): `q`/`p` projections (:227-228), `S` cosine volume (:229), K-softmax `W` with `self.temp.detach().clamp_min(1e-3)` (:230), `top1` (:231), `cons` (:232), `ev = cat([top1, cons])` (:233, (B,2,96,96)), cellcal gain block (:234-239), peakcal block (:240-251), `return self.out(ev)` (:252). `CellCal` owns one zero-init scalar (:255-266); `PeakCal` owns one zero-init scalar (:269-282). `Counter.__init__` (:284-316) attaches SimPrior (:301-302), then CellCal (:307), then PeakCal (:312) AFTER every parent module — the append-only RNG pattern (AGENTS.md rule 13) this card copies. Temp-pin: `self.head.simprior.temp.requires_grad_(False)` when flag on (:315-316). Config: seed 20260830 (config.toml:9), `use_simprior=true` (:27), `use_cellcal=true` (:28), `use_peakcal=true` (:29).

Routing statistic (MEASURED TODAY by Lead, inference-only on N0015 checkpoint, val n=1286 — use as given):

```
m    = fine.detach().mean(dim=1, keepdim=True).clamp_min(0)   # (B,1,96,96)
mbar = m.mean(dim=(2, 3), keepdim=True)                       # (B,1,1,1)
mhat = m / (mbar + 1e-6)                                      # mean 1.0 by construction, scale-invariant
nelev = (mhat > 1.5).sum(dim=(1, 2, 3))                       # (B,) spatial-extent count, detached
dense_like = (nelev >= 1100)                                  # fixed cut C0=1100
```

Measured separation: sparse (gt<50, n=895) nelev med 57, max 1096; dense (gt>500, n=17) nelev med 1636, min 421. C0=1100 routes 100% of sparse images to the sparse weight; the majority of dense images to the dense weight. Trade-off on record: ~4 dense images fall below C0=1100 and receive the sparse weight; the learned weights balance this (895 sparse dominate w_lo). The alternative (lower cut) leaks sparse images into the gain weight — worse, since sparse purity is the card's point.

Ruled-out alternative (DEAD — state explicitly): the H0010-style per-image surrogate g (log global fine-energy) does NOT separate regimes — sparse g mean 0.2022 vs dense 0.1885; any cut strands both classes. A g-cut router is DEAD. Sum-energy is likewise useless (sparse med 2057 vs dense med 2132). The absolute-energy variant n_active_05>200 separates similarly but is scale-fragile (absolute threshold shifts across checkpoints) — rejected in favor of the self-normalized mhat form, which is mean-1.0 by construction.

DualCal (`use_dualcal`, TWO zero-init scalars `w_hi`, `w_lo` in a module constructed LAST, D fixed — no new constants learned), all shapes for S=384, batch B:

```
# --- two-regime per-cell gains; same actuator and surrogate as cellcal ---
m    = fine.detach().mean(dim=1, keepdim=True).clamp_min(0)   # (B,1,96,96)
mbar = m.mean(dim=(2, 3), keepdim=True)                       # (B,1,1,1)
mhat = m / (mbar + 1e-6)                                      # mean 1.0 by construction
nelev = (mhat > 1.5).sum(dim=(1, 2, 3)).detach()              # (B,) routing only, never a gain input
w = torch.where((nelev >= 1100).view(-1, 1, 1, 1), w_hi, w_lo) # per-image regime pick, detached
gain = torch.exp(w * torch.log1p(mhat))                       # (B,1,96,96); w=0 -> all ones
ev   = ev * gain                                              # broadcast over the 2 evidence channels
out  = self.out(ev)                                           # existing zero-init projection; (B,64,96,96)
```

Equivalently per image: ev' = ev * exp(w_hi * log1p(mhat)) if dense_like else ev * exp(w_lo * log1p(mhat)). Routing is deterministic, detached, with NO learned gate (H0009 lesson: fixed cut; both weights get gradient on their subsets from step 0). Relation to cellcal (parent live mechanism): this EXTENDS it — same actuator (per-cell gain on ev), same surrogate (mhat); when w_hi == w_lo the card reduces exactly to cellcal. Twin-objection pre-emption: different claim (regime-split vs global gain), different falsifier (sign split w_hi > 0 > w_lo predicted), different routing statistic (nelev count, never used before in this lineage).

Attach (minimal-diff DECISION — trailing kwargs on `SimPrior.forward`, no refactor of its internals): signature becomes `forward(self, fine, e, use_cellcal=False, w_c=None, use_peakcal=False, w_p=None, use_dualcal=False, w_hi=None, w_lo=None)`; body inserts the dualcal block in place of / after the cellcal block (:234-239) before `return self.out(ev)` (:252); when `use_dualcal` is on, the dualcal gain replaces the single-w_c gain (same guarded ops). Defaults keep every existing call (:178/:180/:182/:184) valid and parent-path byte-identical. `CountingHead.forward` flag-on path (nested inside the `use_simprior` branch; `fine` already in scope at :167): pass `use_dualcal=self.use_dualcal, w_hi=self.dualcal.w_hi, w_lo=self.dualcal.w_lo` alongside the existing kwargs. `CountingHead.__init__` reads `self.use_dualcal = _get(cfg, "use_dualcal", False)` (non-module flag, no RNG — same pattern as :159/:161/:163). `Counter.__init__` constructs `self.head.dualcal = DualCal()` LAST, after the PeakCal attach (:312); both scalars zero-init, drawing no RNG. DualCal holds NO normalization layers, so train/eval agree exactly. Decoder in_ch untouched (192); `e` untouched (read-only cosine input, never gated/rescaled/written); `fine` read via detach only; no match/prototypes/queries/attention anywhere.

Temp-pin hygiene REQUIRED: when `use_dualcal` is on, the Coding Agent sets `self.head.simprior.temp.requires_grad_(False)` at construction AND the W line (:230) keeps `self.temp.detach().clamp_min(1e-3)`. Detach is value-identical, so the step-0 identity proof below is unaffected; the free-temp collapse path under joint perturbation is structurally closed (N0015 lesson: the win rode temp-pin hygiene).

Step-0 byte-identity proof (for the Coding Agent — must hold before smoke): (a) `w_hi`/`w_lo` init exactly 0.0 ⇒ `w` exact zeros on both branches ⇒ `w * log1p(mhat)` exact zeros ⇒ `exp(0)==1.0` exactly ⇒ `ev * gain == ev` bit-for-bit (fp16 identity: x*1.0==x) ⇒ `self.out(ev)` unchanged ⇒ flag-on-at-init forward equals the parent; temp detach changes no values; routing (`nelev`, `dense_like`) is detached integers that select between two identical gains, hence value-invisible; (b) with `use_dualcal=false` (default) no new line executes — forward is the parent's byte-for-byte; (c) def-use: `self.head.dualcal` defined in `Counter.__init__` before any forward; `fine` in scope at :167; Condenser path untouched; (d) no shared state: `w_hi`/`w_lo` live under `self.head.dualcal` only, never aliasing `simprior.*`, `cellcal.*`, `peakcal.*`, or `cond.*`; the `m/mhat/nelev` field is recomputed from `fine.detach()` and never reuses (or writes) cellcal/peakcal locals; (e) smoke must assert: flag-off == parent AND flag-on-at-init == flag-off (max abs diff 0). Params +2 scalars, far under the head budget.

Finite-guard inventory (EVERY division/exp listed — same guarded ops as cellcal): (1) `mbar + 1e-6` denominator (copied from parent :237); (2) `log1p(mhat)` with mhat >= 0 by `clamp_min(0)` on m — no log of negative; (3) `exp` of `w * log1p(mhat)` — bounded in practice by the learned scalar scale (same form as parent cellcal gain :238; Coding Agent keeps the identical guard shape, no exp of unbounded args); (4) temp path keeps parent `clamp_min(1e-3)` plus detach; (5) routing comparisons are integer counts on detached tensors — no gradient, no NaN path; (6) NO new normalization layers, NO softmax, NO log of learned quantities — the only reduction is the parameter-free count, exactly deterministic train/eval.

## §3 Why-not (H0010 scalar, H0012 global-gain, H0009 learned-gate, §11)

- NOT H0010 (per-image scalar g): H0010's surrogate is measured DEAD on this checkpoint (sparse g mean 0.2022 vs dense 0.1885 — overlapping means, any cut strands both classes). This card never computes g; the routing statistic is the spatial-extent count nelev, which separates (sparse max 1096 vs dense med 1636). Different statistic, measured separation, stated numbers.
- NOT H0012 global-weight redux (reduces-to-cellcal argument): when w_hi == w_lo the card IS cellcal arithmetically — that is the point, not the objection. The claim is the SPLIT: one weight cannot discount sparse background texture while keeping dense gain; two weights can. The falsifier (sign split w_hi > 0 > w_lo) is unstatable for H0012 and is the core prediction here. A null split (weights ~= equal) refutes the card while leaving H0012 intact.
- NOT an H0009-style learned gate: routing is a FIXED cut (C0=1100), deterministic and detached — no learned gate parameters, no gating representation path to collapse or starve. Both weights receive gradient on their routed subsets from step 0 (H0009 lesson honored: no gate to open before learning).
- §11 compliance (explicit): §11 bars pre-condenser exemplar gating in ANY granularity and DDCA/spatial-summaries/RGA/final-layer readout/backbone unfreeze. This card never gates, rescales, adapts, or writes `e` — the exemplar path through the Condenser is byte-untouched; no spatial summary feeds the decoder; decoder in_ch stays 192; backbone stays frozen. The intervention is a similarity-side/decoder-side energy gain — a live direction per §11 ("query/similarity-side and decoder-side — unproven, re-book fresh"), not a settled one.

## §4 Predictions + failure modes + 4 reads

Verdict context: live parent N0015_h0014 v2 19.3431 @ep32 (peakcal NULL w_p=0.0017); dense gt>500 mean |del| 330.81; sparse gt<50 5.814 vs 5.45 guard (missed twice running — the card's point).

- P1 (sign split — the core prediction): w_hi_final > 0 > w_lo_final, both decisively off-zero — dense keeps gain, sparse gets discount.
- P2 (sparse correction): gt<50 slice 5.814 → ≤ 5.45 — the sparse weight discounts exactly the false mass cellcal's global gain amplified.
- P3 (dense preserved): dense-tail holds below 330.81 with no dense-block collapse — the dense weight keeps (or re-learns) the gain on dense-like images.
- P4 (overall): final EMA val MAE ≤ 19.0431. All three numeric bars conjunctive.
- F1 (routing useless): w_hi_final ~= w_lo_final (within ±1e-2) ⇒ routing carries no signal ⇒ REFUTE even if a bar flickers.
- F2 (no discount learned): w_lo_final >= 0 ⇒ sparse discount never materialized ⇒ REFUTE with diagnosis; do not re-book energy variants silently.
- F3 (dense collapse): any dense-block prediction craters vs parent or dense-tail rises above the live baseline ⇒ the ~4 misrouted dense images (or gain starvation) broke the dense side ⇒ REFUTE.

Mechanism reads at verdict (Lead hook harness):
1. SIGN SPLIT w_hi > 0 > w_lo at verdict (the core prediction — sparse gets discount, dense keeps gain).
2. Sparse gt<50 slice ≤ 5.45 (the twice-missed guard — the entire point).
3. Dense-block preserved: dense-tail below the live dense baseline with no block-level collapse (no H0011/H0013 signature).
4. BOTH MAE bars: final EMA val MAE ≥ 0.30 below the live v2 parent AND the dense-tail bar jointly.

## §5 Risks + same-seed v2 pair protocol + futility

- Risk: early-routing noise — at ep0-10 features are random so nelev routes noisily and both weights see mixed subsets. Accepted as regularization, not a bug: gradients flow to both weights from step 0, and routing sharpens as fine stabilizes; final routing is evaluated at verdict, not at init.
- Risk: ~4 dense images below C0=1100 receive the sparse weight — bounded by F3 as the refutation path with full diagnostic value; the learned weights balance this (895 sparse dominate w_lo), and lowering the cut leaks sparse into the gain weight, which is worse.
- Risk: split never engages (F1) — cheap null with full diagnostic value (reads 2–4 still valid on best.pth).
- Risk: extra forward ops (one count + where-select + pointwise gain on 96×96) — negligible against the backbone; no RNG draws, no buffers, hence no train/eval skew.
- Risk: temp freeze removes a parent degree of freedom — accepted deliberately: N0015 proved the win rides temp-pin hygiene, and pinning is value-identical at step 0.
- Protocol: SAME-SEED v2 pair only — child (`use_dualcal=true` over the parent flags, `--set augment=true`) vs live N0015_h0014 v2 (`--set augment=true`), seed 20260830 FIXED, 32ep/1800s, EMA eval, `final EMA val MAE` + `gt>500 dense-tail mean |del|` + `gt<50 sparse slice`. NEVER cross-protocol: pre-v2 (augment=false) numbers are historical record only. Verdict binds to live v2 numbers at run time (§6 semantics); booked 19.3431/330.81/5.45 and any re-instantiated live values are both stated in the evidence note. `repro_run --set seed=` is FORBIDDEN without explicit user approval.
- Futility (rule 16): launch carries `--set futility_bar=19.0431`; gates ep16 WARN-only / ep24 HALT; feasibility must be visible by ep24 — the split is a two-scalar correction whose signal (weights leaving zero with opposite signs in early-epoch hooks) appears in the first epochs if the mechanism is real.

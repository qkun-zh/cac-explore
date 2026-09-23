# H0013 idea — background-prototype suppression with dense-safe margin gating on confirmed SimPrior evidence (`use_negsup`)

Parent N0013_h0012 (H0012 cellcal CONFIRMED) · SOLO single-switch card (`use_negsup`, default false; build-time assert it requires `use_simprior`) · frozen backbone, head-only training · decoder in_ch untouched (192 = 128 + 64) · protocol v2 (augment=true via `--set augment=true`, 32ep/1800s, EMA eval) + seed FIXED 20260830 · triple bars (conjunctive): THEN final EMA val MAE <= 20.5861 (0.30 below live 20.8861) AND gt>500 dense-tail mean |del| below 421.13 AND gt<50 sparse slice at or below 5.45 — per §6 bars semantics these numbers re-instantiate from the live v2 parent at verdict time; the hypothesis text itself is never edited.

## §0 Booked-hypothesis placeholder (Lead: pass verbatim to `discovery hypo --new`)

IF background-prototype suppression with dense-safe margin gating (use_negsup) IN the live N0013_h0012 head under v2 protocol seed 20260830, THEN final EMA val MAE falls at least 0.30 below 20.8861 AND gt>500 dense-tail mean |del| falls below 421.13 AND gt<50 sparse slice falls at or below 5.45, BECAUSE ReLU(neg-pos-0.1) against detached SimPrior top1 self-disables where positive evidence is strong (dense recall preserved) while subtracting background-match mass exactly on the weak-pos/strong-neg cells that cellcal amplified on sparse images, DISPROVED IF final EMA val MAE is not at least 0.30 below the live v2 parent or dense-tail mean |del| is not below the live dense baseline or sparse gt<50 slice exceeds 5.45 or lam_final stays near zero or sparse/dense supp-active fractions are equal.

## §1 Survey note (web-verified 2026-09-21, no closed-door derivation)

Claim grounded: explicit negative/background modeling suppresses false positives in counting — positives alone cannot exclude look-alikes, and the H0012 verdict (sparse gt<50 5.771 vs v2-baseline 5.451, +0.32) is exactly a false-mass-on-background signature asking for this correction.

- ConCoNet (Soliven et al., Pattern Recognition Letters 171, 2023, doi:10.1016/j.patrec.2023.04.018 — verified: DOI resolves via doi.org to "ConCoNet: Class-agnostic counting with positive and negative exemplars"; no arXiv version; no public code; FSC147 numbers unverified): precedent that negative exemplars ("what not to count") calibrate similarity and plug into other counters. Our card is the frozen-head analog with surrogate background prototypes built from outside-box cells instead of annotated negatives.
- VA-Count (Zhu et al., ECCV 2024, arXiv:2407.04948 — verified: abs page resolves to "Zero-shot Object Counting with Good Exemplars"; Exemplar Enhancement Module + Noise Suppression Module discriminates positive vs negative exemplars via contrastive separation; code github.com/HopooLinZ/VA-Count): precedent that a dedicated suppression path for negative exemplars is load-bearing alongside the positive path. We differ by operating on dense per-cell evidence rather than exemplar selection.
- CountingDINO (Pacini et al., WACV 2026, arXiv:2504.16570 — verified: abs page resolves; official repo github.com/lorebianchi98/CountingDINO exposes `filter_background=True` as a first-class pipeline switch; training-free DINOv2 val 25.48 / test 20.93): precedent that background filtering on frozen self-supervised features is a separable, ablatable stage. Our card is its trained-head counterpart: a learned-strength background subtraction on frozen DINOv3 features.
- CountGD++ (Amini-Naieni & Zisserman, CVPR 2026, arXiv:2512.23351 — verified: abs page resolves to "CountGD++: Generalized Prompting for Open-World Counting"; accepts positive AND negative visual/textual prompts; negative exemplars cut MAE by an order of magnitude on multi-class images such as red vs white blood cells): strongest direct evidence that "what not to count" resolves exactly the look-alike confusion our sparse slice shows.
- CACViT background suppression (in-repo survey §F6/B7 ablation) is consistent with the same direction; not re-verified here.
- Deliberately NOT cited: divisor-schedule and distributional-statistic literature. This card pins a fixed scalar margin (0.1) and computes no per-image distributions; the mechanism is margin-gated subtraction of background evidence.

## §2 Exact math + attach points (parent `tree/N0001_champion/N0002_h0001/N0013_h0012/model.py`)

Parent facts: backbone frozen (`requires_grad_(False)` model.py:35, `train()` keeps net eval :45-48); `CountingHead.__init__` (:147-161) reads non-module flags `use_simprior` (:159) and `use_cellcal` (:161); `CountingHead.forward(h2, h3, bboxes_in)` (:163-180) computes `fine` (:165, (B,128,96,96)), `e` (:168), `cond_map` (B,64,96,96) (:170), SimPrior residual branch (:173-177), decoder concat `torch.cat([fine, cond_map], 1)` (:178, 128+64=192). `SimPrior.forward` (:217-232): `q`/`p` own projections (:219-220), `S` cosine volume (:221, (B,K,96,96)), K-softmax consensus `W` (:222), `top1` (:223, (B,1,96,96)), `cons` (:224), `ev = cat([top1, cons])` (:225, (B,2,96,96)), cellcal gain (:226-231), `return self.out(ev)` (:232). `CellCal` owns one zero-init scalar (:235-246). `Counter.__init__` (:249-272) attaches SimPrior (:266-267) then CellCal (:272) AFTER every parent module — the append-only RNG pattern (AGENTS.md rule 13) this card copies. Config: seed 20260830 (config.toml:9), `use_simprior=true` (:27), `use_cellcal=true` (:28), `augment=false` live (:32; v2 runs add `--set augment=true`).

NegSup (`use_negsup`, own projections + ONE zero-init scalar `lam`), all shapes for S=384, batch B, K=3, J=4 slots, d=32:

```
# --- deterministic background bank from frozen h3 (B,384,24,24), cell = 16px S-space ---
covered_b = union over K bboxes3 (S-space), each expanded by 8px S-space, rasterized to 24x24
sel_b     = raster-order list of UNCOVERED cells, M = |sel_b|
sz = ceil(M/J); runs = [sel_b[i*sz:(i+1)*sz] for i in range(J)]   # all M cells covered exactly once
proto_j   = mean of h3_d cells in runs[j]; empty runs[j] -> repeat previous run's prototype
M == 0    -> bank_b = global h3_d mean repeated J times            # degenerate full-coverage fallback
bank      = stack(proto)                                          # (B,J,384), detached (frozen backbone)
# --- own-space background match (NEVER reuse simprior qproj/kproj — H0006 lesson) ---
qn   = normalize(Wqn(fine_d), dim=1)         # Wqn: Conv2d(128,32,1,bias=False) own; (B,32,96,96)
bn   = normalize(Wbn(bank), dim=-1)          # Wbn: Linear(384,32,bias=False) own; (B,J,32)
Sneg = einsum('bchw,bjc->bjhw', qn, bn)      # (B,J,96,96) cosine in [-1,1]
neg  = Sneg.max(dim=1, keepdim=True).values  # (B,1,96,96) worst-background match
supp = ReLU(neg - pos_d - 0.1)               # (B,1,96,96); margin 0.1 FIXED constant, never learned
adj  = lam * supp                            # lam: nn.Parameter(zeros(())) own; (B,1,96,96)
evp  = ev - adj                              # broadcast over post-cellcal-gain ev (B,2,96,96)
out  = self.out(evp)                         # existing zero-init projection; (B,64,96,96)
```

Attach (minimal-diff DECISION — two trailing kwargs on `SimPrior.forward`, no refactor of its internals): signature becomes `forward(self, fine, e, use_cellcal=False, w_c=None, return_top1=False, ev_adj=None)`; body inserts `if ev_adj is not None: ev = ev - ev_adj` before `return self.out(ev)` (:232) and `if return_top1: return out, top1.detach()` at the end. Defaults keep every existing call (:175/:177) valid and parent-path byte-identical. `CountingHead.forward` flag-on path (nested inside the `use_simprior` branch; `h3` and `bboxes_in` already in scope at :163):

```
with torch.no_grad():
    _, pos_d = self.simprior(fine, e, use_cellcal=..., w_c=..., return_top1=True)  # probe: no grad, no side effects
adj = self.negsup.adjustment(fine.detach(), h3.detach(), bboxes_in, pos_d)         # lam * supp
cond_map = cond_map + self.simprior(fine, e, use_cellcal=..., w_c=..., ev_adj=adj)  # sole grad path into simprior
```

The two-call form is deliberate: simprior grad dynamics stay IDENTICAL to the parent (gradient flows only through the second call; the probe contributes nothing), and `pos` is the DETACHED simprior top1 — no second positive path is added. `CountingHead.__init__` reads `self.use_negsup = _get(cfg, "use_negsup", False)` (non-module flag, no RNG — same pattern as :159/:161). `Counter.__init__` constructs `self.head.negsup = NegSup()` LAST, after the CellCal attach (:272); Wqn/Wbn Kaiming draws append to RNG, `lam` zero-init draws nothing. NegSup holds NO normalization layers, so the probe call is side-effect-free in train and eval.

Prototype-construction determinism proof: (i) boxes are seed-pinned data; (ii) h3 comes from the frozen backbone under deterministic cudnn, hence identical across runs at fixed seed; (iii) raster partition is pure index arithmetic — no k-means, no value sorting, no RNG at forward time; (iv) slot count is statically J=4 (repeat-last and global-mean fallbacks change values, never shapes); (v) `lam` init is a constant 0. Forward is therefore a pure function of (batch, trained weights) with zero forward-time RNG.

Dense-safety argument (core of the card): on dense images `pos` is strong almost everywhere, so `ReLU(neg-pos-0.1)` is ≈0 and suppression self-disables where recall lives; on sparse images background cells carry weak `pos` plus strong `neg`, so suppression fires exactly on the false-mass cells cellcal amplified (w_c=+0.1658 drove sparse 5.451→5.771). Directional selectivity is therefore structural, and read 2 pre-registers its measure.

Structural-twin pre-emption: NOT H0011 `msq` (that card added h2-grid queries + attention and collapsed the dense block 3425: 312.7→46.7; this card adds no queries, no attention, no feature path — a data-computed field subtracted from confirmed evidence, so the recall-destruction signature is structurally unavailable); NOT an F5 density-logit refine (intervention is pre-decoder on `ev`, decoder in_ch stays 192); NOT §11 exemplar gating (`e` is read-only cosine input inside the simprior match and is never rescaled, gated, or written by negsup).

Step-0 byte-identity def-use check (for the Coding Agent — must hold before smoke): (a) `lam` init exactly 0.0 ⇒ `adj` is exact zeros ⇒ `evp == ev` bit-for-bit ⇒ `self.out(evp) == self.out(ev)` ⇒ flag-on-at-init forward equals the parent (Wqn/Wbn random init is provably irrelevant at init since it is multiplied by `lam`=0); (b) with `use_negsup=false` (default) no new line executes — forward is the parent's byte-for-byte; (c) def-use: `self.head.negsup` defined in `Counter.__init__` before any forward; `pos_d` from the already-computed simprior match, `bank` from in-scope `h3`, `qn` from in-scope `fine`; Condenser path untouched; (d) no shared state: Wqn/Wbn/`lam` live under `self.head.negsup` only, never aliasing `simprior.*`, `cellcal.*`, or `cond.*`; (e) smoke must assert: flag-off == parent AND flag-on-at-init == flag-off (max abs diff 0). Params 128·32 + 384·32 + 1 = 16385 (≈16.4k + the scalar; refinement of the skeleton's ~5k estimate — exact arithmetic from the fixed dims), far under the head budget.

## §3 Why-not (H0009 gate lesson + H0006 sharing + §11)

- NOT an H0009-style zero-init multiplier gate: `lam` scales an ALREADY-COMPUTED `supp` field, and the gate-init problem does not apply because `supp` is data — it is fully computed in the forward pass from frozen features plus detached `pos` regardless of `lam`, and `lam`=0 is the identity. Gradient to `lam` is nonzero from the first backward pass whenever `supp` correlates with the residual error (dL/d`lam` = −sum(dL/d`ev` · `supp`)), so `lam` escapes zero at once if suppression helps and the projections follow; no branch must open before it can learn. Learning sets only the STRENGTH of a fixed-shape field, never opens or closes a representation path.
- H0006 sharing lesson honored: Wqn/Wbn are own projections in the negsup module. Reusing simprior `qproj`/`kproj` would entangle the confirmed positive-match space with the background space and alter simprior grad dynamics; the two-call attach above exists precisely to leave those dynamics identical.
- §11 exemplar-gating compliance (explicit): §11 bars pre-condenser exemplar gating in ANY granularity. This card never gates, rescales, adapts, or writes `e` — the exemplar path through the Condenser is byte-untouched, and negsup holds no exemplar parameters. The intervention is a similarity-side post-readout subtraction on `ev` — a live direction per §11 ("query/similarity-side and decoder-side — unproven, re-book fresh"), not a settled one.

## §4 Predictions + failure modes + 4 reads

Verdict context: live parent N0013_h0012 v2 20.8861 @ep32; dense gt>500 mean |del| 421.13; sparse gt<50 5.771 vs v2-baseline 5.451 (+0.32 guard miss); v2 baseline N0002 21.4362 / 423.40 / 5.451.

- P1 (sparse correction): gt<50 slice 5.771 → ≤5.45 — suppression fires on weak-pos/strong-neg background cells, the exact population cellcal amplified.
- P2 (dense neutrality): dense-tail holds below 421.13 with dense-block predictions (3425/3428/3433) intact — `pos` strong ⇒ ReLU gate ≈0, so recall is structurally preserved, distinguishing from H0011-style collapse.
- P3 (overall): final EMA val MAE ≤ 20.5861. All three bars conjunctive.
- F1 (quiet null): `lam_final` < 1e-2 ⇒ field never engaged ⇒ REFUTE even if a bar flickers.
- F2 (selectivity dead): sparse/dense supp-active fractions comparable (read 2 ratio < 3) ⇒ the dense-safety claim fails ⇒ REFUTE with diagnosis (background texture, not false mass, drove the sparse miss; do not re-book background variants silently).
- F3 (dense collapse): any dense-block prediction drops >3% vs parent (3425/3428/3433) ⇒ H0011 signature ⇒ REFUTE.

Mechanism reads at verdict (Lead hook harness):
1. `lam_final` ≥ 1e-2, decisive positive.
2. Selectivity proof: mean active-cell fraction (`supp`>0) over gt<50 val images ≥ 3× that over gt>500 val images.
3. Dense-block recall preserved: each of 3425/3428/3433 within 97% of the parent prediction.
4. Sparse gt<50 slice ≤ 5.45 (the +0.32 eaten).

## §5 Risks + same-seed v2 pair protocol + futility

- Risk: prototype contamination (8px margin too small for large objects; box-adjacent context leaks into the bank) — bounded by the margin gate itself (edge cells carry strong `pos`) with R3/F3 as the refutation path, not a post-hoc story. Margin stays a fixed constant under phase-1 cost discipline.
- Risk: `lam` dead (supp uncorrelated with residual) — cheap F1 null with full diagnostic value (reads 2–4 still valid on best.pth).
- Risk: repeat-last duplication when M<J and the M=0 global-mean fallback — harmless by construction (duplicated bank rows leave the max unchanged; uniform `neg` on dense meets strong `pos` and gates off).
- Risk: extra no_grad simprior probe ≈ 2× simprior FLOPs — negligible against the backbone; no RNG draws, no buffers, hence no train/eval skew.
- Protocol: SAME-SEED v2 pair only — child (`use_negsup=true` over the parent flags, `--set augment=true`) vs live N0013_h0012 v2 (`--set augment=true`), seed 20260830 FIXED, 32ep/1800s, EMA eval, `final EMA val MAE` + `gt>500 dense-tail mean |del|` + `gt<50 sparse slice`. NEVER cross-protocol: pre-v2 (augment=false) numbers are historical record only. Verdict binds to live v2 numbers at run time (§6 semantics); booked 20.8861/421.13/5.45 and any re-instantiated live values are both stated in the evidence note. `repro_run --set seed=` is FORBIDDEN without explicit user approval.
- Futility (rule 16): launch carries `--set futility_bar=20.5861`; gates ep16 WARN-only / ep24 HALT; feasibility must be visible by ep24 — suppression is a low-capacity correction whose signal (`lam` leaving zero, selectivity in early-epoch hooks) appears in the first epochs if the mechanism is real.

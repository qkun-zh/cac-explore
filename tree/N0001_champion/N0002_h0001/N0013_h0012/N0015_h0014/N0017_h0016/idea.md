# H0016 idea — additive energy-mass residual from self-normalized energy (`use_enmass`)

Parent N0015_h0014 (live: 19.3431/330.81/sparse 5.814; peakcal NULL, win carried by temp-pin hygiene) · SOLO single-switch card (`use_enmass`, default false) · frozen backbone, head-only training · decoder in_ch untouched (192 = 128 + 64) · protocol v2 (augment=true via `--set augment=true`, 32ep/1800s, EMA eval) + seed FIXED 20260830 · triple bars (conjunctive): THEN final EMA val MAE <= 19.0431 (0.30 below live 19.3431) AND gt>500 dense-tail mean |del| below 330.81 AND gt<50 sparse slice at or below 5.45 — per §6 bars semantics these numbers re-instantiate from the live v2 parent at verdict time; the hypothesis text itself is never edited.

## §0 Booked-hypothesis placeholder (Lead: pass verbatim to `discovery hypo --new`)

IF additive energy-mass residual from self-normalized cell energy (use_enmass) IN the live N0015_h0014 head under v2 protocol seed 20260830, THEN final EMA val MAE falls at least 0.30 below 19.3431 AND gt>500 dense-tail mean |del| falls below 330.81 AND gt<50 sparse slice falls at or below 5.45, BECAUSE the similarity-independent additive projection places mass from cell energy alone where the cosine matcher is blind, restoring recall on mid-range collapse images whose fine-energy peaks are present while match evidence is absent, DISPROVED IF final EMA val MAE is not at least 0.30 below the live v2 parent or dense-tail mean |del| is not below the live dense baseline or sparse gt<50 slice exceeds 5.45 or the out-projection norm stays near zero or its mean weight is negative.

## §1 Survey note (web-verified 2026-09-21, no closed-door derivation)

Claim grounded: density placed directly from image/texture energy — without any exemplar match — is a legitimate crowd-counting actuator with precedent. When the matcher is blind, energy alone still carries the crowd signal.

- L2HCount (Xu et al., arXiv:2503.12935 — verified: abs page resolves to "L2HCount: Generalizing Crowd Counting from Low to High Crowd Density via Density Simulation"; High-Density Simulation Module shifts/overlays low-density images to simulate dense crowd texture, Dual-Density Memory Encoding Module re-encodes density features from texture patterns): precedent that dense-crowd mass is placed from texture/energy structure, not from exemplar matching. Our card is its frozen-head pointwise analog: one 1x1 energy→mass projection instead of memory banks.
- Count2Density (Litrico et al., arXiv:2509.03170 — verified: abs page resolves to "Count2Density: Crowd Density Estimation without Location-level Annotations"; Historical Map Bank + unsupervised saliency prior generates pseudo-density maps from count-level supervision, recovering spatial mass without location matches): precedent that spatial density mass is recovered from a non-matching prior (saliency/energy) plus a global constraint. Our card is its supervised per-cell analog: self-normalized cell energy as the spatial prior, projected additively into cond.
- MCNN (Zhang et al., CVPR 2016, doi:10.1109/CVPR.2016.70 — verified: CVF open-access page resolves to "Single-Image Crowd Counting via Multi-Column Convolutional Neural Network"; parallel columns map raw pixels/texture to density maps via 1x1 fusion, no exemplar or query anywhere): precedent that direct texture→density mapping without matching is the founding crowd-counting architecture. Our card re-introduces exactly this path as a residual inside a matching head.
- CRNet (Liu et al., doi:10.1109/TIP.2020.2994410 — verified: DOI resolves to "Crowd Counting Via Cross-Stage Refinement Networks", IEEE TIP 29:6800-6812; stacked FCNs refine density maps from hierarchical density priors, code github.com/lytgftyf/Crowd-Counting-via-Cross-stage-Refinement-Networks): precedent for additive density-prior residuals refined stage by stage. Our card is its single-step head-internal analog: one additive mass residual into cond before the decoder.
- Deliberately NOT cited: temperature/entropy/matching literature (banned for this card). This card computes no K-softmax, no entropy, no cosine volume, no temperature — one self-normalized energy field plus one zero-init 1x1 projection.

## §2 Exact math + attach points (parent `tree/N0001_champion/N0002_h0001/N0013_h0012/N0015_h0014/model.py`)

Parent facts: backbone frozen (`requires_grad_(False)` model.py:35, `train()` keeps net eval :45-48); `CountingHead.__init__` (:147-163) reads non-module flags `use_simprior` (:159), `use_cellcal` (:161), `use_peakcal` (:163); `CountingHead.forward(h2, h3, bboxes_in)` (:165-188) computes `fine` (:167, (B,128,96,96)), `e` (:170), `cond_map` (B,64,96,96) (:172), SimPrior residual branch with trailing cellcal/peakcal kwargs (:175-185), decoder concat `torch.cat([fine, cond_map], 1)` (:186, 128+64=192). `SimPrior.forward` (:225-252): `q`/`p` projections (:227-228), `S` cosine volume (:229), K-softmax `W` with `self.temp.detach().clamp_min(1e-3)` (:230), `top1` (:231), `cons` (:232), `ev = cat([top1, cons])` (:233, (B,2,96,96)), cellcal gain block (:234-239), peakcal block (:240-251), `return self.out(ev)` (:252). `CellCal` owns one zero-init scalar (:255-266); `PeakCal` owns one zero-init scalar (:269-282). `Counter.__init__` (:284-316) attaches SimPrior (:301-302), then CellCal (:307), then PeakCal (:312) AFTER every parent module — the append-only RNG pattern (AGENTS.md rule 13) this card copies. Temp-pin: `self.head.simprior.temp.requires_grad_(False)` when flag on (:315-316). Config: seed 20260830 (config.toml:9), `use_simprior=true` (:27), `use_cellcal=true` (:28), `use_peakcal=true` (:29).

Raison d'etre — energy is PRESENT, similarity is ABSENT (Lead measurements, inference-only on live N0015 checkpoint, val — FACTS):

Error pool: mid gt 50-500 (n=374) carries 56% of total |err| at 37.5/image, signed -26.8, 73% under; dense 23%, sparse 21%. The MID range, not the tails, is the bulk.

| img | gt | pred | m_mean | m_max | top1mean |
|-----|----|------|--------|-------|----------|
| 3481 | 431 | 21.9 | 0.22 | 0.7 | -0.5 |
| 3428 | 458 | 103.8 | 0.21 | 0.6 | -0.75 |
| 3488 | 431 | 82.7 | 0.23 | 0.9 | -0.3 |
| 3484 | 356 | 26.9 | 0.22 | 0.8 | -0.4 |
| 3482 | 315 | 28.1 | 0.21 | 0.5 | -0.6 |
| 5059 | 286 | 28.8 | 0.22 | 0.6 | -0.2 |
| 3476 | 262 | 14.7 | 0.23 | 0.7 | -0.45 |
| 3487 | 321 | 99.5 | 0.21 | 0.8 | +0.25 |
| 3483 | 261 | 40.7 | 0.22 | 0.6 | -0.35 |
| 2850 | 287 | 77.6 | 0.21 | 0.7 | -0.1 |

(m_mean 0.21-0.23 NORMAL not dead; m_max 0.5-0.9 peaked cells exist; top1mean -0.75..+0.25 matcher BLIND; temp pinned 0.07. Same 34xx scene family as the dense block.)

Gain-vs-additive pre-emption (state explicitly): all ev-GAIN mechanisms (H0010/H0012/H0015: gain x similarity-evidence) multiply evidence that is not there — on 3481-class images 1.2 x ~0 = ~0, so gains cannot fix these images. The additive path places mass from energy alone — the only live signal there (m_max 0.5-0.9). Different actuator (additive projection vs multiplicative gain), different claim (recall from energy vs reweighting evidence), different falsifier (proj norm/mean-weight reads vs gain-weight reads).

EnMass (`use_enmass`, ONE zero-init 1x1 `Conv2d(1, 64, 1)` — weight AND bias zero-init, +128 params = 64 weights + 64 biases), all shapes for S=384, batch B:

```
# --- additive energy-mass residual; similarity-independent, decoupled recompute ---
m    = fine.detach().mean(dim=1, keepdim=True).clamp_min(0)   # (B,1,96,96)
mbar = m.mean(dim=(2, 3), keepdim=True)                       # (B,1,1,1)
mhat = m / (mbar + 1e-6)                                      # mean 1.0 by construction, scale-invariant
mass = self.enmass(mhat)                                      # (B,64,96,96); all-zeros at init
cond_map = cond_map + mass                                    # pre-decoder residual into cond
dens = self.decoder(torch.cat([fine, cond_map], 1))           # in_ch still 192
```

Attach (minimal-diff DECISION): `CountingHead.__init__` reads `self.use_enmass = _get(cfg, "use_enmass", False)` (non-module flag, no RNG — same pattern as :159/:161/:163). `Counter.__init__` constructs `self.head.enmass = nn.Conv2d(1, 64, 1)` LAST, after the PeakCal attach (:312), with `nn.init.zeros_` on weight AND bias; zero-init draws no RNG. `CountingHead.forward` inserts the enmass block AFTER the SimPrior branch (:175-185) and BEFORE the decoder concat (:186), gated on `self.use_enmass`; mhat is RECOMPUTED from `fine.detach()` — DO NOT reuse cellcal/peakcal locals, decoupled by construction. Defaults keep the parent path byte-identical. Decoder in_ch untouched (192); `e` untouched (never read, never gated, never written); `fine` read via detach only; no match/prototypes/queries/attention/temperature anywhere in the new path.

Why not H0011-lane (pre-empt explicitly): no queries, no attention, no new feature path — a pointwise energy→mass projection with no spatial mixing, so it cannot learn suppression patterns. It adds mass only where energy is high; the bound reads mass >= 0 iff weights >= 0 — gradient decides, verdict reads the sign.

Temp-pin hygiene REQUIRED: when `use_enmass` is on, the Coding Agent sets `self.head.simprior.temp.requires_grad_(False)` at construction AND the W line (:230) keeps `self.temp.detach().clamp_min(1e-3)`. Detach is value-identical, so the step-0 identity proof below is unaffected; the free-temp collapse path under joint perturbation is structurally closed (N0015 lesson: the win rode temp-pin hygiene).

Step-0 byte-identity proof (for the Coding Agent — must hold before smoke): (a) `enmass` weight AND bias init exactly 0.0 ⇒ `mass` exact zeros ⇒ `cond_map + mass == cond_map` bit-for-bit ⇒ decoder input unchanged ⇒ flag-on-at-init forward equals the parent; temp detach changes no values; mhat recompute is detached and value-invisible when the projection is zero; (b) with `use_enmass=false` (default) no new line executes — forward is the parent's byte-for-byte; (c) def-use: `self.head.enmass` defined in `Counter.__init__` before any forward; `fine` in scope at :167; Condenser path untouched; (d) no shared state: `enmass.*` lives under `self.head.enmass` only, never aliasing `simprior.*`, `cellcal.*`, `peakcal.*`, or `cond.*`; the `m/mbar/mhat` field is recomputed from `fine.detach()` and never reuses (or writes) cellcal/peakcal locals; (e) smoke must assert: flag-off == parent AND flag-on-at-init == flag-off (max abs diff 0). Params +~128, far under the head budget.

Finite-guard inventory (EVERY division listed): (1) `mbar + 1e-6` denominator (copied from parent :237); (2) `clamp_min(0)` on m before any use — no negative energy; (3) NO exp, NO log, NO softmax, NO normalization layers in the new path — the only op is a linear 1x1 projection, exactly deterministic train/eval; (4) temp path keeps parent `clamp_min(1e-3)` plus detach; (5) mhat guarded copy (recompute, never aliased). Coding Agent: keep the identical guard shape to cellcal mhat.

## §3 Why-not (H0012/H0015 gains multiply dead evidence; H0011 queries learn suppression; §11)

- NOT H0012/H0015 (ev-gains): those cards scale existing evidence — on the 3481-class block evidence is ~0 (top1mean -0.75..+0.25), so 1.2 x ~0 stays ~0 (3481: 431->21.9 unmoved by any gain). This card places mass from energy alone (m_max 0.5-0.9 — the only live signal). Different actuator, different claim, different falsifier (proj-norm/mean-weight reads F1/F2 below vs gain-weight reads). A gain twin-objection fails on the arithmetic.
- NOT H0011 (queries/attention): query-side paths learn spatial suppression patterns (the H0011 signature: mean weight < 0, learned suppressor). This card has no queries, no attention, no spatial mixing — a pointwise projection that adds mass where energy is high. F2 (mean weight < 0) refutes the card as an H0011-signature failure, which keeps the lanes falsifiably separate.
- §11 compliance (explicit): §11 bars pre-condenser exemplar gating in ANY granularity and DDCA/spatial-summaries/RGA/final-layer readout/backbone unfreeze. This card never gates, rescales, adapts, or writes `e` — the exemplar path through the Condenser is byte-untouched; no spatial summary feeds the decoder; decoder in_ch stays 192; backbone stays frozen. The intervention is a similarity-side/decoder-side energy residual — a live direction per §11 ("query/similarity-side and decoder-side — unproven, re-book fresh"), not a settled one.

## §4 Predictions + failure modes + 4 reads

Verdict context: live parent N0015_h0014 v2 19.3431 @ep32 (peakcal NULL w_p=0.0017); dense gt>500 mean |del| 330.81; sparse gt<50 5.814 vs 5.45 guard; mid gt 50-500 pool 56% of total |err| at 37.5/image signed -26.8.

- P1 (mass-adding projection): out-proj (1x1) norm decisively off zero AND mean weight > 0 at verdict — the path adds mass, not suppression.
- P2 (3481-class recall): worst-mid block preds rise 5-10x toward gt (e.g. 3481 21.9, 3484 26.9, 3476 14.7 move by hundreds toward gt) — recall, not reshuffle.
- P3 (bounded slippage): sparse gt<50 slice does not exceed 5.9 — energy-mass adds everywhere including sparse backgrounds, but the bet is mid-block gains dwarf it.
- P4 (overall): final EMA val MAE <= 19.0431 AND dense-tail below 330.81. All three numeric bars conjunctive.
- F1 (quiet null): out-proj norm ~= 0 at verdict ⇒ the path never engaged ⇒ REFUTE even if a bar flickers.
- F2 (learned suppressor): mean weight < 0 ⇒ the projection learned suppression, the H0011 signature ⇒ REFUTE with diagnosis; do not re-book energy variants silently.
- F3 (slippage dominates): sparse blows past 6.5 while mid improves ⇒ energy-mass adds more background than crowd ⇒ REFUTE.

Mechanism reads at verdict (Lead hook harness):
1. OUT-PROJ norm decisively off zero AND mean weight > 0 (mass-adding, not suppressing).
2. 3481-class block recall: block preds rise 5-10x toward gt (recall, not reshuffle).
3. SPARSE slippage guard: gt<50 slice must not exceed 5.9 (hard refute past 6.5).
4. BOTH MAE bars + dense bar: final EMA val MAE >= 0.30 below the live v2 parent AND dense-tail below the live baseline jointly.

## §5 Risks + same-seed v2 pair protocol + futility

- Risk: sparse slippage — quantified bet: energy-mass adds everywhere, so sparse backgrounds gain some mass; the bet is the mid pool (56% of total |err|, 37.5/image, signed -26.8, 73% under) dwarfs the sparse pool (21%). Bounded by read 3 (5.9 guard) and refuted by F3 (6.5 blowout). No spatial mixing means no learned background suppressor can offset this — accepted as the card's core risk.
- Risk: dense-block double-count — 34xx family overlaps the dense block; additive mass on peaked dense cells overshoots. Bounded by the dense-tail bar (below 330.81) with full diagnostic value on best.pth.
- Risk: quiet null (F1) — the 1x1 stays near zero because match-evidence gradients dominate. Cheap null with full diagnostic value (reads 2-4 still valid on best.pth).
- Risk: extra forward ops (one energy reduction + one 1x1 conv on 96x96) — negligible against the backbone; no RNG draws, no buffers, hence no train/eval skew.
- Risk: temp freeze removes a parent degree of freedom — accepted deliberately: N0015 proved the win rides temp-pin hygiene, and pinning is value-identical at step 0.
- Protocol: SAME-SEED v2 pair only — child (`use_enmass=true` over the parent flags, `--set augment=true`) vs live N0015_h0014 v2 (`--set augment=true`), seed 20260830 FIXED, 32ep/1800s, EMA eval, `final EMA val MAE` + `gt>500 dense-tail mean |del|` + `gt<50 sparse slice`. NEVER cross-protocol: pre-v2 (augment=false) numbers are historical record only. Verdict binds to live v2 numbers at run time (§6 semantics); booked 19.3431/330.81/5.45 and any re-instantiated live values are both stated in the evidence note. `repro_run --set seed=` is FORBIDDEN without explicit user approval. v2 pair protocol: both sides run under augment=true at the fixed seed; the ONLY delta is the `use_enmass` flag.
- Futility (rule 16): launch carries `--set futility_bar=19.0431`; gates ep16 WARN-only / ep24 HALT; feasibility must be visible by ep24 — the additive residual is a +128-param correction whose signal (proj norm leaving zero with positive mean in early-epoch hooks, mid-block preds lifting) appears in the first epochs if the mechanism is real.

# H0011 idea — single-step image-query exemplar-key cross-attention lane at h2 (`use_msq`)

Parent N0002_h0001 · SOLO single-switch card (`use_msq`, default false) · frozen backbone, head-only training · decoder in_ch untouched (192 = 128 + 64) · protocol v2 (augment=true, 32ep, EMA eval) + seed FIXED 20260830 · bars: THEN final EMA val MAE <= 21.1362 (0.30 below live v2 21.4362) AND gt>500 dense-tail mean |del| below 423.40 — per §6 bars semantics these numbers re-instantiate from the live v2 parent at verdict time; the hypothesis text itself is never edited.

## §0 Booked-hypothesis placeholder (Lead: pass verbatim to `discovery hypo --new`)

IF single-step image-query exemplar-key cross-attention lane at h2 (use_msq) IN N0002_h0001 head-only v2 protocol seed 20260830, THEN final EMA val MAE falls at least 0.30 below 21.4362 and gt>500 dense-tail mean |del| falls below 423.40, BECAUSE 48x48 image queries pulling detached exemplar keys restore small-object spatial recall pooled away before the 1/4 Condenser readout, DISPROVED IF final EMA val MAE is not at least 0.30 below the live v2 parent or dense-tail mean |del| is not below the live v2 baseline.

## §1 Survey note (web-verified 2026-09-21, no closed-door derivation)

- GeCo2 (Pelhan et al., AAAI 2026, arXiv:2511.08048 — verified: abs page resolves to "Generalized-Scale Object Counting with Gradual Query Aggregation", AAAI-26 oral/proceedings pp. 8314–8321; repo github.com/jerpelhan/GECO2): dense queries gradually aggregate exemplar info across scales; per-scale `PrototypeAttentionBlock.forward(image_f, prototypes)` = `image_f + CrossAttn(q=image_f, k/v=prototypes)` then LayerNorm — i.e. IMAGE-QUERY pulls prototype info, prototypes themselves precomputed per scale. Per-scale banks use SEPARATE attention blocks per level (own projections per level), gradual fusion `up+add` per level with trainable LUM (2x bilinear + 3x3 conv + GeLU). Ablation: removing query levels Q1/Q2/Q3 → 12.71/22.64/19.97 val (load-bearing); N_DA=1 ≈ full (9.45 vs 9.40 MAE) — iteration is expendable, per-level interaction is not.
- CoDi (Sustar et al., arXiv:2512.20153 — verified: abs page resolves to "CoDi — an exemplar-conditioned diffusion model for low-shot counting", v2 16 Jul 2026; repo github.com/gsustar/CoDi): exemplar-conditioning layers injected at first+last UNet layers "where the resolution is still sufficiently high to enable small objects detection"; beats LOCA 15% MAE / 29% RMSE on FSC147. Backing for conditioning at HIGH resolution (our h2 lane), cited for placement only — no diffusion machinery is borrowed.
- SQLNet (Wu et al., IEEE TIP, arXiv:2311.10011 — verified: abs page resolves to "SQLNet: Scale-Modulated Query and Localization Network for Few-Shot Class-Agnostic Counting"): Hierarchical Exemplars Collaborative Enhancement + Exemplars-Unified Query Correlation — exemplars interact with query features in a unified manner at feature resolution before localization. Cited as independent precedent that exemplar↔query interaction belongs pre-readout, not post-decoder.
- FamNet (Ranjan et al., CVPR 2021, arXiv:2104.08391 — verified: abs page resolves to "Learning To Count Everything"; introduces FSC-147): correlation-map + density-prediction baseline our head descends from. Cited for task framing only.
- Deliberately NOT cited: Zhai et al. ICML23 (arXiv:2303.06296, entropy/temperature). This card is about spatial recall at higher resolution, not temperature or collapse dynamics; the H0010 surrogate-g spread (0.091, log-compressed) is reported as a readout observation, not a mechanism to fix here.

## §2 Exact math + attach points (parent `tree/N0001_champion/N0002_h0001/model.py`)

Parent facts: `CountingHead.__init__` (model.py:146–159) builds fuser (FineFuser, model.py:51–74) → exemplar (ExemplarEncoder on h3, model.py:77–108) → cond (Condenser, model.py:111–125) → decoder `DensityDecoder(in_ch=D + cond_dim)` = 192 (model.py:156). `forward` (model.py:161–175): `e = self.exemplar(h3, ...)` (:166), `cond = self.cond(fmap, e)` (:167), `cond_map` reshape to (B,64,96,96) (:167–168), H0001 SimPrior residual pattern (:171–172), decoder concat `torch.cat([fine, cond_map], 1)` (:173). h2 is (B,192,48,48) at 1/8; h3 is (B,384,...). `Counter.__init__` (model.py:224–242) attaches SimPrior AFTER all parent modules (model.py:241–242) — the append-only RNG pattern (AGENTS.md rule 13) this lane copies. Config: seed 20260830 (config.toml:9), `augment=false` live (config.toml:31; v2 runs add `--set augment=true`), `d_fine=128, cond_dim=64` (config.toml:21–22).

Lane (`use_msq`, d_lane=64, single head), all shapes for S=384, B batch, K=3 exemplars:

```
e_det  = e.detach()                                  # (B,K,256); NO gradient into exemplar path
Q      = flatten(qproj(h2))                          # qproj: Conv2d(192,64,1) own; (B,2304,64), 48*48=2304
K      = kproj(e_det)                                # kproj: Linear(256,64) own; (B,K,64)
V      = vproj(e_det)                                # vproj: Linear(256,64) own; (B,K,64)
A      = softmax(Q @ K^T / sqrt(64), dim=-1)         # (B,2304,K)
U      = (A @ V).reshape(B,64,48,48)                 # attended exemplar content per h2 cell
H      = LayerNorm(Q_grid + U)                       # residual + norm; exact GeCo2 PrototypeAttentionBlock mirror
J      = GELU(conv3x3(H))                            # own 3x3 conv, single-level LUM analog (B,64,48,48)
lane   = zero1x1(upsample2x(J))                      # 1x1 Conv2d(64,64) ZERO-INIT -> (B,64,96,96)
cond_map' = cond_map + lane                          # gradual residual fusion; decoder input shape unchanged
```

Attach: `CountingHead.__init__` reads `self.use_msq = _get(cfg, "use_msq", False)` (non-module flag, no RNG — same pattern as `use_simprior`, model.py:159). `Counter.__init__` constructs `self.head.msq = MSQ(...)` AFTER the SimPrior attach lines (model.py:241–242), so all parent init draws keep RNG positions; only the lane's own qproj/kproj/vproj/conv draws append. `CountingHead.forward` inserts after :168, beside the :171–172 SimPrior block:

```
if self.use_msq:
    cond_map = cond_map + self.msq(h2, e)
```

`torch.cat([fine, cond_map], 1)` (:173) still sees 128+64=192 channels — decoder in_ch untouched.

Step-0 byte-identity def-use check (for the Coding Agent — must hold before smoke): (a) `zero1x1` weight AND bias init to zeros ⇒ `lane` is all-zeros at init ⇒ with `use_msq=true` at init, `cond_map' == cond_map` bit-for-bit, decoder input identical to parent; (b) with `use_msq=false` (default) the `if` body never executes — forward is the parent's byte-for-byte including the SimPrior branch behavior; (c) def-use: `self.msq` defined in `Counter.__init__` before any forward can run; `e_det` defined from the already-computed `e` (:166) — the Condenser path (`fmap`, `e`, `cond`) is read-only, never reassigned; (d) no shared projections: q/k/v/conv/norm names live under `self.msq` only, never aliasing `self.cond.*` or `self.simprior.*`; (e) smoke must assert: flag-off forward equals parent forward, and flag-on-at-init forward equals flag-off forward (max abs diff 0). Params ≈ qproj 12k + kproj 16k + vproj 16k + 3x3 37k + zero-1x1 4k + norms ≈ 80–120k, well under the 32M mission cap.

## §3 Why this is NOT the refuted families

- NOT H0003 `padapt`: that mutated `e` in place and fed the Condenser (prototype adaptation; +3.0 MAE worse, iteration hurts). Here `e` is DETACHED and the Condenser path (`self.cond(fmap, e)`, :167) is byte-untouched — the lane is a parallel query-side branch, never an exemplar-path mutation.
- NOT H0006 `h2pool`: that SHARED qproj with the readout and starved it (catastrophic). Here every projection (qproj/kproj/vproj/conv/norms) is the lane's OWN — GeCo2's per-level separate-block isolation, the opposite of the sharing variant.
- NOT H0004 `simbank`: that built an explicit 1/8 similarity volume (key-side match map). Here there is NO similarity volume, no cosine/temperature readout — a query-side attention lane reusing the pooled `e` as k/v, fused as a residual into `cond_map`.
- NOT H0008 `hires`: that was a post-decoder residual (corrupts the calibrated readout temperature). Here fusion is PRE-decoder into `cond`, so the decoder and GCA see an enriched conditioning map through their trained interface.

## §4 Failure modes + predictions beyond MAE

Context: H0010 `counttau` (+2 scalars) engaged but redistributed: innov 21.4133 @ep32, d=-0.027 vs the -0.30 bar; dense block moved ±200 (3425.jpg 312.7→106.1 and 3665.jpg 1040.1→819.5 improved; 935.jpg 323.1→381.9 and 840.jpg worse); w_g=+0.1476, w_t=-0.1129 with surrogate-g spread 0.091. Lesson: a global gain moves mass but cannot resolve per-image direction; the 935-class failure (gt 2092, pred ~323–382 — similarity evidence near-zero everywhere) needs SPATIAL recall at 48x48, which is exactly what this lane adds.

- P1 (dense recall): on the 935-class (gt 2092) and 3425-class dense images, predicted counts rise toward gt (recall up, not merely redistributed) — the opposite signature of H0010's ±200 reshuffle. If MAE improves but the dense block only reshuffles again, the mechanism claim fails even under a pass.
- P2 (separating test): correlate the lane output `lane` with `cond_map` over val before training (random init ⇒ ≈uncorrelated) and after: cosine > 0.9 ⇒ the lane learned what the Condenser already computes ⇒ REDUNDANT branch ⇒ REFUTE even if MAE moves. Cosine well below 0.9 with dense recall up ⇒ engaged.
- P3 (mechanism read): lane out-proj (`zero1x1`) weight norm moves OFF zero during training (lane is used); parent readout trajectory (SimPrior qproj/kproj/temp or successor harness gate) matches the baseline trajectory — the gain comes from the lane, not readout drift.
- F1 (quiet null): lane norm stays ≈0 or P2 cosine > 0.9 — Condenser redundancy; refute.
- F2 (h2 noise amplification): sparse/small-gt images degrade while dense improves — lane adds texture hallucination; refute unless BOTH bars hold (bars are conjunctive).

## §5 Risks + isolating protocol

- Risk: Condenser redundancy (F1) — mitigated by the P2 separating test pre-registered as a refutation path, not a post-hoc story.
- Risk: extra 48x48 attention overfits small-object texture — bounded by zero-init (step-0 identity) + single step (no refinement loop per H0003 ledger) + ~100k params.
- Risk: RNG-order drift breaking the paired contrast — mitigated by append-only construction (all lane modules after every parent module) + fixed seed; smoke asserts step-0 identity.
- Protocol: SAME-SEED v2 pair only — child (`use_msq=true`, `--set augment=true`) vs canonical v2 baseline, seed 20260830, 32ep, EMA eval, `final EMA val MAE` + `gt>500 dense-tail mean |del|`. NEVER cross-protocol: pre-v2 (augment=false) numbers incl. 22.5641 are historical record only. Verdict binds to live v2 numbers at run time (§6 semantics); both the booked 21.4362/423.40 and any re-instantiated live values are stated in the evidence note.

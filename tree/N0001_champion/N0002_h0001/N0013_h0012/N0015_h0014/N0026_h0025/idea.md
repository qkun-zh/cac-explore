# H0025 — CapFilm: count-capacity FiLM conditioning of the DensityDecoder (decoder-side consumption interface) (SOLO card)

- Parent: `tree/N0001_champion/N0002_h0001/N0013_h0012/N0015_h0014` (LIVE parent N0015_h0014, val MAE 19.3431 @ep32 v2 protocol, seed 20260830).
- SOLO card: exactly one mechanism switch `use_capfilm` (default false), shipped as a **single-line config delta** (`use_capfilm=true`; `use_simprior`/`use_cellcal`/`use_peakcal` stay `true`, every other key byte-identical). When the flag is on, `DensityDecoder.forward` receives a zero-init per-channel FiLM pair `(gamma, beta)` produced by a new `CapFilm` module from `GAP(fine) ‖ log(ME)` (ME = CACViT-style exemplar-box magnitude embedding, pure geometry) and applies `h ← h·(1+γ) + β` on the **block output, pre-final-1×1-head, pre-softplus** — decoder-internal conditioning, NOT a post-decoder additive and NOT an output-scale multiply. Residual content, `cond_map` concat, SimPrior/cellcal/peakcal, Condenser, FineFuser, GCA, backbone: byte-identical. **~25k params** (two Linear layers, last layer zero-init → step-0 identity), constructed AFTER every parent module (append-only RNG).
- Regime: frozen backbone, head-only training. Decoder `in_ch` untouched (still `D + cond_dim` = 192); optimizer/loss/schedule invariant (AdamW, MSE+w_cnt·L1, cosine, AMP).
- Protocol: v2 (augment=true, EMA eval, seeded loaders), seed FIXED 20260830, 32ep/1800s. Same-seed paired contrast vs live parent N0015 only.
- Triple conjunctive bars (§6 semantics, bound to live parent 19.3431 under v2): THEN final EMA val MAE ≤ 19.0431 (≥0.30 below 19.3431) AND dense-tail gt>500 below 330.81 AND sparse gt<50 at or below 5.45. Missing any one bar refutes the card. Launch carries `--set futility_bar=19.0431` (see §5).
- **Why THIS card after the leverage probe (mandate premise, not re-litigated):** fine=0 → count −91.6%; cond_map=0 → count **+23.8%**; remove SimPrior residual → count **+29.9%**; cond×2 → count −17.6%; channel sens fine mean 15.2 vs cond mean 1.6 (~10×); density max 0.458 / mean 0.029 (headroom). The decoder has learned **fine=generator, cond/simprior=suppressor**; residual content is weak AND net-suppressive — six evidence-layer cards (H0018/22/23 wipe 0/11; H0016/17 quiet-null) all died by editing a low-leverage suppressive path. CapFilm does **not** edit residual content; it re-conditions the **consumption** of the generator path so the decoder's operating point (per-channel gain before the output nonlinearity) becomes capacity-dependent — the S-curve fix at the locus the two-card closure left open (§11: decoder-side explicitly "live direction, unproven, re-book fresh").
- **Honest expectation:** expected negative-to-marginal (HANDOFF discipline). Determinism crisis sd≈1.13; mechanism reads (R1/R2) weighted above level bars where they conflict (user ruling pattern from H0024).

## §0 — Bookable one-liner (for `discovery hypo --new`)

IF a zero-init count-capacity FiLM conditioning of the DensityDecoder hidden block (`use_capfilm`) that maps GAP(fine) concatenated with the log exemplar-box magnitude embedding ME = mean_K(S²/(w_k·h_k)) (S=input size, each annotation box structurally holds one object) to per-channel gamma,beta and applies h ← h·(1+gamma)+beta on the block output before the final 1×1 head and softplus, while the cond_map concat, the SimPrior residual, and every other parent line stay byte-identical, IN the live N0015 head on FSC147 v2 at seed 20260830, THEN final EMA val MAE drops at least 0.30 below 19.3431 with dense-tail gt>500 below 330.81 and sparse gt<50 at or below 5.45, BECAUSE the leverage probe measures the trained parent as fine=generator and cond/simprior=net-suppressor (fine=0 cuts count 91.6 percent, removing the residual raises count 29.9 percent, cond channel sensitivity is 10x below fine), so six residual-content cards were normalized away or wiped while the measured S-curve miscalibration (gt7–10 ratio 1.40 over to gt138+ ratio 0.685 under, mid50–300 at 0.86 carrying 44 percent of AE) is a missing per-image absolute operating point at the decoder consumption interface — zero-init FiLM injects exactly that capacity-conditioned scale pre-final-conv (SNDM’s self-normalization moved inside the decoder, CACViT’s magnitude made geometric), letting the same parent features integrate to different sums without any post-decoder additive and without touching the suppressive residual. DISPROVED IF final EMA val MAE is not at least 0.30 below 19.3431, or dense-tail gt>500 is not below 330.81, or sparse gt<50 exceeds 5.45, or the FiLM fails its engagement read (val coefficient of variation of mean_c(gamma) below 0.05, i.e. no per-image spread, or mean_c(gamma) fails to load capacity: mean on gt≥300 not at least 1.5× mean on gt<50), or the consumption-side leverage read moves the wrong way (cond_map×2 count-ratio after training not closer to 1.0 than the parent’s 0.824 in absolute value).

## §1 — Survey: decoder-side self-normalization / capacity conditioning / magnitude priors (verified IDs with URLs; no matching in-head FiLM card)

Survey date 2026-09-22. Method: live web search + arXiv/publisher verification of every cited ID (AGENTS rule 15). Focus: has anyone conditioned a counting density **decoder’s internal activation scale** on count-capacity (geometry or energy) so the output sum tracks order-of-magnitude, inside a trained pluggable head, without post-decoder additive and without loss change?

1. **SNDM: Self-Normalized Density Map for Counting Microbiological Objects** — arXiv:2203.09474 (Graczyk et al., Sci Rep 12:10583, 2022). URL: https://arxiv.org/abs/2203.09474 · https://www.nature.com/articles/s41598-022-14879-3. Verified (abs + Nature full text). Core mechanism: U²-Net fails “not [at] localization … but rather the normalization of density maps for a large range of possible outcomes (from 0 to 100)”; a bypass from the **smallest encoder block** predicts per-image β that **multiplies the last layers** producing the density map. **Relevance:** direct published statement of our exact S-curve pathology and the cure is a **network-internal scale factor read from deep features**, not a post-hoc scalar. **Difference / legality:** SNDM’s β multiplies the final layers (output-scale, gray vs our post-decoder-additive ban); CapFilm applies FiLM **pre-final-conv on the block hidden** (mandate: “place it pre-final-conv”), per-channel not per-image-scalar, conditioned on GAP(fine)+box ME. No standalone SNDM code repo found; U²-Net base: https://github.com/xuebinqin/U-2-Net.
2. **FiLM: Visual Reasoning with a General Conditioning Layer** — arXiv:1709.07871 (Perez et al., AAAI 2018). URL: https://arxiv.org/abs/1709.07871. Code: https://github.com/ethanjperez/film. Verified (abs + AAAI PDF). Core: `FiLM(F|γ,β)=γ⊙F+β`; explicitly notes concat-conditioning “simply results in a feature-wise conditional bias” and is inferior to learned affine modulation; paper includes counting tasks. **Relevance:** the operator CapFilm uses — channel-wise affine on intermediate features, O(C) params. **Difference:** FiLM is a general conditioning layer, not a counting calibration; CapFilm zero-inits (γ,β)=0 for step-0 parent identity and reads count-capacity (GAP+ME), which FiLM never specifies.
3. **DensFiLM: Density-Conditioned Video Saliency for Crowd Scenes** — arXiv:2607.25465 (Rahman, 2026-07-28). URL: https://arxiv.org/abs/2607.25465. Code: https://github.com/aniskhan25/crowdfix-saliency. Verified (abs page + HTML). Core: inserts FiLM at the Video-Swin **bottleneck**; density embedding → channel scale/shift “allowing the decoder to reconstruct … from features selected for each density regime”; ~100K params; predicted-density conditioning matches oracle labels. **Relevance:** freshest in-domain proof that **lightweight density-conditioned FiLM at the decoder/bottleneck** is the right inductive bias for crowd-density regimes — same family, different task (saliency) and different conditioning source (3-class label / own prediction). **Difference:** DensFiLM conditions on a density **class** with ~100K and no exemplar geometry; CapFilm conditions a few-shot counting decoder on **GAP(fine)+box ME** with ~25K, zero-init, inside a frozen-backbone head.
4. **CACViT: Vision Transformer Off-the-Shelf … Few-Shot Class-Agnostic Counting** — arXiv:2305.04440 (Wang et al., AAAI 2024). URL: https://arxiv.org/abs/2305.04440. Code: https://github.com/Xu3XiWang/CACViT-AAAI24. Verified (abs + PDF + repo). Core: softmax/normalization loses **order-of-magnitude**; Magnitude Embedding ME=(W_p·H_p)/(W_k·H_k)×similarity + scale emb → +21% MAE / +41% RMSE test (B8 vs B10 ablation). **Relevance:** independent naming of our scale-free finding; ME is pure **exemplar-box geometry** (annotation structure we already have in `bboxes_in`). **Difference:** CACViT re-injects ME at the **encoder input tokens** (ViT-specific); CapFilm feeds `log ME` as one scalar channel into a decoder-side FiLM MLP — no token surgery, no encoder touch, no similarity multiply (that would be gain-domain / evidence-side and refuted-adjacent).
5. **CountingDINO: Training-free Pipeline for Class-Agnostic Counting** — arXiv:2504.16570 (Pacini et al., WACV 2026). URL: https://arxiv.org/abs/2504.16570. Code: https://github.com/lorebianchi98/CountingDINO. Verified (abs + HTML; **reproduced in-lab** val 39.70). Core: unit-mass via in-box response normalization (eq.1–2), training-free. **Relevance:** shows absolute mass anchoring works when applied to **the full density pathway**, and on our 11 wipes only gently undercounts — the missing scale is consumable. **Difference:** CapFilm does not divide its own output by z; it learns a capacity→(γ,β) map so the **decoder** scales pre-head activations; no kernels, no hybrid (hybrid line dead, err-err 0.566).
6. **DecideNet** — arXiv:1712.06679 (Liu et al., CVPR 2018). URL: https://arxiv.org/abs/1712.06679. Verified (abs + CVF PDF). **DAVE** — arXiv:2404.16622 (Pelhan et al., CVPR 2024). URL: https://arxiv.org/abs/2404.16622. Code: https://github.com/jerpelhan/DAVE. Verified (abs + repo). Both: detection overcounts sparse / density undercounts dense; attention or verify-routing between regimes. **Relevance:** field consensus that **one regime cannot cover the S-curve ends** — our mid/dense under + sparse over is the density-side half of that split. **Difference:** both are two-path / detection hybrids (out of regime: second method, hybrid judged dead); CapFilm keeps ONE density path and moves its operating point per capacity — DecideNet/DAVE’s “choose regime” becomes FiLM’s “choose gain.”
7. **Learn to Scale / AutoScale** — arXiv:1907.12428 (ICCV 2019) and arXiv:1912.09632 (IJCV 2021). Code: https://github.com/dk-liang/AutoScale. Verified (abs + PDFs). Core: dense regions accumulate overlapping blobs → long-tailed density values; L2S learns **scale factors** on dense patches (AutoScale also **re-predicts** the rescaled patch). **Relevance:** same dense pathology family as our S-curve tail. **Difference:** AutoScale’s re-prediction is region decomposition (banned extra spatial summary / second prediction); CapFilm has no crop, no second forward, no region route — one global FiLM from already-computed features.
8. **DAN: Shallow Feature based Dense Attention Network** — Liu et al., AAAI 2020 (in-domain; loss/architecture side). Inhomogeneous density → per-pixel MSE regression-to-mean; dense attention + density-aware loss. **Relevance:** names the regression-to-the-mean mechanism our fixed MSE+L1 invariant cannot change (loss ban). **Difference:** DAN edits loss/attention modules; CapFilm is loss-invariant and only conditions the existing decoder — the consumption-side residual of what DAN would fix if we could touch the loss.
9. **Rejected in survey (recorded):** CountGD / CounTR / LOCA SOTA numbers (level context only, not mechanisms); CoDi (arXiv:2512.20153, https://github.com/gsustar/CoDi — latent diffusion, wholesale architecture); L2HCount (arXiv:2503.12935 — data synthesis, protocol-invariant); EBC-ZIP (arXiv:2506.19955 — loss-side NLL, loss ban); QICA quantity-hypothesis decoder losses (CVPR 2026 supp — loss-side); Count2Density (arXiv:2509.03170 — count-level training signal, not inference conditioning).

**What the survey decides:** nobody publishes “FiLM the counting density decoder on exemplar-box ME + fine GAP inside a frozen head.” Adjacent pieces all exist (SNDM self-norm, FiLM operator, DensFiLM crowd FiLM, CACViT ME, CountingDINO unit mass) but never fused at the **consumption interface** under our bans. First-principles lens (pure math): ĉ=Σ softplus(f_θ)_i under MSE on 69% gt<50 images estimates a compressed E[ĉ|feat] — a monotone S-curve is the Bayes act without a per-image scale DOF; multiplying the pre-softmax sufficient statistic by (1+γ(x)) restores one continuous absolute DOF with **zero post-decoder additive** (softplus(h·(1+γ)) ≠ softplus(h)+c) and, because γ is per-channel, lets training amplify generator channels while de-amplifying suppressor channels when capacity is high — addressing the leverage probe’s “cond=suppressor” asymmetry that a uniform output β cannot. Champion-lineage lens: cellcal CONFIRMED a *relative* gain one layer upstream on residual-into-cond (+0.55 val) but the leverage probe now measures that pathway as **net-suppressive and 10× weaker than fine** — the confirmed win was sparse FP control, not dense generation; six cards proved **content** edits there die; the untested quadrant is **modulating the fine→decoder consumption path**, which is exactly where fine=0 → −91.6% says all the count lives. Counter-intuitive/low-cost lens: after six “add more evidence” failures, the cheapest legal move is to add **almost nothing** — ~25k zero-init params that only change the decoder’s operating point, no new residual, no new substrate, no output scale, step-0 identity; if MSE never rewards capacity scaling, γ collapses to 0 and the card refutes itself as a clean quiet-null with full diagnostic value (H0024’s residual-scale fork and this consumption fork then close the OR from the two-card closure).

## §2 — Exact mechanism: CapFilm — zero-init FiLM inside DensityDecoder

### 2.1 The measured consumption-side flaw (raison d’être)

Parent N0015 path (model.py:165–188):

```
fine  = fuser(h2,h3)                          # (B,128,96,96)  GENERATOR (sens 15.2)
cond_map = Condenser(...) [+ simprior residual]  # (B,64,96,96)  SUPPRESSOR (cond=0 → +23.8% count)
dens  = softplus( head( block( cat[fine, cond_map] ) ) )   # model.py:139-140, :186
# block: Conv192→256, GN, GELU, dil-conv 256→128, GN, GELU   → h ∈ (B,128,96,96)
# head:  Conv128→1 1×1 ; softplus
```

Three measured facts (mandate premises + reexamination):

1. **Leverage:** ablations on trained best.pth — fine=0 → −91.6% count; residual removal → **+29.9%** (residual suppresses); cond×2 → −17.6%; fine channel sens 15.2 vs cond 1.6. Content on `cond_map`/residual is the wrong lever (quiet-null or dense blowup; six cards).
2. **S-curve:** val n=1286 — gt7–10 ratio 1.40 → gt138+ ratio 0.685; mid50–300 ratio 0.86 = 44% AE; dense≥1000 ratio 0.47; bias −12.4; 11 wipes are the extreme tail. Lifting 39 imgs (ratio<0.5, gt≥100) to 0.7 ⇒ −3.97 MAE.
3. **Headroom:** density max 0.458, mean 0.029 — the softplus head is nowhere near saturated; the sum is small because the **operating point** (pre-head activation scale) is trained toward the sparse-heavy conditional mean, not because the nonlinearity clips.

Therefore the card places a **capacity-conditioned (γ,β) on `h` immediately before `self.head`**, i.e. pre-final-conv (mandate-legal placement), so `Σ softplus((1+γ)·head_in(h) + β_bias_effect)` can scale per image and per channel without any post-decoder additive.

### 2.2 CapFilm exact math

New module (constructed only when flag on; class lives after `PeakCal` in model.py for append-only style):

```python
class CapFilm(nn.Module):
    """H0025 count-capacity FiLM encoder: GAP(fine) || log(ME) -> (gamma, beta).
    Last Linear ZERO-INIT => step-0 (gamma,beta)=(0,0) => decoder FiLM is identity.
    ME = mean_K( S^2 / (w_k * h_k) ) from annotation boxes (CACViT geometry, log-compressed).
    """
    def __init__(self, d_fine=128, d_mid=64, d_out=128):  # d_out = decoder block out = hidden//2 = 128
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_fine + 1, d_mid), nn.GELU(),
            nn.Linear(d_mid, 2 * d_out),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, fine, bboxes, img_size):
        gap = fine.mean(dim=(2, 3))                                   # (B, d_fine)
        wh = (bboxes[:, :, 2:4] - bboxes[:, :, :2]).clamp_min(1.0)
        area = (wh[..., 0] * wh[..., 1]).clamp_min(1.0)               # (B, K)
        me = (float(img_size) * float(img_size) / area).mean(dim=1)   # (B,) order-of-magnitude capacity
        me = torch.log(me.clamp(1.0, 1.0e4))                          # finite, guarded
        g = self.mlp(torch.cat([gap, me.unsqueeze(1)], dim=1))        # (B, 2*d_out)
        gamma, beta = g.chunk(2, dim=1)
        return gamma, beta                                            # both (B, d_out), zeros at init
```

`DensityDecoder.forward` (model.py:139–140) gains optional kwargs — `__init__` **unchanged** (no params inside decoder → no parent RNG shift at decoder construction):

```python
def forward(self, x, gamma=None, beta=None):
    h = self.block(x)
    if gamma is not None:
        # pre-final-conv FiLM; step-0 gamma=beta=0 => h*(1+0)+0 == h bit-identical
        h = h * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
    return F.softplus(self.head(h))
```

`CountingHead.forward` call site (model.py:186):

```python
gamma = beta = None
if self.use_capfilm and getattr(self, "capfilm", None) is not None:
    gamma, beta = self.capfilm(fine, bboxes_in, self.S)
dens = self.decoder(torch.cat([fine, cond_map], 1), gamma=gamma, beta=beta)
```

**Why FiLM here is not any ban:**
- **Not post-decoder additive:** nothing is added to `dens` or `out["density"]` (GCA’s pre-existing `dens+bias` at model.py:334 is parent code, untouched). Intervention is strictly **before** `self.head` and **before** softplus.
- **Not output-scale multiply (SNDM gray):** we never form `β·dens`; the scale acts on the hidden sufficient statistic, so spatial shape and softplus nonlinearity stay in the loop (per-channel γ can reshape, not just rescale).
- **Not global calibration scalar:** (γ,β) are per-image vectors of length 128 with measured-spread engagement gate (CV ≥ 0.05) — a global scalar has CV=0 by construction (H0002/H0010 dead).
- **Not gain-domain mask / evidence edit:** SimPrior residual, cellcal gain, substrate, sign: byte-identical. CapFilm never reads `ev`, `top1`, `S`, or `e`.
- **Not pre-condenser exemplar gating:** `e`/Condenser untouched; only `bboxes` geometry (already an input) and `fine` are read.
- **Not DDCA/RGA/final-layer readout:** no new spatial summary, no readout of decoder logits as a count head; FiLM is modulation of existing block features.
- **Not transfer-by-learned-fusion of external evidence:** no external features; ME is annotation geometry, GAP is in-model fine — both native inputs.

**Gradient / dead-gate check (H0009 lesson):** at (γ,β)=(0,0), `∂L/∂γ = ∂L/∂h ⊙ h` and `∂L/∂β = ∂L/∂h`; `h = block(x)` is generically nonzero, so gradients are **nonzero from the first backward** — the last Linear escapes zero iff the loss rewards capacity scaling. No path opens before learning; learning sets the strength of a fixed affine family (FiLM), never a gate on/off.

**Params:** Linear(129→64)=8,320 + Linear(64→256)=16,640 ≈ **24,960** (≪ 32M budget). One non-module flag `use_capfilm`.

### 2.3 Twin pre-emption (novelty gate will probe H0024/H0016 first)

- **vs H0024 CountAnchor (residual scale, likely training N0025):** H0024 multiplies the **post-`out` SimPrior residual** by α (still on the suppressive residual path — leverage says remove residual → +29.9%, so residual-scale is low-leverage by construction). CapFilm **never touches the residual**; it modulates `block` features after cat. Different tensor, different layer, different operator (FiLM affine on 128-ch hidden vs scalar on 64-ch residual), different falsifier (γ capacity-loading + cond-leverage ratio vs alpha CV/wipe median). The two cards test **opposite forks of the closure OR**: H0024 = missing absolute magnitude **on residual**; H0025 = consumption interface **decoder operating point**. If both fail, the OR closes with two measured negatives — paper-grade.
- **vs H0016/H0017 additive energy:** no new projection into cond, no energy substrate, no additive path into `cond_map`. Quiet-null signature (proj norm → 0 on residual) does not apply; engagement is γ spread on the decoder side.
- **vs H0018/H0022/H0023 evidence layer:** no gain form, no sign mask, no substrate swap; `SimPrior.forward` body byte-identical.
- **vs H0010 counttau / H0015 dualcal:** no learned per-image scalar count surrogate, no regime split weights; 128-vector FiLM from GAP+ME with CV≥0.05 gate.
- **vs H0002 gca_cal:** not a global scalar on the output; GCA path untouched; per-channel, per-image, capacity-loaded.
- **vs Family C (gated residual entry):** rejected — still operates on the suppressive cond path (probe: cond is suppressor, sens 1.6) and cannot express absolute magnitude for the S-curve; FiLM on the generator-side block is the high-leverage consumption actuator.
- **vs Family A pure output β (SNDM literal):** output multiply is gray against the post-decoder ban and cannot reshape channels; pre-final-conv FiLM is the mandate-preferred legal placement of the same idea.

If the gate still rules twin, the card dies honestly: §5 fallback is exemplar-distinctness (registered, expected-negative) with a NEW falsifier — never a silent residual-scale re-encode.

### 2.4 Attach points (file:line against live parent `…/N0015_h0014/model.py`)

| # | Site | Change |
|---|---|---|
| 1 | `DensityDecoder.forward` model.py:139–140 | signature `forward(self, x, gamma=None, beta=None)`; insert FiLM on `h = self.block(x)` **before** `self.head`/softplus. `__init__` :129–137 **untouched**. |
| 2 | `CountingHead.__init__` model.py:163 (beside `use_peakcal`) | `self.use_capfilm = _get(cfg, "use_capfilm", False)` — non-module flag, no RNG. |
| 3 | `CountingHead.forward` model.py:186 | before decoder call: if flag and `self.capfilm` exists → `gamma, beta = self.capfilm(fine, bboxes_in, self.S)`; pass `gamma=…, beta=…`. `fine` already in scope (:167); `bboxes_in` in scope (:165). SimPrior cascade :175–185 untouched. |
| 4 | New class `CapFilm` after `PeakCal` model.py:281 | new module body (§2.2). |
| 5 | `Counter.__init__` after PeakCal attach model.py:312 (before temp-pin :315) | `if self.head.use_capfilm: self.head.capfilm = CapFilm(d_fine=D, d_out=D)` — **AFTER every parent module** (append-only RNG, AGENTS rule 13); Linear inits draw RNG only here. |
| 6 | Config `config.toml` | one new line `use_capfilm = true` at run time (`--set` or delta file); seed/config otherwise byte-identical. |

Untouched: `Backbone` :22–48, `FineFuser` :51–74, `ExemplarEncoder` :77–108, `Condenser` :111–125, `DensityDecoder.__init__` :129–137, `GCA` :191–203, `SimPrior` :206–252, `CellCal`/`PeakCal`, temp-pin :315–316, `Counter.forward` GCA bias path :332–335.

### 2.5 Step-0 identity + guards + RNG

- **Flag off:** `gamma=beta=None` → decoder executes parent lines → byte-identical to live N0015 at every step.
- **Flag on at init:** last Linear zero-init ⇒ γ=β=0 exact ⇒ `h*(1+0)+0 == h` (fp16: x*1.0+0.0==x) ⇒ softplus path bit-identical ⇒ step-0 forward equals parent.
- **RNG:** CapFilm constructed at Counter.__init__ end; all parent module draws (backbone, head internals, GCA, SimPrior, CellCal, PeakCal) occur earlier — identical positions. Zero-init on last layer only; earlier Linears draw RNG solely inside CapFilm (appended).
- **Guards:** `area.clamp_min(1)`, `me.clamp(1,1e4)` before log; no exp, no division by learned quantity, no softmax, no temp; no new normalization layers (train/eval agree); FiLM is pure affine — finite for any finite γ,β.
- **Smoke (Coding Agent):** flag-off max-abs-diff vs parent == 0; flag-on-at-init == flag-off == 0; param count delta ≈ 25k; temp still 0.07 pinned; `use_peakcal=true` retained.

## §3 — Why not (every exhausted/rejected family, with the leverage-probe premise)

- **Any more residual/evidence content (H0016/17/18/22/23 + H0024 family):** leverage probe **measures** residual as weak and **net-suppressive** (remove → +29.9% count); six cards confirmed content edits are normalized away (quiet-null) or fail wipe rescue (0/11). CapFilm adds **zero** residual content — it changes how the decoder **consumes** `cat[fine, cond]`, the only path with 10× sensitivity (fine=0 → −91.6%).
- **Family C — gated/restructured cond entry (gated additive vs concat):** still rides the suppressive cond path (cond×2 → −17.6%); cannot inject absolute magnitude for the S-curve (probe: density mean 0.029, sum headroom unused). Inferior: changes interface form without a count-bearing capacity DOF.
- **Family A literal SNDM output β · dens:** post-decoder multiply is gray vs the post-decoder-additive ban; per-image scalar only (no channel selectivity to counter cond-suppression). CapFilm is the mandate-preferred placement (pre-final-conv, per-channel FiLM) of the same self-normalization idea.
- **Post-decoder ADDITIVE (any):** STATE ban (H0016/17 lesson + Batch-4). CapFilm adds nothing after `decoder(...)`.
- **Global calibration scalars / H0002 / H0010 / H0015:** dead; CapFilm’s engagement read requires CV(γ)≥0.05 and capacity loading (mean γ on gt≥300 ≥1.5× mean on gt<50) — unsatisfiable by a scalar.
- **Loss / optimizer / schedule / backbone unfreeze / DDCA / RGA / final-layer readout / temp reopen / shared kernels / pre-condenser gating / transfer-by-fusion / gain-domain masks:** all §11 or standing bans — none present. Loss-invariant is why CapFilm must fix S-curve at consumption (DAN’s regression-to-mean cannot be attacked at the loss).
- **Hybrid / second counting method:** dead (err-err 0.566, every router worse). ME and GAP are not a second method.
- **CACViT encoder-token ME surgery:** encoder touch out of regime; we take only the **geometry formula** as a scalar side-channel into decoder FiLM.
- **Exemplar-distinctness (registered fallback):** encoding-layer; H0019 diversification failed and H0023 showed even phase-correct evidence gains no counts — encoding still delivers into the same magnitude-blind consumption. Held as fallback only if novelty gate kills this card.

## §4 — Predictions, failure modes, mechanism reads (R1–R4)

Verdict context: live parent N0015 v2 **19.3431 / dense 330.81 / sparse 5.814** (bar 5.45 standing miss); calibration ratios above; leverage probe numbers above; H0022/H0023 0/11; H0016/17 quiet-null; H0024 (N0025) training — likely same residual-leverage failure.

- **R1 — FiLM ENGAGES AND LOADS CAPACITY (mechanism read, consumption-side):** on val @ best.pth, (a) CV of per-image `mean_c(γ)` ≥ 0.05 (not collapsed to ~0); (b) `mean_c(γ)` on gt≥300 ≥ **1.5×** `mean_c(γ)` on gt<50 (capacity loading — high-count images run a higher decoder operating point); (c) optional channel split sanity: at least 25% of channels have positive mean γ on the gt≥300 subset (not a uniform dead zero). Flat γ (CV<0.05) or wrong-direction loading REFUTES even if bars flicker.
- **R2 — CALIBRATION ON THE UNDERCOUNT SIDE (the S-curve claim):** pred/gt ratio on **gt≥300** improves from the parent’s ~0.685 (gt138+ decile / mid300+ buckets) to **≥0.75**, and/or mid50–300 ratio rises from 0.86 toward ≥0.90 without sparse ratio (already 1.06 over) worsening past bar. Dense-tail mean |del| must stay **< 330.81** (no blowup). Wipe subset (11 ids): mean ratio need not fully rescue (priced honestly like H0024) but must not regress >0.02.
- **R3 — CONSUMPTION LEVERAGE READ (specific to this card’s claim):** re-run the parent’s cond_map×2 probe on the **trained child** at best.pth (same synthetic B=8 protocol as the leverage probe): parent ratio was cond×2 → count ×0.824 (−17.6%). Child should move |ratio − 1| **closer to 1** (decoder less pathologically suppressive under doubled cond) OR fine×2 sensitivity **up** — at least one of: (i) cond×2 count-ratio in [0.90, 1.10], (ii) mean fine-channel sens ≥ 15.2×1.1. If both unchanged and R1 holds, FiLM engaged but did **not** change consumption dynamics → report as engagement-without-effect (refutes the leverage story even if R2/R4 luck out).
- **R4 — BARS:** final EMA val MAE ≤ 19.0431 **AND** dense-tail gt>500 < 330.81 **AND** sparse gt<50 ≤ 5.45 (same-seed v2 paired vs live N0015: 19.3431 / 330.81 / 5.814).

Failure modes (each an honest refutation):

- **F1 — QUIET NULL:** γ stays ≈0 (MSE never rewards capacity scaling in 32ep) → R1 fails → REFUTE; diagnostic value: consumption FiLM is unnecessary or gradient-starved under joint MSE — successor must be a different consumption actuator, not a γ re-tune.
- **F2 — UNIFORM OVERGAIN / SPARSE BLOWUP:** γ goes positive everywhere → sparse ratio worsens, sparse > 5.45 (or >6.2) → REFUTE; ME/GAP not discriminative for “need more counts here.”
- **F3 — DENSE BLOWUP:** dense-tail ≥ 330.81 or any dense-block pred +>10% vs parent → REFUTE (softplus operating point overshoots on high ME).
- **F4 — ENGAGE BUT WRONG FORK:** R1+R3 hold (γ loads capacity, cond leverage normalizes) but R4 bars fail on level noise / mid AE → REFUTE as booked; note mechanism-true + level-miss under determinism crisis for Lead’s evidence weighting (user ruling pattern).
- **F5 — LATE CROSSOVER:** if curve crosses parent ~ep16 like N0023 → ep24 futility HALT path; verdict = futility refutation + full eval_test; R1–R3 still reported on best.pth.
- **F6 — H0024 COLLATERAL:** if N0025 (H0024) is still running or refuted, do not attribute residual-scale effects to CapFilm; CapFilm child is a **direct child of N0015**, not of N0025.

## §5 — Risks, v2 pair protocol, futility

- **Dominant risk — F1 quiet null** (priced): 32ep may leave γ small; R1 CV gate makes this a clean mechanism refutation, not a mystery. Expected-negative stance recorded in header.
- **Secondary:** (a) ME mis-scale — log+clamp bounds it; (b) GAP collinear with GCA’s n_aux → redundant global signal (risk that FiLM duplicates GCA bias); GCA bias is post-decoder tiny (×0.02) so overlap is weak — R3 distinguishes; (c) FiLM β drift breaking step-0 only after training (fine — identity is at init, not forever); (d) determinism crisis sd≈1.13 — single-seed, mechanism reads weighted; (e) sparse 5.45 standing miss may fail independently → truthful card refutation isolating sparse.
- **Protocol:** SAME-SEED v2 pair — child (`use_capfilm=true` over parent flags, `--set augment=true`) vs live N0015 v2; seed 20260830 FIXED; 32ep/1800s; EMA eval; never cross-protocol; `repro_run --set seed=` FORBIDDEN.
- **Futility (rule 16):** launch `--set futility_bar=19.0431`; ep16 WARN / ep24 HALT (best > 20.00 ⇒ HALT); HALT in (20.0, 21.6) draw-contaminated — arbitrate R1/R2 first. Futility-halted run keeps best.pth; full eval_test + γ histogram + leverage re-probe still valid and mandatory.

## §6 — Line-by-line §11 / prior-hypothesis disambiguation

| Constraint / prior hyp | Why H0025 does not re-encode it |
|---|---|
| §11 decoder-side = live, re-book fresh | This IS the decoder-side booking: only new actuator is pre-final-conv FiLM inside `DensityDecoder.forward`. |
| §11 post-decoder ADDITIVE (any) | No term added to `dens` or `out["density"]`; FiLM is pre-`head`/pre-softplus. |
| §11 global calibration scalars | 128-vector per image; falsifier requires CV≥0.05 + capacity loading — a global scalar cannot pass. |
| §11 pre-condenser exemplar gating | Reads `bboxes` geometry + `fine` only; `e`, Condenser, queries byte-untouched. |
| §11 transfer-by-learned-fusion external evidence | No external features; ME = annotation box areas already in `bboxes_in`. |
| §11 gain-domain masks (H0018/H0022) | No sign test, no mask; SimPrior `ev` path byte-identical. |
| §11 frozen hs(2,3) + cross-attn condenser load-bearing | Both untouched; decoder `in_ch` 192 unchanged. |
| §11 DDCA / RGA / final-layer readout / unfreeze | None; no new spatial summary; no second count head. |
| §11 temp reopen / shared kernels | Temp-pin path unchanged (`use_peakcal=true`); no shared projections (CapFilm has its own Linears, no alias to simprior). |
| H0016/H0017 additive energy quiet-null | No additive path into cond; different failure signature (γ on decoder, not proj norm on residual). |
| H0022/H0023 evidence-layer wipes 0/11 | Never touches evidence sign/substrate/gain. |
| H0024 CountAnchor residual α scale | Opposite fork: residual untouched vs decoder block FiLM; mutually distinct R1 reads (alpha CV/wipe-median vs γ capacity-load + cond-leverage ratio). |
| H0010/H0015/H0002 scalar/regime calibration | No scalar, no regime split, no output affine. |
| H0006/H0007 second grad path | CapFilm does not share weights with simprior/qproj/kproj; gradients flow only through CapFilm’s own MLP + decoder block (parent path also trains as before). |
| Loss/optimizer/schedule invariant | Untouched; FiLM is architecture, not loss. |
| Seed / pluggable / append-only | Seed 20260830; one switch `use_capfilm`; CapFilm built after PeakCal; flag-off ≡ parent. |

Runner parse line (STATE gotcha 2):

1. **H0025** — IF a zero-init count-capacity FiLM conditioning of the DensityDecoder hidden block (`use_capfilm`) that maps GAP(fine) concatenated with the log exemplar-box magnitude embedding ME = mean_K(S²/(w_k·h_k)) (S=input size, each annotation box structurally holds one object) to per-channel gamma,beta and applies h ← h·(1+gamma)+beta on the block output before the final 1×1 head and softplus, while the cond_map concat, the SimPrior residual, and every other parent line stay byte-identical, IN the live N0015 head on FSC147 v2 at seed 20260830, THEN final EMA val MAE drops at least 0.30 below 19.3431 with dense-tail gt>500 below 330.81 and sparse gt<50 at or below 5.45, BECAUSE the leverage probe measures the trained parent as fine=generator and cond/simprior=net-suppressor (fine=0 cuts count 91.6 percent, removing the residual raises count 29.9 percent, cond channel sensitivity is 10x below fine), so six residual-content cards were normalized away or wiped while the measured S-curve miscalibration (gt7–10 ratio 1.40 over to gt138+ ratio 0.685 under, mid50–300 at 0.86 carrying 44 percent of AE) is a missing per-image absolute operating point at the decoder consumption interface — zero-init FiLM injects exactly that capacity-conditioned scale pre-final-conv (SNDM’s self-normalization moved inside the decoder, CACViT’s magnitude made geometric), letting the same parent features integrate to different sums without any post-decoder additive and without touching the suppressive residual. DISPROVED IF final EMA val MAE is not at least 0.30 below 19.3431, or dense-tail gt>500 is not below 330.81, or sparse gt<50 exceeds 5.45, or the FiLM fails its engagement read (val coefficient of variation of mean_c(gamma) below 0.05, i.e. no per-image spread, or mean_c(gamma) fails to load capacity: mean on gt≥300 not at least 1.5× mean on gt<50), or the consumption-side leverage read moves the wrong way (cond_map×2 count-ratio after training not closer to 1.0 than the parent’s 0.824 in absolute value).

## Verification record (fill as gate proceeds)

- novelty gate: **PENDING** → `local/research/h0025_novelty.json`
- booking: (Lead books via `discovery hypo --new --solo` on the live parent — Idea Agent does NOT book)
- coding: (delta + CPU checks — Coding Agent; up to 3 fix retries)
- launch: `run_node N00XX_h0025 --parent N0015_h0014 --hyp … --set augment=true --set futility_bar=19.0431`

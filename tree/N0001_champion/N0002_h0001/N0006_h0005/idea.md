# N0006_h0005 — idea (Idea Agent, H0005 use_verify)

- **Node:** N0006_h0005 · **Parent:** N0002_h0001 · **Booked hypothesis:** H0005 · **Switch:** `use_verify` (child `true`; parent default `false`) · **Composition:** SOLO single-switch.
- **Falsifier bar:** final EMA val MAE ≤ 22.26 at seed 20260830 under the canonical 32ep/1800s protocol (live parent 22.5641 @ep30; confirm iff ≥0.30 lower AND gt>500 dense-image mean |Δ| reduced below parent 511.7 baseline).
- **Booked hypothesis text (verbatim — do NOT edit):**
- IF zero-init repetition-gated proposal-verification residual via use_verify IN a solo single-switch child of live parent N0002_h0001 trained under the canonical protocol at seed 20260830 for 32ep/1800s with the module constructed AFTER all parent modules with append-only RNG and decoder in_ch untouched at 20k added params, THEN final EMA val MAE falls at least 0.30 below live parent 22.5641 to 22.26 or lower AND gt>500 dense-image mean |Δ| falls below the parent 511.7 baseline, BECAUSE a dense proposal head multiplied by a sigmoid verification gate conditioned on fine features plus per-cell exemplar repetition suppresses smeared over-smooth blobs where repetition is low and sharpens surviving peaks where repetition is high, restoring small-object recall that MSE-blurred cond mass collapses on dense images with flag-off byte-identical forward and zero-init out giving all-zero residual at step 0, DISPROVED IF final EMA val MAE exceeds 22.26 under the canonical protocol at seed 20260830 AND gt>500 dense-image mean |Δ| is not reduced below the parent 511.7 baseline.

---

# H0005 — Repetition-gated proposal-verification residual (`use_verify`)

- **Hypothesis id:** H0005 · **Child node:** N0006_h0005 under N0002_h0001 · **Switch:** `use_verify` (config default `false`) · **Composition:** SOLO single-switch.
- **One-line intent:** Add a zero-init proposal×verification gate, conditioned on fine features plus per-cell exemplar repetition, as an additive residual into `cond_map` so smeared blobs are suppressed where repetition is low and surviving peaks sharpen where repetition is high.

## 1. Web-survey summary (RULE 15 — code repos preferred; fetched 2026-09-20)

- **UpCount 2026 — proposal-verification head with repetition signal** · Wijaya et al., arXiv:2607.16826 · https://github.com/r28112072-rgb/upcount · **confidence H.** Reference-free head: proposal P=Softplus(Phi_prop(Fcount)), verification V=sigmoid(Phi_ver(Fcount||Srep)), final D=c·(P⊙V⊙A), where Srep is a non-local top-K repetition signal identifying repeated patterns; FSC-147 test 12.39 MAE. Borrowed: the multiplicative P×V gating conditioned on a repetition signal. Not borrowed: the ViT-B/16 + DPT + FeatUp encoder and the reference-free setting — ours is an exemplar-conditioned frozen-backbone head, so the gate must be a pluggable residual on `cond_map` (decoder in_ch untouched), not a new encoder.
- **CoDi 2025 — dense small-object regions need crisp high-frequency maps** · Sustar et al., arXiv:2512.20153 · https://github.com/gsustar/CoDi · **confidence H.** Core finding for us: density regressors get totals right but localize poorly (blurry blobs), while point detectors run out of queries on very dense images; latent diffusion wins via narrow kernels + iterative refinement (−15% MAE few-shot). Borrowed: the sharpening prior — crisp narrow peaks over dense small-object regions. Not borrowed: diffusion steps — we do it in one feed-forward gate (P×V residual), affordable under the 1800 s budget.
- **DM-Count — MSE-on-blurred-GT is the over-smoothing source** · Wang et al., NeurIPS 2020 · https://github.com/cvlab-stonybrook/DM-Count · **confidence H.** Gaussian smoothing of dots hurts generalization; pixel-independent losses encourage blurry maps; OT+TV distribution matching localizes dense regions better (toy PSNR/SSIM, UCF-QNRF localization). Borrowed: justification that our MSE-trained `cond` mass is smeared by construction, so the fix must be spatial gating, not global calibration.
- **SANet — density sharpness lives at the decoder interface** · Cao et al., ECCV 2018 · **confidence M.** High-res DME (transposed convs to input size) alone −26 MAE vs MCNN baseline; Euclidean-only loss gives blurry maps, SSIM local-consistency loss sharpens. Borrowed: attach the gate residually into `cond_map` right before the decoder (in_ch stays 192) — the cheapest place where per-cell sharpness reaches the density output.

## 2. Multi-angle reasoning

### (a) Pure-mathematics lens — what P×V computes that a linear residual cannot

The parent's `cond_map` enters the decoder concatenated with `fine`; any linear residual (e.g. H0001-style projection) can only shift mass additively per channel and cannot express "keep this peak, kill that blob" conditioned on local evidence. P×V is a different functional: a **state-dependent multiplicative gate**. P proposes non-negative density-shaped mass from `cond_map` itself (two-layer 64→32→64 MLP per cell with GELU), V ∈ (0,1)^64 is a sigmoid mask conditioned on the concatenation of the projected fine query q (32ch) and the 2ch repetition evidence [top1, consensus]. The residual P⊙V is therefore a rank-1-style per-cell AND gate: large only where BOTH the condenser proposes structure AND the frozen-geometry repetition evidence agrees. At step 0, prop2 is zero-init so the residual is exactly 0 and V=0.5 everywhere (ver2 bias 0); learning then moves V away from 0.5 only where the evidence pays. This is the UpCount P×V factorization transplanted from a full encoder into a 20k-param pluggable residual: same gating algebra, none of the encoder cost.

### (b) Champion-lineage lens — file:line facts from `tree/N0001_champion/N0002_h0001/model.py`

- **Condenser output (`model.py:121-125,165-168`):** `cond` is a 3-key cross-attention mix reshaped to (B,64,96,96); H0001 already added dense cosine evidence into it, but the result is still consumed *linearly* by the decoder (`cat([fine, cond_map])`, line 173, in_ch 192). Nothing in the lineage multiplies condenser mass by a local confidence — the design cell "gated condenser readout" is empty.
- **Failure evidence:** N0002 heatmaps show dense small-object recall collapse (val[229] 2092→95, val[437] 885→52), blurry-blob predictions vs sharp GT dots, plus one bimodal over-prediction (val[535] 907→1397). STATE diagnostic: tail-heavy error, gt>500 mean |Δ| ≈ 511.7, and global affine calibration has ≤0.34 headroom — a single global scale cannot fix collapse in one image without blowing up the over-prediction in another. The fix must be per-cell, and per-cell evidence (fine q + exemplar repetition S) already exists in this file's tensors.
- **Attachment consequence:** because `fine` (B,128,96,96), `e` (B,K=3,256), and `cond_map` (B,64,96,96) are all live at lines 163-168, the gate reuses them read-only: q from `fine`, S from cosine(q,e) with the H0001 temperature pattern, P from `cond_map`, V from cat(q,ev). No new backbone taps, no change to `e`/`fine`, no touch to GCA/XScale/hs(2,3).

### (c) Counter-intuitive / low-cost lens

- **Counter-intuitive:** adding a *multiplicative suppressor* to fix *under-counting* looks backwards — collapse suggests we need more mass, not a gate that kills mass. But the collapse mechanism is smearing: MSE-blurred `cond` mass merges neighboring dots into one blob whose integral undercounts (two dots → one hump of area ~1.2 instead of 2). A repetition-gated sharpener kills the saddle between dots (low repetition) while keeping the peaks (high repetition), so the decoder sees two separable humps and integrates closer to 2. Suppression of blur IS recall restoration — the same reason CoDi's narrow kernels beat blurry regressors on >300-object images.
- **Low-cost:** ≈19.7k params (≈0.06% of the 32M cap, ≈3% of the ≈630k headroom): qproj 4,096 + kproj 8,192 + temp 1 + prop1 2,080 + prop2 2,112 + ver1 1,120 + ver2 2,112 = 19,713. One 1×1-conv-scale forward at 96×96 (~0.1 GFLOPs, epoch impact <1%, no `_BudgetStop` risk). Zero-init prop2 gives step-0 forward bit-identical to the parent; flag-off skips the line entirely (safe ablation: `false` reproduces N0002 exactly).
- **Cheap falsification:** one canonical run decides it; the dual bar (global MAE ≤22.26 AND dense-tail mean |Δ| below 511.7) separates "gate helped where it claims" from "lucky global shift". A negative with V stuck at ≈0.5 or prop2 norm ≈0 cleanly closes "P×V gating of cond mass helps" without disturbing any other lineage question.

## 3. Exact implementation spec for the Coding Agent (ONE targeted change → H0005)

**Single change:** new module `Verify(nn.Module)` + one gated residual line. No other edits to parent modules, shapes, or defaults.

```python
class Verify(nn.Module):
    def __init__(self, d_fine=128, d_model=256, d_q=32, cond_dim=64, d_h=32, n_ev=2):
        super().__init__()
        self.qproj = nn.Conv2d(d_fine, d_q, 1, bias=False)  # 128 -> 32
        self.kproj = nn.Linear(d_model, d_q, bias=False)    # 256 -> 32
        self.temp = nn.Parameter(torch.tensor(0.07))        # learnable divisor, clamp >= 1e-3
        self.prop1 = nn.Conv2d(cond_dim, d_h, 1)            # 64 -> 32
        self.prop2 = nn.Conv2d(d_h, cond_dim, 1)            # 32 -> 64, ZERO-INIT
        self.ver1 = nn.Conv2d(d_q + n_ev, d_h, 1)           # 34 -> 32
        self.ver2 = nn.Conv2d(d_h, cond_dim, 1)             # 32 -> 64, bias-0 init
        nn.init.zeros_(self.prop2.weight); nn.init.zeros_(self.prop2.bias)
        with torch.no_grad():
            self.ver2.weight.zero_() if hasattr(self.ver2.weight, 'zero_') else nn.init.zeros_(self.ver2.weight)
            nn.init.zeros_(self.ver2.bias)  # V = sigmoid(0) = 0.5 at step 0
    def forward(self, fine, e, cond_map):
        q = F.normalize(self.qproj(fine), dim=1)            # (B,32,96,96)
        p = F.normalize(self.kproj(e), dim=-1)              # (B,K,32)
        S = torch.einsum('bchw,bkc->bkhw', q, p)            # (B,K,96,96)
        W = (S / self.temp.clamp_min(1e-3)).softmax(dim=1)  # over K
        top1 = S.max(dim=1, keepdim=True).values
        cons = (W * S).sum(dim=1, keepdim=True)
        ev = torch.cat([top1, cons], dim=1)                 # (B,2,96,96)
        P = self.prop2(F.gelu(self.prop1(cond_map)))       # (B,64,96,96), all-zero at init
        V = torch.sigmoid(self.ver2(F.gelu(self.ver1(torch.cat([q, ev], dim=1)))))
        return P * V                                        # all-zero at init via prop2
```

**Constructor order (append-only RNG, AGENTS rule 13):** construct `Verify` in `Counter.__init__` AFTER every parent module (after `self.gca` / SimPrior wiring — i.e., the last `nn.Module` construction in `__init__`) so all parent init draws keep exact RNG positions. Forward contains no stochastic ops.

**Exact attachment point** (in `CountingHead.forward`, after the H0001 simprior lines at `model.py:171-172`, before decoder line 173): inputs reused read-only (`fine`, `e` as-is, `cond_map` (B,64,96,96)); `cond_map = cond_map + self.verify(fine, e, cond_map)` gated by `if _get(cfg, "use_verify", False):`. Decoder call unchanged, **in_ch still 192**. Flag off → parent forward byte-for-byte.

**Config key and default:** `use_verify = false` in `config.toml` (mirroring `use_simprior` style, read via `_get(cfg, "use_verify", False)`). Child sets `use_verify = true`; parent config untouched.

**Param delta arithmetic:** qproj 128·32=4,096; kproj 256·32=8,192; temp 1; prop1 64·32+32=2,080; prop2 32·64+64=2,112; ver1 34·32+32=1,120; ver2 32·64+64=2,112. **Total Δ = 19,713 (≈19.7k, booked as 20k).** Parent total ≈31.345M (N0002: ≈31.32M + 24.8k simprior) → child total ≈31.345M + 0.020M ≈ **31.37M ≤ 32M**, using ≈3% of the ≈630k headroom. No other module changes size.

**Expected FLOPs change** (per 384px image): qproj 128·32·96·96 ≈ 37.7M MACs; einsum ≈ 0.9M; prop pair ≈ 64·32·96·96 + 32·64·96·96 ≈ 37.7M; ver pair ≈ 34·32·96·96 + 32·64·96·96 ≈ 28.9M. **Total ≈ 105M MACs (≈0.1 GFLOPs)**; epoch-time impact <1%; memory + a few 96×96 maps (negligible on 12 GB).

**RNG-order statement:** only new RNG consumption is `Verify` parameter init, appended strictly after all parent draws; forward is deterministic (no dropout); data-loader seeding (seed=20260830, augment=false) unchanged. Parent-vs-child divergence comes only from the learned gate (AGENTS §5 rule 13 paired-contrast semantics, precision ~0.02).

**Register compliance (AGENTS §11):** no DDCA/dilated branch; no extra spatial summaries/RGA; no final-layer readout; no unfreeze; no pre-condenser exemplar gating (no channel/per-exemplar scalar gate on `e` or `fine` — pure additive P×V residual into post-Condenser `cond_map`; `e`/`fine` read-only). GCA/XScale/hs(2,3) intact; contract `build_model(cfg) → forward(imgs,bboxes[,bboxes3]) → {"density",...}` unchanged, only `out["density"]` feeds the loss.

## 4. Falsification reading

- **Confirm/refute rule:** run N0006_h0005 at canonical protocol (seed 20260830, 32ep/1800s, EMA eval) as a same-seed paired contrast against live parent N0002 (22.5641 @ep30). **Confirm** H0005 iff child final EMA val MAE ≤ 22.26 (≥0.30 lower) AND gt>500 dense-image mean |Δ| is reduced below the parent 511.7 baseline. Otherwise **refute** — including a child that matches the parent (gate ignored) or regresses. No cross-seed comparison, no re-run shopping (AGENTS §5 rules 1/5/6).
- **What to inspect in the TB curve:** (i) best-epoch EMA val MAE and epoch (expect ≤22.26 at or before ep32, not an ep32-edge fluke); (ii) dense-tail slice (gt>500 mean |Δ| vs 511.7) — the mechanism's claimed territory; (iii) val RMSE/MAE ratio (a drop concentrated in RMSE implicates catastrophic-image fixes, consistent with blob-splitting); (iv) gate diagnostics — prop2 norm (nonzero iff the gate is used) and V histogram (should polarize away from 0.5 on dense images; stuck at 0.5 means the optimizer discarded the gate).
- **What a negative means:** a clean negative falsifies "repetition-gated P×V sharpening of cond mass restores dense small-object recall on frozen DINOv3 features". It does NOT falsify UpCount's P×V factorization in general (their encoder is trained end-to-end; ours gates frozen-geometry evidence) — it points at: (1) pooled-`e` repetition too degenerate for small exemplars (then a finer key bank is the next test, new hypothesis); (2) cosine anisotropy lacking contrast even with learned temperature (then metric/distribution calibration is next); (3) the 0.30 bar living outside dense-tail images (then per-image dump redirects the portfolio). Per AGENTS §11, a refuted H0005 is never silently retried — any revisit books as new evidence with a new falsifier.

## Booked hypothesis set (machine-readable)

1. **H0005** — IF zero-init repetition-gated proposal-verification residual via use_verify IN a solo single-switch child of live parent N0002_h0001 trained under the canonical protocol at seed 20260830 for 32ep/1800s with the module constructed AFTER all parent modules with append-only RNG and decoder in_ch untouched at 20k added params, THEN final EMA val MAE falls at least 0.30 below live parent 22.5641 to 22.26 or lower AND gt>500 dense-image mean |Δ| falls below the parent 511.7 baseline, BECAUSE a dense proposal head multiplied by a sigmoid verification gate conditioned on fine features plus per-cell exemplar repetition suppresses smeared over-smooth blobs where repetition is low and sharpens surviving peaks where repetition is high, restoring small-object recall that MSE-blurred cond mass collapses on dense images with flag-off byte-identical forward and zero-init out giving all-zero residual at step 0, DISPROVED IF final EMA val MAE exceeds 22.26 under the canonical protocol at seed 20260830 AND gt>500 dense-image mean |Δ| is not reduced below the parent 511.7 baseline.

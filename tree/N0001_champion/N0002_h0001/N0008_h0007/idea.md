# N0008_h0007 — idea (Idea Agent, H0007 use_simcal)

- **Node:** N0008_h0007 · **Parent:** N0002_h0001 (live parent 22.5641 @ep30) · **Booked hypothesis:** H0007 · **Switch:** `use_simcal` (child `true`; parent default `false`) · **Composition:** SOLO single-switch (inherits `use_simprior=true` ON; only `use_simcal` toggles vs parent).
- **Falsifier bar:** confirm iff child final EMA val MAE ≤ **22.26** at seed 20260830 under the canonical 32ep/1800s protocol (≥0.30 below live parent 22.5641) AND gt>500 dense-image mean |Δ| below the parent 511.7 baseline; otherwise refute. Same-seed paired contrast only (precision ~0.02); no cross-seed comparison, no re-run shopping.
- **Booked hypothesis text (verbatim — do NOT edit):** see machine-readable section at end (single runner-parseable line).

---

## 1. Web-survey note (AGENTS rule 15 — survey first; reasoning decides)

Fresh web checks 2026-09-20 plus the two repo-local reports this node is booked from (`tree/N0001_champion/N0002_h0001/N0006_h0005/synthesis.md` §4 Booking 2; `local/research/fallback_verify_next.md` Direction 2):

- **QK-Norm — Query-Key Normalization, cosine attention + learnable scale** · https://arxiv.org/abs/2010.04245 · repo https://github.com/CyndxAI/QKNorm · +0.928 BLEU avg over 5 low-resource pairs; ablation: dropping the learnable scale `g` hurts most (softmax over [-1,1] cannot ignore anything). Mechanism: L2-normalize Q,K along head dim (dot → cosine in [-1,1]) then scale up by learnable `g` instead of dividing by √d. Lesson for us: our simprior softmax-over-K with scalar temp init 0.07 inherits the same saturation problem; the fix is de-mean/re-center first, then let a learnable scale act on true margins — not a smaller temp on raw offsets.
- **Anisotropy / QK-interaction in ViTs (Xu et al., NeurIPS 2024)** · https://doi.org/10.48550/arxiv.2405.14880 · analysis paper (no single repo; code pattern = SVD of WqᵀWk per head). Mechanism: embeddings are anisotropic (cone effect); random-pair cosine > 0, contrast compressed; DINO-family late layers drift off similar-token grouping. Lesson: frozen DINOv3 hs(2,3) similarities carry a per-exemplar spatial baseline that must be explicitly removed per cell before temperature — motivates per-exemplar μ_k/σ_k calibration alongside raw sim.
- **CACViT — magnitude + scale calibration and background suppression (AAAI 2024)** · https://doi.org/10.1609/aaai.v38i6.28396 · repo https://github.com/Xu3XiWang/CACViT · FSC147 3-shot val 10.63 / test 9.13; magnitude embedding −6.2 val / −7.35 test RMSE; background suppression −15.4% MAE / −23.1% RMSE (B6→B7). Mechanism: softmax normalization wipes absolute mass — pair any temperature fix with a magnitude-bearing channel so calibration does not fight the count anchor. Lesson: full spatial softmax would destroy count scale and fight GCA; keep one magnitude-weighted consensus channel.
- **DINOv3 — Gram anchoring; dense cosine maps degrade with long training** · https://arxiv.org/html/2508.10104 · repo https://github.com/facebookresearch/dinov3 · late-training patch-CLS cosine rises, patch-patch maps go noisy; Gram-matrix anchoring to an early teacher repairs locality. Lesson: frozen DINOv3-ConvNeXt-Tiny dense cosine geometry is usable but noisy/biased — a statistics-only calibration (no new matcher) is the proportionate response, not a bigger bank.
- **Repo-local Direction 2 draft** · `local/research/fallback_verify_next.md:26-36` (proposed `use_simcal`: reuse S, μ_k/σ_k, SNR weights, magnitude-weighted consensus, ~1–3k params) and **N0006 Booking 2** · `tree/N0001_champion/N0002_h0001/N0006_h0005/synthesis.md:38-40` (the exact H0007 text booked). This idea.md implements that booking; it does not re-argue the portfolio order (finer bank H0006 fires on the sibling node; high-res `use_hires` deferred per N0006 §4).

"SOTA did it" is not the mechanism (§6): the survey informs the calibration shape; first-principles (§2a) and the N0006 temp-collapse lead (§2b) decide.

## 2. Multi-angle reasoning

### (a) Pure-mathematics lens — what calibration computes that a smaller temp cannot

Raw simprior volume `S[k,h,w] = <q(h,w), p_k>` lives in [-1,1] with an anisotropic offset: for fixed exemplar k, `S_k = m_k + d_k(h,w)` where `m_k = E_hw[S_k]` is the image- and exemplar-dependent cone baseline (typically > 0 on DINOv3) and `d_k` is the true margin field. Softmax-over-K with divisor temp `t`, `W_k ∝ exp(S_k / t)`, is translation-invariant only across K at fixed (h,w) — it is NOT invariant to per-k spatial offsets `m_k`: a large `m_k` with flat `d_k` (unreliable exemplar, background-like) still wins mass against a smaller-`m_j` exemplar with sharp `d_j`. Shrinking `t` sharpens `W` around whoever has the largest offset, not whoever has the best peak — it amplifies baseline luck.

Per-exemplar spatial de-meaning removes the offset: `Z_k = (S_k − μ_k)/(σ_k+ε)` with `μ_k, σ_k` spatial moments is unit-free shape evidence (peaks vs surroundings for that exemplar), while `s_k = μ_k/(σ_k+ε)` is its signal-to-noise ratio. Weighting `w = softmax(τ·s)` over K (single learnable `τ`) selects the exemplar whose field is peaky, not the one with the highest background cosine. The pair `[Σ_k w_k S_k, Σ_k w_k Z_k]` then spans the two things the decoder needs: a magnitude-bearing consensus (absolute scale preserved — CACViT lesson; full spatial softmax would wipe it and fight the GCA anchor) and a calibrated margin field on which temperature acts truthfully. No new matcher is introduced: the operation is reductions over the existing (K=3,96,96) volume plus one 2→64 projection.

### (b) Champion-lineage lens — file:line facts plus the N0006 temp-collapse lead

- **Simprior readout (parent `model.py:193-221`):** `q = norm(qproj(fine))`, `p = norm(kproj(e))`, `S = einsum(q,p)` (B,K=3,96,96), `W = softmax(S/t)`, `ev = [top1, consensus]`, `out` 2→64 zero-init, residual `cond_map + simprior(fine,e)` (`model.py:171-172`), decoder `in_ch` still 192 (`model.py:156,173`). Working point: N0002 synthesis §1 causal — `out` norm 0.5874 off zero-init, both channels equal, `temp` 0.07→0.0393 (learned sharpening that helped: 23.293→22.5641, 2.43× the 0.30 bar).
- **N0006 temp-collapse lead (`feedback/causal.md:19,36` + `synthesis.md:11`):** the verify gate's own 32-dim cosine path learned `temp` 0.07→**0.0214** (÷3.3, 2.5× peakier than the working simprior temp 0.0546), with prop2 norm 2.24 + ver2 3.86 out-shouting the confirmed simprior term (0.96) while MAE went +1.016 the wrong way. Causal path §3: over-sharpened K=3 verification on lower-fidelity cosine turns small noise into hard per-cell V decisions (V polarized, mean 0.5435, only 29.9% near 0.5). Synthesis §2 remaps this explicitly as the lead for calibration framing, not a standalone finding. Reading: temperature *did* move — on the wrong quantity (raw offsets + noise). Calibration is the upstream fix: de-mean first so the learned scale acts on true margins `d_k`, SNR-weight so peaky exemplars dominate, magnitude-consensus so scale survives.
- **Why not the sibling bank:** N0006 synthesis §4 orders finer-bank first, calibration second, on the theory that pooled-`e` is degenerate for small exemplars. That ordering is about portfolio attribution, not exclusivity: the bank attacks key coarseness, calibration attacks comparison dynamics on the keys we already trust. Both are fresh ids with new falsifiers (dedup §3 of that synthesis); this node tests only the calibration half, reusing `S` read-only, adding statistics not matchers.

### (c) Counter-intuitive / low-cost lens

- **Counter-intuitive:** the parent already *has* a learned temperature (0.0393) that helped — adding "more calibration" looks like tuning a tuned knob. It is not tuning: the knob currently scales `m_k + d_k` jointly and cannot separate them by construction (scalar division commutes with the offset). De-meaning changes the functional (offset-free input), SNR-weighting changes the selection rule (peaky-over-loud); the scalar `τ` then learns a different quantity (SNR sharpness, not cosine sharpness). If anisotropy is not the binding constraint the module has a clean null: weights collapse to uniform, `τ` stays near init, `out` norm stays ≈0 — which reads as mechanism rejection, not under-tuning.
- **Low-cost:** ≈0.2k actual params (two scalars + one zero-init 2→64 proj) inside the booked 3k envelope — ≈0.03% of the ≈655k headroom; reductions-only FLOPs (~5M MACs + 1.2M out-proj, <0.01 GFLOPs, no second qproj, no full-grid 75-key einsum that killed H0004 on budget); zero-init so step-0 forward is bit-identical to N0002 (safe ablation: `false` reproduces the parent exactly, `true` starts from the parent and learns only calibration trust). One canonical run decides it.

## 3. Exact implementation spec for the Coding Agent (ONE targeted change only)

**New module class** (name fixed): `SimCal(nn.Module)`. No other new module; no edit to `SimPrior`, `Condenser`, `FineFuser`, `ExemplarEncoder`, `DensityDecoder`, `GCA`, or `Backbone` constructors except wiring.

```python
class SimCal(nn.Module):
    def __init__(self, cond_dim=64, n_ev=2, eps=1e-5):
        super().__init__()
        self.tau = nn.Parameter(torch.tensor(1.0))  # SNR sharpness (softmax scale over K)
        self.eps = eps
        self.out = nn.Conv2d(n_ev, cond_dim, 1)     # 2 -> 64, ZERO-INIT (see below)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
    def forward(self, S):
        # S: (B,K=3,96,96) cosine volume recomputed read-only via simprior projections (below)
        mu = S.mean(dim=(2, 3), keepdim=True)                        # (B,K,1,1)
        sd = S.std(dim=(2, 3), keepdim=True, unbiased=False)         # (B,K,1,1)
        snr = (mu / (sd + self.eps)).squeeze(-1).squeeze(-1)         # (B,K)
        w = (snr * self.tau).softmax(dim=1).view(-1, 3, 1, 1)        # (B,K,1,1)
        Z = (S - mu) / (sd + self.eps)                               # (B,K,96,96) de-meaned
        c_mag = (w * S).sum(dim=1, keepdim=True)                     # (B,1,96,96) magnitude-weighted consensus
        c_cal = (w * Z).sum(dim=1, keepdim=True)                     # (B,1,96,96) SNR-weighted calibrated margin
        ev = torch.cat([c_mag, c_cal], dim=1)                        # (B,2,96,96)
        return self.out(ev)                                          # (B,64,96,96), all-zeros at init
```

**How to obtain S read-only (no new qproj/kproj):** inside `CountingHead.forward`, after the existing `cond` line, recompute `S` functionally with the *existing* simprior weights — `q = F.normalize(self.simprior.qproj(fine), dim=1); p = F.normalize(self.simprior.kproj(e), dim=-1); S = torch.einsum('bchw,bkc->bkhw', q, p)` — without constructing any new projection and without modifying `SimPrior`. (Equivalent alternative: expose `S` from `SimPrior.forward`; either satisfies read-only reuse as long as no new q/k parameters are created and `SimPrior` weights/shapes are untouched — state the chosen path in a code comment.)

**Exact attachment (in `CountingHead.forward`, after the `use_simprior` residual at `model.py:171-172`, before the decoder call):**

```python
if _get(cfg, "use_simcal", False):
    # S reused read-only from simprior projections; SimCal adds statistics, not matchers.
    cond_map = cond_map + self.simcal(S)
```

- `use_simprior` stays `true` (inherited from N0002); the simprior lines are untouched. Flag-off (`use_simcal=false`) skips the line above and the forward is the parent's byte-for-byte.
- **Explicit non-violation statement:** decoder `in_ch` stays 192 (still `cat([fine, cond_map])`); no parent module shape/init is touched; `e` and `fine` are read-only (no gating/rescaling — pure additive residual into `cond_map` post-Condenser); this is query/similarity-side distribution calibration consumed additively at the decoder interface, NOT pre-condenser exemplar gating in any granularity; no Condenser QK-scale edit (the learnable `tau` above is the single QK-Norm-style scale, inside SimCal, not a second change).

**Constructor order (append-only RNG, AGENTS rule 13):** construct `SimCal` in `Counter.__init__` **after** `self.head.simprior` (i.e., after every parent module: `Backbone` → `CountingHead(fuser, exemplar, cond, decoder)` → `GCA` → `SimPrior`), attached to the head (`self.head.simcal = SimCal(...)`). It must be the last `nn.Module` construction in `__init__` so all parent init draws keep exact RNG positions. `SimCal`'s own init draws are only the zeroed `out` (no randomness) plus scalar init — forward has no stochastic ops (reductions/softmax/einsum/norm deterministic), data seeding (`seed=20260830`, `augment=false`) unchanged.

**Zero-init plan:** `self.out` weight and bias exact zeros; `tau` init 1.0 (uniform `w` at step 0). Consequence: at step 0 `delta == 0` regardless of `S`, so the child's forward (`density`, `n_aux`) is numerically identical to N0002 at init; the switch learns only calibration trust. Lineage precedent: GCA zero-init final layer (`model.py:183-184`) and SimPrior zero-init `out` (`model.py:209-210`).

**Config key and default:** `use_simcal = false` in `config.toml` (mirroring `use_simprior` style, read via `_get(cfg, "use_simcal", False)`). Child sets `use_simcal = true` and keeps `use_simprior = true`; parent config untouched.

**Register compliance (AGENTS §11):** no DDCA/extra summaries/RGA/final-layer readout/unfreeze (backbone stays `requires_grad_(False)`, `net.eval()`); no pre-condenser exemplar gating (no channel/per-exemplar scalar gate on `e`/`fine`); GCA/XScale/hs(2,3) readout intact; contract `build_model(cfg) → forward(imgs,bboxes[,bboxes3]) → {"density",...}` unchanged, only `out["density"]` feeds loss.

## 4. Param delta accounting + FLOPs + RNG

- `tau`: 1 scalar.
- `out` Conv2d(2→64,1, with bias): 2·64 + 64 = 192.
- **Actual Δ = 1 + 192 = 193 params (≈0.2k), inside the booked "at 3k" envelope (upper bound, not exact).** No qproj/kproj (reused read-only), no Condenser edit.
- Head ≈3.529M → ≈3.529M + 0.0002M; node total ≈31.345M + 0.0002M ≈ **31.345M ≤ 32M** (backbone 27,820,128 frozen), using <0.1% of the ≈655k headroom. No other module changes size.
- **FLOPs** (per 384px image): μ/σ reductions over 3×96×96 ≈ 55k elems; Z/W/c_mag/c_cal elementwise ≈ 100k; out-proj 2·64·96·96 ≈ 1.2M MACs; S recompute reuses simprior einsum (no second qproj matmul beyond the parent's — the qproj 1×1 at 96×96 is already paid by simprior). **Marginal ≈ 1–2M MACs (≈0.002 GFLOPs)**, epoch-time impact ≪1%, no `_BudgetStop` risk (contrast H0004's 75-key full-grid blow-up).
- **RNG-order statement:** only new RNG consumption is `SimCal` init (zeroed `out` + scalar, appended strictly after all parent draws); forward deterministic; loader seeding unchanged. Parent-vs-child divergence under the canonical harness comes only from learned calibration (AGENTS §5 rule 13 paired-contrast semantics, precision ~0.02).

## 5. Falsification reading

- **Confirm/refute rule:** run N0008_h0007 at canonical protocol (seed 20260830, 32ep/1800s, EMA eval) as same-seed paired contrast vs live parent N0002_h0001 (22.5641 @ep30). **Confirm** H0007 iff child final EMA val MAE ≤ 22.26 (≥0.30 lower) AND gt>500 dense-image mean |Δ| below 511.7. Otherwise **refute** — including child ≈ parent (calibration ignored) or regressing. Global-bar miss alone decides per N0006 precedent (dense conjunct unevaluable from run scalars alone is noted, not a rescue).
- **What to inspect in the TB curve + checkpoint:** (i) best-epoch EMA val MAE and epoch (expect ≤22.26 at or before ep32, not edge-only); (ii) `tau` trajectory (stuck ≈1.0 + uniform `w` → anisotropy not binding); (iii) `simcal.out` weight norm (stays ≈0 → optimizer discarded calibration; mechanism rejected, not under-tuned); (iv) per-exemplar μ_k/σ_k spread and `w` entropy (collapsed-uniform vs peaky-selective); (v) train/val divergence (parent overfits ep30→32; calibration must move val, not just train); (vi) val RMSE/MAE ratio (drop concentrated in RMSE as well implicates dense-image fixes, consistent with sharper per-cell selection).
- **What a negative means:** clean negative (child ≈ parent or worse, `out` norm ≈0 or `w` uniform) falsifies "anisotropic-baseline removal + SNR re-weighting of the pooled-K=3 similarity volume adds counting signal beyond raw simprior on frozen DINOv3". It does NOT falsify finer banks (H0006 sibling) or decoder-side fixes — it points at: (1) pooled `e` too degenerate for any K-normalization to save (then bank lineage decides); (2) cosine margins already sufficient at temp 0.0393 (then resolution/decoder family is next); (3) the 0.30 bar lives in catastrophic images calibration cannot reach (then per-image dump redirects). Per AGENTS §11, a refuted H0007 is never silently retried — any revisit books new evidence with a new falsifier.

## Booked hypothesis set (machine-readable)

1. **H0007** — IF similarity-distribution calibration via use_simcal IN a solo single-switch child of live parent N0002_h0001 trained under the canonical protocol at seed 20260830 for 32ep/1800s with the module constructed AFTER all parent modules with append-only RNG and decoder in_ch untouched at 3k added params, THEN final EMA val MAE falls at least 0.30 below live parent 22.5641 to 22.26 or lower AND gt>500 dense-image mean |Δ| falls below the parent 511.7 baseline, BECAUSE per-exemplar spatial de-meaning plus signal-to-noise re-weighting of the reused simprior similarity volume removes the anisotropic cosine baseline so the learned temperature acts on true margins while the magnitude-weighted consensus preserves absolute count scale that full spatial normalization destroys, DISPROVED IF final EMA val MAE exceeds 22.26 under the canonical protocol at seed 20260830 AND gt>500 dense-image mean |Δ| is not reduced below the parent 511.7 baseline.

# N0002_h0001 — idea (Idea Agent, H0001 simprior)

- **Node:** N0002_h0001 · **Parent:** N0001_champion · **Booked hypothesis:** H0001 · **Switch:** `use_simprior` (child `true`; parent default `false`) · **Composition:** SOLO single-switch.
- **Falsifier bar:** final val MAE ≤ 22.99 at seed 20260830 under the canonical 32ep/1800s protocol (live parent 23.293 @ep26; confirm iff ≥0.30 lower).
- **Booked hypothesis text (verbatim — do NOT edit):**
- IF dense exemplar-similarity prior readout IN N0002_h0001 child of N0001_champion with use_simprior=true, THEN final val MAE drops at least 0.30 below live parent 23.293, BECAUSE explicit per-cell cosine match evidence over K=3 exemplar embeddings supplies spatially sharp class-conditioned signal that 3-key cross-attention condensation smooths away and lets the decoder place density mass on true instances, DISPROVED IF final val MAE at seed 20260830 under 32ep/1800s protocol is not at or below 22.99.

---

# H0001 — Dense exemplar-similarity prior readout (`use_simprior`)

- **Hypothesis id:** H0001 · **Child node:** N0002_h0001 under N0001_champion · **Switch:** `use_simprior` (config default `false`) · **Composition:** SOLO single-switch.
- **One-line intent:** Inject an explicit, spatially sharp per-cell exemplar-match prior into the decoder path by reading cosine similarity between the fine map and the K=3 exemplar embeddings, compressed through a zero-init projection added into `cond`.

## 1. Multi-angle reasoning

### (a) Pure-mathematics lens — what functional the similarity prior computes that 3-key attention cannot

The Condenser's per-pixel conditioning is a 3-way convex combination: 9216 queries at 1/4 res attend to K=3 pooled keys (`model.py:121-125`), so for pixel `i` the output is `Σ_k α_{ik}(q_i·k_k) v_k` with `Σ_k α=1`. The only per-pixel degree of freedom is the 3-logit vector `q_i·k_k`; the value side carries no spatial information beyond those 3 vectors, and the softmax + `out` Linear(256→64) further smooths whatever sharpness the logits had. The output is therefore a *low-rank, globally mixed* conditioning field: two pixels with the same 3-logit direction get near-identical conditioning even if one sits on a true instance and the other on background texture.

The similarity prior computes a different functional: a **pointwise, non-convex, exemplar-anchored evidence field** `S[k,h,w] = <q(h,w), p_k>` over the full 96×96 grid, followed by order statistics over K (`max_k`, softmax-weighted consensus) that are *not* representable as a convex combination of 3 values. `max_k` is a piecewise selector (sharp boundaries, sub-exemplar localization); the softmax-over-K margin (`max − second-max`) measures decision confidence per cell, which no attention weight vector exposes. Adding this evidence additively into `cond` gives the decoder a channel that says "this cell looks like exemplar 2 and nothing else" with pixel-level sharpness, while the Condenser's implicit match says "this cell's query is 60/30/10 over exemplars". The two are complementary by construction: one is an explicit normalized readout of frozen-feature geometry, the other a learned soft mixer. (Essence mapping: first-principles §5 subproblem (1), matching/evidence; `local/research/first_principles.md` S3 diagnoses the same 3-key bottleneck from the capacity side.)

### (b) Champion-lineage lens — file:line facts from `tree/N0001_champion/model.py`

- **ExemplarEncoder pooling (`model.py:91-108`):** exemplars are ROI-aligned from h3@1/16 (`roi_align(feat, rois, 7×7)`, line 97), projected 384→256, shape-biased, passed through a 2-layer transformer, then **softmax-pooled over 49 tokens to one vector per exemplar** (`a = self.attn(tok).softmax(1); out = (tok*a).sum(1)`, lines 102-103) plus the XScale coarse average (lines 104-107). The object that reaches the head is `e` of shape (B,K=3,256) — three pooled vectors. All part-level spatial detail inside the exemplar is destroyed here; the Condenser can never recover it.
- **Condenser 3-key attention (`model.py:112-125`):** `tok = proj_in(fmap)` (128→256, line 122), `attn(norm1(tok), e, e)` with K=3 keys (line 123), FFN + `out` 256→64 (line 125). Every one of the 9216 pixel queries selects among the same 3 keys. Attention entropy over exemplars is therefore near image-global whenever exemplars look alike — the matcher has no way to say "pixel (61,44) matches exemplar 2's corner" because corners no longer exist downstream of pooling.
- **Decoder input (`model.py:156,166`):** `DensityDecoder(in_ch = D + cond_dim = 128+64 = 192)` consumes `cat([fine, cond])` (line 166); GCA reads `GAP(fine)+e_mean` only (`model.py:179-183`) and XScale/hs(2,3) readout stay intact. The two proven-positive lineage facts are XScale coarse exemplar enrichment (+0.95, README) and GCA (+1.6); both are global/coarse. Nothing in the champion exposes **dense, per-cell, exemplar-indexed** evidence to the decoder — that cell of the design space is empty, and it is exactly where frozen-backbone counting must win (survey F1: frozen features need explicit readout, not representation learning).
- **Attachment consequence:** because `e` already exists with the right semantics (post-transformer, post-XScale pooled class vectors), the cheapest high-leverage readout reuses `e` verbatim as the similarity bank and `fine` (B,128,96,96) as the query field — no new backbone taps, no change to `e` itself, no touch to GCA/XScale/hs(2,3).

### (c) Counter-intuitive / low-cost lens

- **Counter-intuitive:** the head already *has* a matcher (Condenser cross-attention), so adding "another matcher" looks redundant. It is not: the Condenser is a learned mixer trained end-to-end through the density loss, so its attention is free to encode anything that minimizes MSE (including count-scale shortcuts); an explicit cosine readout with its own normalization is pinned to geometry and cannot drift into a scale shortcut. Training-free CountingDINO proves the point from the other side: normalized similarity maps on frozen DINO features alone reach 25.48/20.93 val/test with zero training — the signal is real before any learning happens.
- **Low-cost:** ≈24.8k params (≈0.08% of the 32M cap, ≈3.6% of the ≈680k headroom), one 1×1 conv + one Linear + one 1×1 proj, ~0.08 GFLOPs/forward (epoch-time impact <1%, no `_BudgetStop` risk against the 106 s headroom), zero-init so step-0 forward is bit-identical to the parent (safe ablation: `false` reproduces the champion exactly, `true` starts from the champion and learns the prior's trust).
- **Cheap falsification:** one canonical run decides it; a negative (no ≥0.30 gain with the prior's projection norm staying near zero or the evidence channels ignored) cleanly closes "explicit similarity helps a 3-key condenser" without disturbing any other lineage question.

## 2. Web-survey citations (RULE 15 — satisfied by the two completed reports; no new fetches needed)

- **CountingDINO — training-free normalized similarity readout on frozen DINO features** · https://arxiv.org/abs/2504.16570 (arXiv:2504.16570) · **confidence H.** ROI-align prototypes used as conv kernels → similarity maps → normalization + background filter + divide-et-impera multiscale; val 25.48 / test 20.93 with a ~300M DINOv2 ViT-L and *no training at all*. Direct proof that normalized cosine-similarity maps over frozen DINO geometry carry counting signal. Survey §2 row 2; F1 attachment notes (`local/research/survey_mechanisms.md:50-64`).
- **SAFECount — score-map normalization across exemplar AND spatial dims** · https://arxiv.org/abs/2201.08959 (arXiv:2201.08959) · **confidence H.** Dense per-pixel score maps normalized over K and over space (val 15.28 / test 14.32, ResNet-18). Its normalization half is the direct precedent for our softmax-over-K + per-exemplar spatial handling; its FE (similarity-weighted query enhancement) half is the banned pre-condenser gating family and is explicitly NOT reused. Survey §2 row 3, §4 caution.
- **CACViT — magnitude/scale calibration + background suppression by similarity** · https://arxiv.org/abs/2305.04440 (arXiv:2305.04440) · **confidence H.** 3-shot val 10.63 / test 9.13 with ViT-B; ablates magnitude embeddings (B3) and similarity-based background suppression (B7). Precedent that raw similarity needs calibration (temperature/scale) and that positive-only matching leaves false positives — bounds what H0001 claims (matching evidence only, no suppression; suppression is a separate future switch). Survey §2 row 6, F1/F4 notes.
- **Survey F1 attachment notes** · `local/research/survey_mechanisms.md:50-64` (family F1, switch `use_simprior`, ~25k params, `use_simprior=false` reproduces champion bit-exactly) · **confidence H** (interface reasoning on our exact hs(2,3)+e shapes, not measured gain). Chinook: failure modes listed there (1/16 ROI noise, DINOv3 anisotropy → learned scale, normalization wiping count scale → GCA interaction, top-1 vs softmax-K ablation) are inherited as H0001's known risks.
- **First-principles error analysis** · `local/research/first_principles.md:190-194` (S3: 3-key bottleneck), `:178-188` (S1: error mass concentrated, 0.30 bar = 386 counts), `:232-236` (test order F1 first) · supporting, **confidence M** (S1/S3 inferred from RMSE/MAE ratio 3.90 and code anatomy, not from a per-image dump D1).
- **Contrast (what H0001 is not):** LOCA OPE adaptation (https://arxiv.org/abs/2211.08217) modifies `e` pre-Condenser (register edge, survey F3); LSSD FEM uses banned channel/spatial-summary attention (survey §4); BMNet+ learned bilinear metric underperforms fixed cosine + adaptation (LOCA Table 1) — hence H0001 uses fixed cosine geometry with a learned temperature, not a learned metric.

## 3. Exact implementation spec for the Coding Agent

**New module class** (name fixed): `SimPrior(nn.Module)`.

```python
class SimPrior(nn.Module):
    def __init__(self, d_fine=128, d_model=256, d_proj=64, cond_dim=64, n_ev=2):
        super().__init__()
        self.qproj = nn.Conv2d(d_fine, d_proj, 1, bias=False)   # 128 -> 64
        self.kproj = nn.Linear(d_model, d_proj, bias=False)     # 256 -> 64
        self.temp = nn.Parameter(torch.tensor(0.07))            # learnable temperature (divisor)
        self.out = nn.Conv2d(n_ev, cond_dim, 1)                 # 2 -> 64, ZERO-INIT (see below)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
    def forward(self, fine, e):
        # fine: (B,128,96,96); e: (B,K=3,256)
        q = F.normalize(self.qproj(fine), dim=1)                # (B,64,96,96)
        p = F.normalize(self.kproj(e), dim=-1)                  # (B,K,64)
        S = torch.einsum('bchw,bkc->bkhw', q, p)                # (B,K,96,96) cosine in [-1,1]
        W = (S / self.temp.clamp_min(1e-3)).softmax(dim=1)      # softmax over K
        top1 = S.max(dim=1, keepdim=True).values                # (B,1,96,96)
        cons = (W * S).sum(dim=1, keepdim=True)                 # (B,1,96,96) softmax-weighted consensus
        ev = torch.cat([top1, cons], dim=1)                     # (B,2,96,96)
        return self.out(ev)                                     # (B,64,96,96), all-zeros at init
```

**Constructor order (append-only RNG, AGENTS rule 13):** construct `SimPrior` in `Counter.__init__` **after** `self.gca` (i.e., after every parent module: `Backbone` → `CountingHead(fuser, exemplar, cond, decoder)` → `GCA`). It must be the last `nn.Module` construction in `__init__` so all parent init draws keep their exact RNG positions. `SimPrior`'s own init draws (`qproj`/`kproj` default init) come last and affect nothing else.

**Exact attachment point and tensor shapes/ops** (in `CountingHead.forward`, after `cond = self.cond(fmap, e)` at `model.py:165`, before `model.py:166`):
- Inputs reused verbatim: `fine` (B,128,96,96) = FineFuser output; `e` (B,K=3,256) = ExemplarEncoder output **as-is** (post-transformer, post-XScale pooling — read-only, never modified, never re-pooled).
- `cond` is (B,9216,64) token sequence from the Condenser; reshape to (B,64,96,96) exactly as line 166 does.
- `delta = simprior(fine, e)` → (B,64,96,96); `cond_map = cond_map + delta` (residual add); decoder call becomes `self.decoder(torch.cat([fine, cond_map], 1))` with **in_ch still 192** — the decoder's shape and init are untouched.
- Gating: `if _get(cfg, "use_simprior", False): cond_map = cond_map + self.simprior(fine, e)` — with the flag off the forward is the parent's line 166 byte-for-byte. `SimPrior` may live in `CountingHead` (constructed after `self.decoder`, i.e., after all parent submodules) or in `Counter`; either satisfies append-only order as long as it is constructed after every parent module — state the chosen location in the code comment.
- **Explicit non-violation statement:** this change does NOT add decoder input channels (decoder stays 192ch), does NOT change the shape or init of any parent module (`FineFuser`, `ExemplarEncoder`, `Condenser`, `DensityDecoder`, `GCA` constructors untouched), does NOT modify `e` or `fine` (no multiplicative gating, no rescaling — pure additive residual into `cond`), and is therefore NOT pre-condenser exemplar gating in any granularity: it is a post-exemplar query/similarity-side readout consumed additively at the decoder interface.

**Normalization choice and why:** softmax over K with a single learnable scalar temperature (init 0.07, clamped ≥1e-3), following SAFECount's across-exemplar normalization and CACViT's magnitude-calibration lesson; per-exemplar spatial normalization (SAFECount's second axis) is deliberately omitted in v1 to keep the evidence count-scale-bearing (full spatial softmax would wipe absolute mass and fight the GCA count anchor — survey F1 failure mode (c)). Evidence pair `[top1, softmax-consensus]` covers both sharp detection (top-1) and multi-exemplar agreement (consensus); the margin-to-second variant is left as a follow-up ablation, not co-composed (one hypothesis per switch).

**Zero-init plan:** `self.out` (final 1×1 conv 2→64) has weight and bias initialized to exact zeros (`nn.init.zeros_`). Consequence: at step 0 `delta == 0` regardless of `qproj`/`kproj`/`temp` values, so the child's forward output (`density`, `n_aux`) is **numerically identical to the parent's** at init; the switch learns only the trust weight of the prior. `qproj`/`kproj` keep default init (their effect is gated by the zero proj). The existing GCA zero-init final layer (`model.py:176-177`) is the lineage precedent for this pattern.

**Config key and default:** `use_simprior = false` in `config.toml` (mirroring `use_xscale`/`use_gca` style, read via `_get(cfg, "use_simprior", False)`). Child sets `use_simprior = true`; parent config untouched.

**Param delta arithmetic** (bias-free projections + biased zero-init out + 1 scalar):
- `qproj` Conv2d(128→64,1, no bias): 128·64 = 8,192
- `kproj` Linear(256→64, no bias): 256·64 = 16,384
- `temp`: 1
- `out` Conv2d(2→64,1, with bias): 2·64 + 64 = 192
- **Total Δ = 8,192 + 16,384 + 1 + 192 = 24,769 (≈24.8k).** Head total 3,504,643 → ≈3.529M; node total ≈31.32M + 0.025M ≈ **31.345M ≤ 32M**, using ≈3.6% of the ≈680k headroom. No other module changes size.

**Expected FLOPs change** (per 384px image, forward): qproj 128·64·96·96 ≈ 75.5M MACs; einsum K·64·96·96 ≈ 1.8M; softmax over K≈3 negligible; out-proj 2·64·96·96 ≈ 1.2M. **Total ≈ 78–80M MACs (≈0.08 GFLOPs)** — dominated by one 1×1 conv at 1/4 res; backbone + head are orders of magnitude larger. Expected epoch-time impact <1% (≈1 s of the 106 s headroom: 1800−1694 s); no `_BudgetStop` risk. Backward cost is proportional; memory +2 activation maps at 96×96 (negligible on 12 GB).

**RNG-order statement:** the only new RNG consumption is `SimPrior` parameter init, appended strictly after all parent module init draws; forward contains no stochastic ops (no dropout, softmax/einsum/norm are deterministic); data-loader seeding (`seed=20260830`, `augment=false`) is unchanged. Hence parent-vs-child training divergence under the canonical harness comes only from the learned prior, satisfying AGENTS §5 rule 13 paired-contrast semantics (precision ~0.02).

**Register compliance (AGENTS §11):** no DDCA/dilated branch (FineFuser `ctx` untouched, `use_ddca=false` stays); no extra spatial summaries/RGA (no pooling-summary tokens, no channel gate); no final-layer readout (hs(2,3) taps unchanged); no unfreeze (backbone stays `requires_grad_(False)`, `Backbone.train` keeps `net.eval()`); no pre-condenser exemplar gating (no channel/per-exemplar scalar gate on `e` or `fine` — the prior is additive into `cond` post-Condenser). GCA/XScale/hs(2,3) readout stay intact; contract `build_model(cfg) → forward(imgs,bboxes[,bboxes3]) → {"density",...}` unchanged and only `out["density"]` feeds the loss.

## 4. Exact one-line hypothesis text (to be booked verbatim by the Lead)

IF dense exemplar-similarity prior readout IN N0002_h0001 child of N0001_champion with use_simprior=true, THEN final val MAE drops at least 0.30 below live parent 23.293, BECAUSE explicit per-cell cosine match evidence over K=3 exemplar embeddings supplies spatially sharp class-conditioned signal that 3-key cross-attention condensation smooths away and lets the decoder place density mass on true instances, DISPROVED IF final val MAE at seed 20260830 under 32ep/1800s protocol is not at or below 22.99.

## 5. Falsification reading

- **Confirm/refute rule:** run N0002_h0001 at canonical protocol (seed 20260830, 32ep/1800s, EMA eval) as a same-seed paired contrast against the live parent (val MAE 23.293 @ep26). **Confirm** H0001 iff child final val MAE ≤ 22.99 (≥0.30 lower). Otherwise **refute** — including a child that matches the parent (prior ignored) or regresses. No cross-seed comparison, no re-run shopping (AGENTS §5 rules 1/5/6).
- **What to inspect in the TB curve:** (i) best-epoch EMA val MAE and its epoch (parent: 23.293 @ep26; expect the child to beat 22.99 at or before ep32, not merely at ep32 edge); (ii) val RMSE/MAE ratio (parent ≈3.90 — a drop concentrated in RMSE as well as MAE implicates catastrophic-image fixes, consistent with a sharp match prior); (iii) `train/loss_epoch` vs val MAE divergence (parent overfits ep26→32; the prior must move val, not just train); (iv) the learned `out`-proj norm and `temp` trajectory — a confirmed win should show nonzero proj norm with `temp` settled away from init; a refute with proj norm ≈0 means the optimizer discarded the prior (mechanism rejected, not just under-tuned).
- **What a negative means for the mechanism:** a clean negative (child ≈ parent or worse, prior norm →0 or evidence ignored) falsifies "explicit dense cosine evidence over pooled K=3 embeddings adds counting signal beyond 3-key cross-attention on frozen DINOv3 features". It does NOT falsify similarity readouts in general — it points at the known failure modes in order: (1) h3-pooled `e` is too degenerate a bank for small exemplars (then the fix is a finer bank, e.g. dense exemplar keys — a different hypothesis); (2) cosine on anisotropic DINOv3 features lacks contrast even with learned temperature (then metric/distribution calibration F4 is the next test); (3) the 0.30 bar lives in catastrophic images the prior cannot reach (then S1/D1 per-image dump redirects the portfolio). Per AGENTS §11, a refuted H0001 is never silently retried — any revisit books as new evidence with a new falsifier.

## Booked hypothesis set (machine-readable)

1. **H0001** — IF dense exemplar-similarity prior readout IN N0002_h0001 child of N0001_champion with use_simprior=true, THEN final val MAE drops at least 0.30 below live parent 23.293, BECAUSE explicit per-cell cosine match evidence over K=3 exemplar embeddings supplies spatially sharp class-conditioned signal that 3-key cross-attention condensation smooths away and lets the decoder place density mass on true instances, DISPROVED IF final val MAE at seed 20260830 under 32ep/1800s protocol is not at or below 22.99.
